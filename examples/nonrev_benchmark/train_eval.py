# -*- coding: utf-8 -*-
"""
train_eval.py — system-agnostic ISOKANN training + scoring harness for the
non-reversible-target benchmark.

One `train_chi(system, variant, seed)` entry point trains a ChiNetMultiLinear via either
the isotarget fixed-point loop (variants: isa, gramschmidt, schurisa, gpcca) or subspace
power iteration (svd_power), caching results under runs/. The training loop is the same
one used by benchmark_v3 / benchmark_v4 (GramSchmidt warm-up -> variant target, Adam,
plateau early-stop, best-on-validation checkpoint); the only additions are the two
non-reversible variants (via nonrev_targets.PipelineTarget) and per-iteration collection
of their feasibility diagnostics.

Scoring follows the established conventions:
  * shape  : Hungarian-matched |Pearson r| of chi vs continuous references (v3/v4 style)
  * fate   : Hungarian-matched AUROC of chi vs discrete state labels (cr2_benchmark style)
  * Spearman, per-mode SD, k_eff (SD>0.05) reported alongside (feedback_sd_metric)
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from amore.isotarget import apply_target, gramschmidt_target, shiftscale  # noqa: E402
from amore.isokann import ChiNetMultiLinear, power_method_multi          # noqa: E402
from amore.scrna.koopman import init_from_warmup                         # noqa: E402
from nonrev_targets import PipelineTarget, NONREV_VARIANTS               # noqa: E402

DEVICE = torch.device("cpu")
K = 3

VARIANTS = ["isa", "gramschmidt", "svd_power", "schurisa", "gpcca"]
LABELS = {
    "isa": "ISA (reversible)", "gramschmidt": "GramSchmidt (reversible)",
    "svd_power": "SVD-Power (reversible)",
    "schurisa": "Schur-ISA (non-rev)", "gpcca": "GPCCA (non-rev)",
}
# The simplex-based variants (ISA and the two non-reversible Schur transforms) use the
# cr2_benchmark / v4-ssm_isa warm-up: train a genuine 1-D ShiftScale committor first, then
# seed every output row of the k-D net from it (init_from_warmup) so ISA deflation starts
# from the dominant slow coordinate. GramSchmidt is its own target and needs no warm-up;
# SVD-Power is power iteration.
ISA_FAMILY = {"isa", "schurisa", "gpcca"}

def _warmup_1d(f0, fts, tr, IN, hidden, warmup_iters, lr, grad_clip, seed):
    """Train a 1-D ShiftScale ISOKANN committor (the classical k=1 warm-up) and return it.
    kchi is the burst average of the warm net at the burst endpoints (1, n_tr)."""
    torch.manual_seed(seed * 12345 + 7)
    warm = ChiNetMultiLinear(IN, 1, hidden=hidden).to(DEVICE)
    optw = torch.optim.Adam(warm.parameters(), lr=lr)
    for _ in range(warmup_iters):
        warm.eval()
        with torch.no_grad():
            kc = np.mean([warm(ft[tr]).cpu().numpy().T for ft in fts], axis=0)   # (1, n_tr)
        tt = torch.tensor(shiftscale(kc).T, dtype=torch.float32, device=DEVICE)  # [0,1]
        warm.train()
        loss = nn.functional.mse_loss(warm(f0[tr]), tt)
        optw.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(warm.parameters(), grad_clip); optw.step()
    return warm


# ── config ─────────────────────────────────────────────────────────────────
class Cfg:
    MAX_ITER = 1200
    MIN_ITER = 400
    W        = 250
    REL_TOL  = 1e-3
    WARMUP   = 100
    LR       = 1e-3
    GRAD_CLIP = 5.0
    HIDDEN   = [128, 32, 8]
    POWER_N_ITER = 80
    POWER_EPOCHS = 50
    NONREV_MAXITER = 60     # inner L-BFGS budget for the Schur transforms (warm-started)


# ── helpers ────────────────────────────────────────────────────────────────
def _plateau(vh, it, cfg):
    if it < cfg.MIN_ITER or len(vh) < cfg.W:
        return False
    r = np.array(vh[-cfg.W:]); r = r[np.isfinite(r)]
    return len(r) >= cfg.W // 2 and (r.max() - r.min()) < cfg.REL_TOL * max(abs(np.median(r)), 1e-12)

def _chi_sd(c):
    return c.std(0)


# ── isotarget training ──────────────────────────────────────────────────────
def _run_isotarget(system, variant, seed, cfg) -> dict:
    torch.manual_seed(seed * 12345 + 7); np.random.seed(seed * 12345 + 7)
    F0     = system["feat"]
    BURSTS = system["bursts"]                       # (N, Kb, F)
    N, Kb, IN = BURSTS.shape

    f_all = torch.tensor(F0, device=DEVICE)
    f0    = torch.tensor(F0, device=DEVICE)
    fts   = [torch.tensor(BURSTS[:, j, :], device=DEVICE) for j in range(Kb)]

    m = np.random.rand(N) < 0.8
    tr, te = np.where(m)[0], np.where(~m)[0]

    net = ChiNetMultiLinear(IN, K, hidden=cfg.HIDDEN).to(DEVICE)
    # 1-D ShiftScale committor warm-up -> seed the k-D net (cr2_benchmark / v4-ssm_isa style)
    if variant in ISA_FAMILY:
        warm_net = _warmup_1d(f0, fts, tr, IN, cfg.HIDDEN, cfg.WARMUP, cfg.LR, cfg.GRAD_CLIP, seed)
        init_from_warmup(warm_net, net, K, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.LR)

    nonrev = PipelineTarget(variant, maxiter=cfg.NONREV_MAXITER) if variant in NONREV_VARIANTS else None

    def kchi(sub):                                  # mean over bursts of chi(endpoint), (k, |sub|)
        net.eval()
        with torch.no_grad():
            return np.mean([net(ft[sub]).cpu().numpy().T for ft in fts], axis=0)

    def target(chi0, kc):
        if variant == "gramschmidt":
            return gramschmidt_target(kc)
        if nonrev is not None:
            return nonrev(chi0, kc)
        return apply_target("isa", chi0, kc)

    vh, sdh, diagh = [], [], []
    best_val, best_chi = np.inf, None
    t0 = time.perf_counter()

    for it in range(cfg.MAX_ITER):
        net.eval()
        with torch.no_grad():
            chi0 = net(f0[tr]).cpu().numpy().T
        kc = kchi(tr)
        try:
            tgt = target(chi0, kc)
            if nonrev is not None and nonrev.last_diag is not None:
                d = nonrev.last_diag
                diagh.append([it, float(np.abs(np.imag(d["eigs"])).max()),
                              d["min_membership_before_proj"], d["min_membership_after_proj"],
                              d["condA"], d["gap"]])
        except (ValueError, np.linalg.LinAlgError):
            vh.append(np.nan)
            with torch.no_grad():
                sdh.append(_chi_sd(net(f_all).cpu().numpy()))
            continue

        tt = torch.tensor(tgt.T, dtype=torch.float32, device=DEVICE)
        net.train()
        loss = nn.functional.mse_loss(net(f0[tr]), tt)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.GRAD_CLIP); opt.step()

        # validation: GramSchmidt target gives a cheap, deterministic "is chi still
        # moving" signal used only for plateau stop + best-checkpoint selection. (A
        # fresh Schur L-BFGS per val step would dominate runtime and add no signal.)
        net.eval()
        kc_te = kchi(te)
        try:
            tgt_te = gramschmidt_target(kc_te)
            with torch.no_grad():
                val = float(nn.functional.mse_loss(
                    net(f0[te]), torch.tensor(tgt_te.T, dtype=torch.float32, device=DEVICE)))
        except (ValueError, np.linalg.LinAlgError):
            val = np.nan
        vh.append(val)
        with torch.no_grad():
            chi_all = net(f_all).cpu().numpy()
        sdh.append(_chi_sd(chi_all))
        if np.isfinite(val) and val < best_val:
            best_val, best_chi = val, chi_all.copy()
        if _plateau(vh, it, cfg):
            break

    if best_chi is None:
        with torch.no_grad():
            best_chi = net(f_all).cpu().numpy()
    return dict(chi_best=best_chi.astype(np.float32),
                sd_history=np.array(sdh, np.float32),
                val_loss=np.array(vh, np.float32),
                diag=np.array(diagh, np.float64) if diagh else np.zeros((0, 6)),
                elapsed=time.perf_counter() - t0, n_iter=len(vh))


# ── power iteration ─────────────────────────────────────────────────────────
def _run_power(system, seed, cfg) -> dict:
    torch.manual_seed(seed * 12345 + 7); np.random.seed(seed * 12345 + 7)
    F0     = system["feat"]
    BURSTS = system["bursts"]
    N, Kb, IN = BURSTS.shape
    net = ChiNetMultiLinear(IN, K, hidden=cfg.HIDDEN).to(DEVICE)
    x0 = torch.tensor(np.tile(F0, (Kb, 1)), device=DEVICE)
    x1 = torch.tensor(np.concatenate([BURSTS[:, j, :] for j in range(Kb)], 0), device=DEVICE)
    t0 = time.perf_counter()
    res = power_method_multi(net, x0, x1, n_iter=cfg.POWER_N_ITER,
                             epochs_per_iter=cfg.POWER_EPOCHS, lr=cfg.LR, verbose=False)
    net.eval()
    with torch.no_grad():
        chi = net(torch.tensor(F0, device=DEVICE)).cpu().numpy().astype(np.float32)
    return dict(chi_best=chi,
                sd_history=(np.array(res["spans"]) / (2 * np.sqrt(3))).astype(np.float32),
                val_loss=np.array(res["losses"], np.float32),
                diag=np.zeros((0, 6)),
                elapsed=time.perf_counter() - t0, n_iter=len(res["losses"]))


# ── top-level dispatcher with caching ───────────────────────────────────────
def train_chi(system, variant, seed, cfg=Cfg, force=False, verbose=True) -> dict:
    out_dir = os.path.join(RUNS, system["tag"], variant, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    cpath = os.path.join(out_dir, "chi_best.npy")
    if os.path.exists(cpath) and not force:
        res = dict(chi_best=np.load(cpath),
                   sd_history=np.load(os.path.join(out_dir, "chi_sd_history.npy")),
                   val_loss=np.load(os.path.join(out_dir, "val_loss.npy")),
                   diag=np.load(os.path.join(out_dir, "diag.npy")),
                   cached=True)
        with open(os.path.join(out_dir, "meta.json")) as f:
            res.update(json.load(f))
        return res

    res = _run_power(system, seed, cfg) if variant == "svd_power" \
        else _run_isotarget(system, variant, seed, cfg)

    np.save(cpath, res["chi_best"])
    np.save(os.path.join(out_dir, "chi_sd_history.npy"), res["sd_history"])
    np.save(os.path.join(out_dir, "val_loss.npy"), res["val_loss"])
    np.save(os.path.join(out_dir, "diag.npy"), res["diag"])
    sdf = res["sd_history"][-1] if len(res["sd_history"]) else np.zeros(K)
    meta = dict(elapsed=res["elapsed"], n_iter=res["n_iter"],
                k_eff=int((sdf > 0.05).sum()), sd_final=[float(s) for s in sdf])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"  [{system['tag']}/{variant}/seed{seed}] iters={res['n_iter']:4d} "
              f"sd=[{', '.join(f'{s:.3f}' for s in sdf)}] k_eff={meta['k_eff']} "
              f"t={res['elapsed']:.0f}s", flush=True)
    res.update(meta); res["cached"] = False
    return res


# =========================================================================== #
# scoring
# =========================================================================== #
def pearson_r(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(a @ b / (na * nb))

def _spearman(a, b) -> float:
    from scipy.stats import rankdata
    return pearson_r(rankdata(a), rankdata(b))

def _auroc(score, pos) -> float:
    """Rank-based AUROC (Mann-Whitney). score: (N,), pos: (N,) bool."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def hungarian_assign(score_matrix):
    """score_matrix (k, R): maximise total over a k->R assignment. Returns (mean, perm)."""
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-np.nan_to_num(score_matrix, nan=0.0))
    return float(np.nanmean(score_matrix[ri, ci])), list(zip(ri.tolist(), ci.tolist()))

def shape_r(chi, refs):
    """Hungarian-matched mean |Pearson r| of chi (N,k) vs refs (R,N). v3/v4 convention."""
    k, R = chi.shape[1], refs.shape[0]
    M = np.array([[abs(pearson_r(chi[:, j], refs[i])) for i in range(R)] for j in range(k)])
    return hungarian_assign(M)

def fate_auroc(chi, labels):
    """Hungarian-matched mean AUROC of chi (N,k) vs discrete labels. cr2 convention."""
    states = np.unique(labels)
    M = np.array([[_auroc(chi[:, j], labels == s) for s in states] for j in range(chi.shape[1])])
    mean, perm = hungarian_assign(M)
    return mean, perm, M

def fate_spearman(chi, refs):
    """Best |Spearman| of any chi column vs each ref, averaged (continuous-shape companion)."""
    return float(np.mean([max(abs(_spearman(chi[:, j], refs[i]))
                              for j in range(chi.shape[1])) for i in range(refs.shape[0])]))

def k_eff(sd_history):
    if len(sd_history) == 0:
        return np.nan
    return int((sd_history[-1] > 0.05).sum())
