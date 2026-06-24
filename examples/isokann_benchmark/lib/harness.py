# -*- coding: utf-8 -*-
"""
harness.py — system-agnostic ISOKANN training + scoring for the consolidated
benchmark. Uses ONLY the methods defined in src/amore (no reimplementation).

Two method families, each with its own NATURAL representation (head + dimension):

  MEMBERSHIP family  — softmax head (ChiNetMulti), k = N_STATES = 3.
      isa, schurisa, gpcca   (inner-simplex / Schur targets, amore.isotarget / nonrev_targets)
      vamp                   (VAMP-2 score ascent = VAMPnets, amore.isotarget.vamp2_score)
    The softmax head architecturally enforces the probability simplex (chi>=0, sum=1),
    so the outputs ARE metastable memberships. It also prevents the amplitude collapse /
    mode-selection that the linear ISA suffered on 231-D ADP — softmax-ISA recovers BOTH
    phi and psi from scratch (no warm-up). k = N_STATES (3 memberships for 3 states).

  BASIS family — linear head (ChiNetMultiLinear), k = N_STATES - 1 = 2, constant deflated.
      gramschmidt, pseudoinv, cross   (orthogonalisation / Rayleigh-Ritz targets; the
                                       Koopman target is mean-centered = constant deflated
                                       so the net learns the 2 NON-TRIVIAL eigenfunctions
                                       {EV2,EV3}, not {const,EV2})
      svd_power                       (subspace power iteration; power_method_multi already
                                       mean-centers internally, so self-deflating)
    3 metastable states span a 3-D slow subspace = constant + 2 non-trivial eigenfunctions,
    so a basis method's natural output is 2 eigenfunctions. The membership view is obtained
    post-hoc by PCCA+ on [const, the 2 eigenfunctions] (see to_memberships).

No warm-up is used in the main pipeline (softmax removes the need). The warm-up machinery
(get_warmup + the head='linear'/warmup=True overrides) is retained ONLY for the
"why softmax" comparison section (linear-ISA collapse vs warm-up band-aid vs softmax-ISA).

Constant across methods within a system: data/anchors/bursts, trunk arch in->[128,32,8],
Tanh hidden, Adam lr=1e-3, grad-clip 5, plateau stop + best-on-held-out checkpoint, 80/20
split. Varied: the method (head+target), the seed, the system.

Caches under paths.RUNS/<system>/<variant>/seed_<s>/ (override head/warm-up -> suffixed dir).
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
from numpy.linalg import qr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
sys.path.insert(0, paths.AMORE_SRC)

import torch
import torch.nn as nn

from amore.isotarget import apply_target, gramschmidt_target, shiftscale, vamp2_score
from amore.isokann import ChiNetMultiLinear, ChiNetMulti, power_method_multi
from amore.scrna.koopman import init_from_warmup
from nonrev_targets import PipelineTarget, NONREV_VARIANTS
from schur_isotargets import _inner_simplex_vertices

N_STATES = 3
MEMBERSHIP_VARIANTS = ["isa", "vamp", "schurisa", "gpcca"]          # softmax head, k=3
BASIS_VARIANTS = ["gramschmidt", "pseudoinv", "cross", "svd_power"]  # linear head, k=2
DEFLATE_VARIANTS = {"gramschmidt", "pseudoinv", "cross"}             # svd_power self-deflates
ALL_VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd_power", "vamp",
                "schurisa", "gpcca"]

LABELS = {
    "isa": "ISA", "gramschmidt": "GramSchmidt", "pseudoinv": "PseudoInv",
    "cross": "Cross", "svd_power": "SVD-Power", "vamp": "VAMP-2",
    "schurisa": "Schur-ISA (non-rev)", "gpcca": "GPCCA (non-rev)",
}


def is_membership(variant):
    return variant in MEMBERSHIP_VARIANTS

def n_out(variant):
    return N_STATES if is_membership(variant) else N_STATES - 1

def default_head(variant):
    return "softmax" if is_membership(variant) else "linear"


# ── per-system config ────────────────────────────────────────────────────────
class Cfg:
    HIDDEN = [128, 32, 8]
    LR = 1e-3
    GRAD_CLIP = 5.0
    MAX_ITER = 1500
    MIN_ITER = 400
    W = 250
    REL_TOL = 1e-3
    WARMUP_ITERS = 400          # only for the legacy linear warm-up comparison
    POWER_N_ITER = 80
    POWER_EPOCHS = 50
    NONREV_MAXITER = 60
    SCHUR_WARM = 200            # Schur-ISA/GPCCA: iters of plain-ISA pre-spread before the
                               # Schur feasibility/crispness target (it refines an ISA pivot,
                               # and breaks the near-uniform softmax symmetry it can't escape).
    MAX_ANCHORS = None


class TWCfg(Cfg):
    MAX_ITER = 1500; MIN_ITER = 400; W = 250; WARMUP_ITERS = 400; POWER_N_ITER = 80


class ADPCfg(Cfg):
    MAX_ITER = 2500; MIN_ITER = 400; W = 250; WARMUP_ITERS = 1200
    POWER_N_ITER = 80; MAX_ANCHORS = 25000; NONREV_MAXITER = 40; SCHUR_WARM = 300


class RingCfg(Cfg):
    MAX_ITER = 1000; MIN_ITER = 300; W = 200; WARMUP_ITERS = 300
    POWER_N_ITER = 60; NONREV_MAXITER = 60


CFG = {"triple_well": TWCfg, "adp_300k_0p1": ADPCfg, "directed_ring": RingCfg}
def get_cfg(tag): return CFG.get(tag, Cfg)


# ── helpers ──────────────────────────────────────────────────────────────────
def _device(use_gpu):
    return torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

def _make_net(variant, IN, cfg, head=None):
    k = n_out(variant); head = head or default_head(variant)
    Net = ChiNetMulti if head == "softmax" else ChiNetMultiLinear
    return Net(IN, k, hidden=cfg.HIDDEN), k, head

def _plateau(vh, it, cfg):
    if it < cfg.MIN_ITER or len(vh) < cfg.W:
        return False
    r = np.array(vh[-cfg.W:]); r = r[np.isfinite(r)]
    return len(r) >= cfg.W // 2 and (r.max() - r.min()) < cfg.REL_TOL * max(abs(np.median(r)), 1e-12)

def _gs_residual(chi_kn, kchi_kn):
    try:
        tgt = gramschmidt_target(kchi_kn)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(np.mean((chi_kn - tgt) ** 2))


class _CrossHist:
    def __init__(self, maxcols=0):
        self.maxcols = maxcols; self.X = None; self.Y = None
    def update(self, chi_kn, kchi_kn):
        Xn, Yn = chi_kn.T.astype(np.float64), kchi_kn.T.astype(np.float64)
        self.X = Xn if self.X is None else np.hstack([self.X, Xn])
        self.Y = Yn if self.Y is None else np.hstack([self.Y, Yn])
        if self.maxcols and self.X.shape[1] > self.maxcols:
            self.X = self.X[:, -self.maxcols:]; self.Y = self.Y[:, -self.maxcols:]
        return self.X, self.Y


# ── representation converters (post-hoc) ─────────────────────────────────────
def eigfns(chi, m=None):
    """The m=N_STATES-1 non-trivial eigenfunctions: top-m left singular vectors of the
    mean-centered output. For a k=2 basis run this is its 2 outputs (orthonormalised);
    for a k=3 membership run it removes the constant dimension and returns the 2 eigfns."""
    m = m or (N_STATES - 1)
    B0 = chi - chi.mean(0)
    U, S, Vt = np.linalg.svd(B0, full_matrices=False)
    return U[:, :m]

def to_memberships(chi, k=None):
    """N_STATES memberships. Softmax outputs are returned as-is; a basis (eigenfunction)
    output is rotated to memberships by PCCA+ (inner-simplex on [const, eigenfunctions])."""
    k = k or N_STATES
    if chi.shape[1] == k and chi.min() > -0.12 and np.abs(chi.sum(1) - 1).mean() < 0.15:
        return chi                                   # already memberships (softmax)
    E = eigfns(chi, k - 1)
    Q, _ = qr(np.column_stack([np.ones(len(E)), E]))
    X = Q[:, :k].copy(); X[:, 0] = 1.0
    idx, A0 = _inner_simplex_vertices(X)
    return X @ A0


# ── legacy 1-D ShiftScale warm-up (only for the "why softmax" comparison) ─────
def _warmup_path(tag, seed):
    d = os.path.join(paths.RUNS, tag, "_warmup"); os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"seed_{seed}.pt")

def get_warmup(system, seed, cfg, device, force=False):
    tag = system["tag"]; wp = _warmup_path(tag, seed); IN = system["feat"].shape[1]
    warm = ChiNetMultiLinear(IN, 1, hidden=cfg.HIDDEN).to(device)
    if os.path.exists(wp) and not force:
        warm.load_state_dict(torch.load(wp, map_location=device)); return warm
    torch.manual_seed(seed * 12345 + 7); np.random.seed(seed * 12345 + 7)
    F0 = system["feat"]; BURSTS = system["bursts"]; N, Kb, _ = BURSTS.shape
    f_all = torch.tensor(F0, device=device); f0 = torch.tensor(F0, device=device)
    fts = [torch.tensor(BURSTS[:, j, :], device=device) for j in range(Kb)]
    m = np.random.rand(N) < 0.8; tr, te = np.where(m)[0], np.where(~m)[0]
    opt = torch.optim.Adam(warm.parameters(), lr=cfg.LR)
    def _kc(sub):
        warm.eval()
        with torch.no_grad():
            return np.mean([warm(ft[sub]).cpu().numpy().T for ft in fts], axis=0)
    wl_tr, wl_val = [], []
    for _ in range(cfg.WARMUP_ITERS):
        kc = _kc(tr); tt = torch.tensor(shiftscale(kc).T, dtype=torch.float32, device=device)
        warm.train(); loss = nn.functional.mse_loss(warm(f0[tr]), tt)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(warm.parameters(), cfg.GRAD_CLIP); opt.step()
        wl_tr.append(float(loss.detach()))
        kc_te = _kc(te); tte = torch.tensor(shiftscale(kc_te).T, dtype=torch.float32, device=device)
        warm.eval()
        with torch.no_grad(): wl_val.append(float(nn.functional.mse_loss(warm(f0[te]), tte)))
    torch.save(warm.state_dict(), wp)
    warm.eval()
    with torch.no_grad(): chi1d = warm(f_all).cpu().numpy().ravel()
    d = os.path.dirname(wp)
    np.save(os.path.join(d, f"seed_{seed}_chi.npy"), chi1d.astype(np.float32))
    np.save(os.path.join(d, f"seed_{seed}_loss.npy"), np.stack([wl_tr, wl_val]).astype(np.float32))
    return warm

def load_warmup_artifacts(tag, seed):
    d = os.path.join(paths.RUNS, tag, "_warmup")
    cp = os.path.join(d, f"seed_{seed}_chi.npy"); lp = os.path.join(d, f"seed_{seed}_loss.npy")
    if os.path.exists(cp) and os.path.exists(lp):
        return np.load(cp), np.load(lp)
    return None, None


# ── isotarget / Schur training (membership softmax OR basis linear) ──────────
def _run_isotarget(system, variant, seed, cfg, device, head=None, warmup=False) -> dict:
    torch.manual_seed(seed * 12345 + 7); np.random.seed(seed * 12345 + 7)
    F0 = system["feat"]; BURSTS = system["bursts"]; N, Kb, IN = BURSTS.shape
    f_all = torch.tensor(F0, device=device); f0 = torch.tensor(F0, device=device)
    fts = [torch.tensor(BURSTS[:, j, :], device=device) for j in range(Kb)]
    m = np.random.rand(N) < 0.8; tr, te = np.where(m)[0], np.where(~m)[0]

    net, k, head = _make_net(variant, IN, cfg, head)
    net = net.to(device)
    if warmup and head == "linear" and variant in ({"isa"} | set(NONREV_VARIANTS)):
        warm = get_warmup(system, seed, cfg, device)
        init_from_warmup(warm, net, k, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.LR)

    deflate = variant in DEFLATE_VARIANTS
    nonrev = PipelineTarget(variant, maxiter=cfg.NONREV_MAXITER) if variant in NONREV_VARIANTS else None
    cross_hist = _CrossHist(maxcols=k * 3) if variant == "cross" else None

    def kchi(sub):
        net.eval()
        with torch.no_grad():
            return np.mean([net(ft[sub]).cpu().numpy().T for ft in fts], axis=0)

    def make_target(chi0, kc, it):
        if deflate:
            chi0 = chi0 - chi0.mean(1, keepdims=True); kc = kc - kc.mean(1, keepdims=True)
        if variant == "gramschmidt":
            return gramschmidt_target(kc)
        if nonrev is not None:
            # Schur-ISA/GPCCA refine an ISA pivot; pre-spread with plain ISA so the Schur
            # coarse-propagator target starts from a non-degenerate (non-uniform) chi.
            if it < cfg.SCHUR_WARM:
                return apply_target("isa", chi0, kc)
            return nonrev(chi0, kc)
        if variant == "cross":
            X, Y = cross_hist.update(chi0, kc)
            return apply_target("cross", chi0, kc, cross_hist=(X, Y))
        return apply_target(variant, chi0, kc)

    loss_tr, loss_val, opt_loss, sdh, diagh = [], [], [], [], []
    best_val, best_chi = np.inf, None
    t0 = time.perf_counter()
    for it in range(cfg.MAX_ITER):
        net.eval()
        with torch.no_grad(): chi0 = net(f0[tr]).cpu().numpy().T
        kc = kchi(tr)
        try:
            tgt = make_target(chi0, kc, it)
            if nonrev is not None and nonrev.last_diag is not None:
                d = nonrev.last_diag
                diagh.append([it, float(np.abs(np.imag(d["eigs"])).max()),
                              d["min_membership_before_proj"], d["min_membership_after_proj"],
                              d["condA"], d["gap"]])
        except (ValueError, np.linalg.LinAlgError):
            opt_loss.append(np.nan); loss_tr.append(_gs_residual(chi0, kc)); loss_val.append(np.nan)
            with torch.no_grad(): sdh.append(net(f_all).cpu().numpy().std(0))
            continue
        tt = torch.tensor(tgt.T, dtype=torch.float32, device=device)
        net.train(); loss = nn.functional.mse_loss(net(f0[tr]), tt)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.GRAD_CLIP); opt.step()
        opt_loss.append(float(loss.detach()))
        net.eval()
        with torch.no_grad(): chi0_tr = net(f0[tr]).cpu().numpy().T
        loss_tr.append(_gs_residual(chi0_tr, kchi(tr)))
        with torch.no_grad(): chi_te = net(f0[te]).cpu().numpy().T
        val = _gs_residual(chi_te, kchi(te)); loss_val.append(val)
        with torch.no_grad(): chi_all = net(f_all).cpu().numpy()
        sdh.append(chi_all.std(0))
        if np.isfinite(val) and val < best_val: best_val, best_chi = val, chi_all.copy()
        if _plateau(loss_val, it, cfg): break
    if best_chi is None:
        with torch.no_grad(): best_chi = net(f_all).cpu().numpy()
    return dict(chi_best=best_chi.astype(np.float32),
                loss_train=np.array(loss_tr, np.float32), loss_val=np.array(loss_val, np.float32),
                opt_loss=np.array(opt_loss, np.float32), sd_history=np.array(sdh, np.float32),
                diag=np.array(diagh, np.float64) if diagh else np.zeros((0, 6)),
                elapsed=time.perf_counter() - t0, n_iter=len(loss_val))


# ── subspace power iteration (basis, linear, self-deflating) ─────────────────
def _run_power(system, seed, cfg, device, head=None, warmup=False) -> dict:
    torch.manual_seed(seed * 12345 + 7); np.random.seed(seed * 12345 + 7)
    F0 = system["feat"]; BURSTS = system["bursts"]; N, Kb, IN = BURSTS.shape
    net, k, _ = _make_net("svd_power", IN, cfg, head); net = net.to(device)
    x0 = torch.tensor(np.tile(F0, (Kb, 1)), device=device)
    x1 = torch.tensor(np.concatenate([BURSTS[:, j, :] for j in range(Kb)], 0), device=device)
    t0 = time.perf_counter()
    res = power_method_multi(net, x0, x1, n_iter=cfg.POWER_N_ITER,
                             epochs_per_iter=cfg.POWER_EPOCHS, lr=cfg.LR, verbose=False)
    net.eval()
    with torch.no_grad(): chi = net(torch.tensor(F0, device=device)).cpu().numpy().astype(np.float32)
    losses = np.array(res["losses"], np.float32)
    return dict(chi_best=chi, loss_train=losses, loss_val=np.full_like(losses, np.nan),
                opt_loss=losses, sd_history=(np.array(res["spans"]) / (2 * np.sqrt(3))).astype(np.float32),
                diag=np.zeros((0, 6)), elapsed=time.perf_counter() - t0, n_iter=len(losses))


# ── VAMP-2 (VAMPnets: softmax head, k=3, no warm-up) ─────────────────────────
def _run_vamp(system, seed, cfg, device, head=None, warmup=False) -> dict:
    torch.manual_seed(seed * 12345 + 7); np.random.seed(seed * 12345 + 7)
    F0 = system["feat"]; BURSTS = system["bursts"]; N, Kb, IN = BURSTS.shape
    f_all = torch.tensor(F0, device=device); f0 = torch.tensor(F0, device=device)
    bursts_t = torch.tensor(BURSTS, device=device)
    m = np.random.rand(N) < 0.8; tr, te = np.where(m)[0], np.where(~m)[0]
    net, k, head = _make_net("vamp", IN, cfg, head); net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.LR)
    linear = head == "linear"
    def feat(c):                                   # linear head needs a guard against collapse
        return c / c.std(0, keepdim=True).clamp(min=0.05) if linear else c
    def pairs(sub):
        return f0[sub].repeat_interleave(Kb, 0), bursts_t[sub].reshape(-1, IN)
    x0tr, x1tr = pairs(tr); x0te, x1te = pairs(te)
    loss_tr, loss_val, sdh = [], [], []
    best_val, best_chi = np.inf, None; t0 = time.perf_counter()
    for it in range(cfg.MAX_ITER):
        net.train(); loss = -vamp2_score(feat(net(x0tr)), feat(net(x1tr)))
        if torch.isfinite(loss):
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.GRAD_CLIP); opt.step()
        loss_tr.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            val = float(-vamp2_score(feat(net(x0te)), feat(net(x1te))).detach())
            chi_all = net(f_all).cpu().numpy()
        loss_val.append(val); sdh.append(chi_all.std(0))
        if np.isfinite(val) and val < best_val: best_val, best_chi = val, chi_all.copy()
        if _plateau(loss_val, it, cfg): break
    if best_chi is None:
        with torch.no_grad(): best_chi = net(f_all).cpu().numpy()
    return dict(chi_best=best_chi.astype(np.float32),
                loss_train=np.array(loss_tr, np.float32), loss_val=np.array(loss_val, np.float32),
                opt_loss=np.array(loss_tr, np.float32), sd_history=np.array(sdh, np.float32),
                diag=np.zeros((0, 6)), elapsed=time.perf_counter() - t0, n_iter=len(loss_val))


# ── dispatcher with caching ──────────────────────────────────────────────────
def _run_dir(tag, variant, seed, head, warmup):
    sub = variant
    if head is not None and head != default_head(variant):
        sub = f"{variant}__{head}"
    if warmup:
        sub = sub + "_warm"
    d = os.path.join(paths.RUNS, tag, sub, f"seed_{seed}"); os.makedirs(d, exist_ok=True)
    return d

def train_chi(system, variant, seed, cfg=None, use_gpu=False, force=False, verbose=True,
              head=None, warmup=False) -> dict:
    tag = system["tag"]; cfg = cfg or get_cfg(tag)
    out = _run_dir(tag, variant, seed, head, warmup)
    cpath = os.path.join(out, "chi_best.npy")
    if os.path.exists(cpath) and not force:
        res = {kk: np.load(os.path.join(out, f"{kk}.npy"))
               for kk in ("chi_best", "loss_train", "loss_val", "opt_loss", "sd_history", "diag")
               if os.path.exists(os.path.join(out, f"{kk}.npy"))}
        with open(os.path.join(out, "meta.json")) as f: res.update(json.load(f))
        res["cached"] = True; return res
    device = _device(use_gpu)
    if variant == "svd_power":
        res = _run_power(system, seed, cfg, device, head=head, warmup=warmup)
    elif variant == "vamp":
        res = _run_vamp(system, seed, cfg, device, head=head, warmup=warmup)
    else:
        res = _run_isotarget(system, variant, seed, cfg, device, head=head, warmup=warmup)
    for kk in ("chi_best", "loss_train", "loss_val", "opt_loss", "sd_history", "diag"):
        np.save(os.path.join(out, f"{kk}.npy"), res[kk])
    sdf = res["sd_history"][-1] if len(res["sd_history"]) else np.zeros(n_out(variant))
    meta = dict(elapsed=res["elapsed"], n_iter=res["n_iter"],
                k_eff=int((res["chi_best"].std(0) > 0.05).sum()), sd_final=[float(s) for s in sdf])
    with open(os.path.join(out, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
    if verbose:
        print(f"  [{tag}/{variant}{'' if head is None else '/'+head}{'+warm' if warmup else ''}/seed{seed}] "
              f"iters={res['n_iter']:4d} k_eff={meta['k_eff']} t={res['elapsed']:.0f}s", flush=True)
    res.update(meta); res["cached"] = False; return res


# =========================================================================== #
# scoring
# =========================================================================== #
def pearson_r(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10: return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(a @ b / (na * nb))

def hungarian_assign(M):
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-np.nan_to_num(M, nan=0.0))
    return float(np.nanmean(M[ri, ci])), list(zip(ri.tolist(), ci.tolist()))

def shape_r(chi, refs):
    """Hungarian-matched mean |Pearson r| of chi (N,k) vs refs (R,N)."""
    k, R = chi.shape[1], refs.shape[0]
    M = np.array([[abs(pearson_r(chi[:, j], refs[i])) for i in range(R)] for j in range(k)])
    return hungarian_assign(M)

def eig_r(chi, ev_refs):
    """Eigenfunction-subspace recovery: |r| of the N_STATES-1 extracted eigenfunctions
    vs the true non-trivial eigenvectors (ev_refs is (N_STATES-1, N))."""
    return shape_r(eigfns(chi, ev_refs.shape[0]), ev_refs)

def membership_r(chi, comm_refs):
    """Hungarian |r| of the N_STATES memberships vs committor references."""
    return shape_r(to_memberships(chi), comm_refs)

def maxr_vs(chi, ref):
    return max(abs(pearson_r(chi[:, j], ref)) for j in range(chi.shape[1]))

def k_eff(chi):
    return int((np.asarray(chi).std(0) > 0.05).sum())

def _auroc(score, pos):
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool); n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def fate_auroc(chi, labels):
    states = np.unique(labels)
    M = np.array([[_auroc(chi[:, j], labels == s) for s in states] for j in range(chi.shape[1])])
    return hungarian_assign(M)
