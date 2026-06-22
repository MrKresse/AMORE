# -*- coding: utf-8 -*-
"""
Benchmark v3 training script.

What changed from v2
--------------------
1. All method code imported from src/amore (isotarget.py, isokann.power,
   isokann.network).  No local reimplementations.
2. Network: ChiNetMultiLinear (Tanh hidden, linear output) — correct for
   isotarget variants whose targets are in natural scale (not [0,1]).
3. Device bug fixed: validation target tensor moved to DEVICE before MSE.
4. No bare except clauses swallowing errors into NaN / silent fallback.
   Failures surface loudly.
5. "svd" slot uses power_method_multi (subspace iteration) — not DMD eigen(H).
   See BENCHMARK_V2_POSTMORTEM.md bug 3.
6. Multi-tau condition now uses real 231-dim features at both lag times.
   v2's 0.1ps data was grid-snapped (postmortem bug 6) — replaced by
   01_simulate_alanine_0p1ps.py which reuses the existing anchors.
7. Warm-up uses gramschmidt_target from src/amore.isotarget.

Datasets
--------
  triple_well      : TW potential, tau=0.30, k=3
  alanine_5ps      : ADP 450K, tau=5ps, k=3
  alanine_multitau : ADP 450K, joint 5ps+0.1ps bursts (N_K=40), k=3

Architecture
------------
  ChiNetMultiLinear: input -> [128, 32, 8] -> k
  Tanh hidden, linear output.

Training
--------
  Max iter: 5000  (isotarget variants)
  Power method ("svd"): n_iter=100, epochs_per_iter=50  (~5000 total SGD steps)
  Early stopping: plateau criterion on val_loss, window W=500, after MIN_ITER=1000
  Warm-up: 100 iterations of GramSchmidt on all k  (isotarget variants only)
  5 paired seeds

Outputs
-------
  benchmark_v3/runs/{dataset}/{variant}/seed_{s}/
    val_loss.npy         (n_iter,)
    chi_sd_history.npy   (n_iter, k)
    chi_atstop.npy       (N_ANC, k)
    chi_best.npy         (N_ANC, k)
    meta.json
"""

import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

# Import all method code from src/amore — no local reimplementations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from amore.isotarget import (
    apply_target, gramschmidt_target, vamp2_score, VARIANT_NAMES,
)
from amore.isokann import ChiNetMultiLinear, power_method_multi

HERE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(HERE, "..", "benchmark", "data")
RUNS_DIR  = os.path.join(HERE, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
MAX_ITER  = 5000
MIN_ITER  = 1000
W         = 500
REL_TOL   = 1e-3
WARMUP    = 100
LR        = 1e-3
GRAD_CLIP = 5.0
N_SEEDS   = 5

# power_method_multi parameters (total SGD steps ≈ MAX_ITER for fair comparison)
POWER_N_ITER         = 100
POWER_EPOCHS_PER_ITER = 50   # 100 * 50 = 5000 total SGD steps

VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd", "vamp2"]
# "svd" label = subspace power iteration (power_method_multi), NOT DMD eigen(H)
# See BENCHMARK_V2_POSTMORTEM.md for why.


# ── Cross history accumulator ────────────────────────────────────────────────

class _CrossHist:
    """Rolling (X_hist, Y_hist) accumulator for the Cross isotarget variant."""

    def __init__(self, maxcols: int = 0):
        self.maxcols = maxcols
        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None

    def update(self, chi_np: np.ndarray,
               kchi_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # chi_np: (k, n) → append as columns to (n, cols) history
        X_new = chi_np.T.astype(np.float64)
        Y_new = kchi_np.T.astype(np.float64)
        if self.X is None:
            self.X, self.Y = X_new, Y_new
        else:
            self.X = np.hstack([self.X, X_new])
            self.Y = np.hstack([self.Y, Y_new])
        if self.maxcols > 0 and self.X.shape[1] > self.maxcols:
            self.X = self.X[:, -self.maxcols:]
            self.Y = self.Y[:, -self.maxcols:]
        return self.X, self.Y


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def eval_chi(net: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Evaluate chi on x, return (n, k) numpy."""
    net.eval()
    with torch.no_grad():
        return to_np(net(x))


def kchi_avg(net: nn.Module, bursts_t: torch.Tensor) -> np.ndarray:
    """
    bursts_t: (N, N_K, F)
    Returns (k, N): E[chi(x_tau) | x0], burst-averaged.
    """
    net.eval()
    N, N_K, _ = bursts_t.shape
    with torch.no_grad():
        acc = sum(to_np(net(bursts_t[:, ki, :])) for ki in range(N_K))
    return (acc / N_K).T   # (k, N)


def chi_sd(chi_np: np.ndarray) -> np.ndarray:
    """Per-mode standard deviation. chi_np: (N, k) → (k,)."""
    return chi_np.std(axis=0)


def plateau_stop(val_history: list, it: int) -> bool:
    if it < MIN_ITER or len(val_history) < W:
        return False
    recent = np.array(val_history[-W:])
    recent = recent[np.isfinite(recent)]
    if len(recent) < W // 2:
        return False
    rng = recent.max() - recent.min()
    med = np.median(recent)
    return rng < REL_TOL * max(abs(med), 1e-12)


# ── Power method variant ──────────────────────────────────────────────────────

def run_power(net: nn.Module, feat_t: torch.Tensor, bursts_t: torch.Tensor,
              feat_all_t: torch.Tensor, seed: int) -> dict:
    """
    Run subspace power iteration (power_method_multi) as the 'svd' variant.

    x0 = each anchor repeated N_K times to match the N*N_K burst pairs.
    x1 = all burst endpoints flattened.
    Uses the full (unsplit) data — power_method_multi handles its own
    stochastic batching internally.
    """
    torch.manual_seed(seed * 12345 + 7)
    np.random.seed(seed * 12345 + 7)

    N, N_K, F = bursts_t.shape
    x0 = feat_t.repeat_interleave(N_K, dim=0)   # (N*N_K, F) on DEVICE
    x1 = bursts_t.reshape(-1, F)                  # (N*N_K, F)

    t0 = time.perf_counter()
    result = power_method_multi(
        net, x0, x1,
        n_iter=POWER_N_ITER,
        epochs_per_iter=POWER_EPOCHS_PER_ITER,
        lr=LR,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    chi_final = eval_chi(net, feat_all_t)         # (N_all, k)

    # spans → approximate SD: for uniform[a,b], sd = (b-a)/(2*sqrt(3))
    spans = result["spans"]                        # (n_iter, k)
    sd_approx = spans / (2.0 * np.sqrt(3.0))

    return {
        "val_loss"       : np.array(result["losses"], dtype=np.float32),
        "chi_sd_history" : sd_approx.astype(np.float32),
        "chi_atstop"     : chi_final.astype(np.float32),
        "chi_best"       : chi_final.astype(np.float32),  # no checkpoint in power method
        "elapsed_s"      : elapsed,
        "n_iter"         : len(result["losses"]),
        "eigenvalues"    : result["eigenvalues"].tolist(),
        "timescales"     : result["timescales"].tolist(),
    }


# ── Isotarget variant training ────────────────────────────────────────────────

def run_isotarget(variant: str,
                  feat_tr: torch.Tensor, feat_te: torch.Tensor, feat_all: torch.Tensor,
                  bursts_tr: torch.Tensor, bursts_te: torch.Tensor,
                  net: nn.Module, seed: int) -> dict:
    """
    Train one (isotarget variant, seed) condition.

    Warm-up: 100 iterations of GramSchmidt on all k.
    Main loop: isotarget variant.
    Validation: MSE on test split with the SAME target (device bug fixed).
    Best checkpoint: saved when val improves.
    """
    torch.manual_seed(seed * 12345 + 7)
    np.random.seed(seed * 12345 + 7)

    opt   = torch.optim.Adam(net.parameters(), lr=LR)
    cross_hist = _CrossHist(maxcols=net(feat_tr[:1]).shape[-1] * 3) if variant == "cross" else None

    val_history: list[float] = []
    sd_history:  list[np.ndarray] = []
    best_val     = np.inf
    best_chi     = None

    t0 = time.perf_counter()

    for it in range(MAX_ITER):

        # ── Warm-up (GramSchmidt, all k) ──────────────────────────────────
        if it < WARMUP:
            with torch.no_grad():
                kchi_w = kchi_avg(net, bursts_tr)          # (k, N_tr)
                tgt_w  = gramschmidt_target(kchi_w)        # (k, N_tr)
            tgt_w_t = torch.tensor(tgt_w.T, dtype=torch.float32, device=DEVICE)
            net.train()
            pred_w = net(feat_tr)
            loss   = nn.functional.mse_loss(pred_w, tgt_w_t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            val_history.append(float(loss.detach()))
            sd_history.append(chi_sd(eval_chi(net, feat_all)))
            continue

        # ── Compute isotarget ─────────────────────────────────────────────
        chi_x0_np = eval_chi(net, feat_tr).T    # (k, N_tr)
        kchi_np   = kchi_avg(net, bursts_tr)     # (k, N_tr)

        cross_args: tuple[np.ndarray, np.ndarray] | None = None
        if variant == "cross" and cross_hist is not None:
            cross_args = cross_hist.update(chi_x0_np, kchi_np)

        # ISA and PseudoInv raise ValueError when chi is near-degenerate
        # (singular simplex submatrix). This is a documented, expected failure
        # mode — not a hidden infrastructure error. Skip the weight update,
        # log it, and continue: chi may spread in subsequent iterations.
        try:
            target_np = apply_target(variant, chi_x0_np, kchi_np,
                                     cross_hist=cross_args)
        except ValueError as e:
            if it % 100 == 0:
                print(f"      [it={it} {variant} degenerate: {e}]", flush=True)
            val_history.append(float("nan"))
            sd_history.append(chi_sd(eval_chi(net, feat_all)))
            continue

        target_t = torch.tensor(target_np.T, dtype=torch.float32, device=DEVICE)
        net.train()
        pred  = net(feat_tr)
        loss  = nn.functional.mse_loss(pred, target_t)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()

        # ── Validation (device bug fixed) ─────────────────────────────────
        net.eval()
        with torch.no_grad():
            chi_te_np  = eval_chi(net, feat_te).T
            kchi_te_np = kchi_avg(net, bursts_te)

        try:
            tgt_te   = apply_target(variant, chi_te_np, kchi_te_np)
            tgt_te_t = torch.tensor(tgt_te.T, dtype=torch.float32,
                                    device=DEVICE)         # ← v2 device bug fixed here
            with torch.no_grad():
                val = float(nn.functional.mse_loss(net(feat_te), tgt_te_t).detach())
        except ValueError:
            val = float("nan")   # degenerate test split — same failure mode

        val_history.append(val)
        chi_all_np = eval_chi(net, feat_all)
        sd_history.append(chi_sd(chi_all_np))

        if np.isfinite(val) and val < best_val:
            best_val = val
            best_chi = chi_all_np.copy()

        if plateau_stop(val_history, it):
            break

    chi_atstop = eval_chi(net, feat_all)
    if best_chi is None:
        best_chi = chi_atstop

    elapsed = time.perf_counter() - t0
    return {
        "val_loss"       : np.array(val_history,  dtype=np.float32),
        "chi_sd_history" : np.array(sd_history,   dtype=np.float32),
        "chi_atstop"     : chi_atstop.astype(np.float32),
        "chi_best"       : best_chi.astype(np.float32),
        "elapsed_s"      : elapsed,
        "n_iter"         : len(val_history),
    }


# ── VAMP2 variant ─────────────────────────────────────────────────────────────

def run_vamp2(feat_tr: torch.Tensor, feat_te: torch.Tensor, feat_all: torch.Tensor,
              bursts_tr: torch.Tensor, bursts_te: torch.Tensor,
              net: nn.Module, seed: int) -> dict:
    """
    VAMP-2 score maximisation (no isotarget, no warm-up).

    Column normalisation prevents the chi=0 degenerate fixed point (VAMP-2
    gradient vanishes at chi=0 with linear output; normalising removes this).
    """
    torch.manual_seed(seed * 12345 + 7)
    np.random.seed(seed * 12345 + 7)

    in_dim = feat_tr.shape[1]
    opt    = torch.optim.Adam(net.parameters(), lr=LR)

    val_history: list[float] = []
    sd_history:  list[np.ndarray] = []
    best_val     = np.inf
    best_chi     = None

    def _norm(c: torch.Tensor) -> torch.Tensor:
        s = c.std(dim=0, keepdim=True).clamp(min=0.05)
        return c / s

    t0 = time.perf_counter()

    for it in range(MAX_ITER):
        net.train()
        b_flat  = bursts_tr.reshape(-1, in_dim)
        x0_rep  = feat_tr.repeat_interleave(bursts_tr.shape[1], dim=0)
        chi0    = _norm(net(x0_rep))
        chi1    = _norm(net(b_flat))
        loss    = -vamp2_score(chi0, chi1)

        if torch.isfinite(loss):
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()

        # Validation
        net.eval()
        with torch.no_grad():
            b_te_f  = bursts_te.reshape(-1, in_dim)
            x0_te_r = feat_te.repeat_interleave(bursts_te.shape[1], dim=0)
            val = float(-vamp2_score(_norm(net(x0_te_r)), _norm(net(b_te_f))).detach())

        val_history.append(val)
        chi_all_np = eval_chi(net, feat_all)
        sd_history.append(chi_sd(chi_all_np))

        if np.isfinite(val) and val < best_val:
            best_val = val
            best_chi = chi_all_np.copy()

        if plateau_stop(val_history, it):
            break

    chi_atstop = eval_chi(net, feat_all)
    if best_chi is None:
        best_chi = chi_atstop

    elapsed = time.perf_counter() - t0
    return {
        "val_loss"       : np.array(val_history,  dtype=np.float32),
        "chi_sd_history" : np.array(sd_history,   dtype=np.float32),
        "chi_atstop"     : chi_atstop.astype(np.float32),
        "chi_best"       : best_chi.astype(np.float32),
        "elapsed_s"      : elapsed,
        "n_iter"         : len(val_history),
    }


# ── Dataset runner ─────────────────────────────────────────────────────────────

def run_dataset(ds_name: str, feat: np.ndarray, bursts: np.ndarray,
                patch_splits: np.ndarray) -> None:
    """Run all variants × seeds for one dataset."""
    print(f"\n{'='*65}")
    print(f"Dataset: {ds_name}  k=3  variants={VARIANTS}")
    print(f"{'='*65}")

    k      = 3
    in_dim = feat.shape[1]
    N_ANC  = feat.shape[0]

    feat_t   = torch.tensor(feat,   dtype=torch.float32, device=DEVICE)
    bursts_t = torch.tensor(bursts, dtype=torch.float32, device=DEVICE)

    for variant in VARIANTS:
        print(f"\n  Variant: {VARIANT_NAMES.get(variant, variant)}"
              + (" [power_method_multi]" if variant == "svd" else ""))

        for seed in range(N_SEEDS):
            out_dir = os.path.join(RUNS_DIR, ds_name, variant, f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)

            if os.path.exists(os.path.join(out_dir, "chi_atstop.npy")):
                print(f"    seed={seed}  [already done, skipping]")
                continue

            print(f"    seed={seed}", end="  ", flush=True)

            # Fresh network for each (variant, seed)
            torch.manual_seed(seed * 12345 + 7)
            net = ChiNetMultiLinear(in_dim, k, hidden=[128, 32, 8]).to(DEVICE)

            # Train/test split
            split   = patch_splits[seed]
            tr_mask = split == 0
            te_mask = split == 1
            f_tr    = feat_t[tr_mask];    f_te = feat_t[te_mask]
            b_tr    = bursts_t[tr_mask];  b_te = bursts_t[te_mask]

            if variant == "svd":
                # Subspace power iteration on the full (unsplit) data
                res = run_power(net, feat_t, bursts_t, feat_t, seed=seed)
            elif variant == "vamp2":
                res = run_vamp2(f_tr, f_te, feat_t, b_tr, b_te, net, seed=seed)
            else:
                res = run_isotarget(variant, f_tr, f_te, feat_t,
                                    b_tr, b_te, net, seed=seed)

            np.save(os.path.join(out_dir, "val_loss.npy"),       res["val_loss"])
            np.save(os.path.join(out_dir, "chi_sd_history.npy"), res["chi_sd_history"])
            np.save(os.path.join(out_dir, "chi_atstop.npy"),     res["chi_atstop"])
            np.save(os.path.join(out_dir, "chi_best.npy"),       res["chi_best"])

            meta = {
                "n_iter"   : res["n_iter"],
                "elapsed_s": res["elapsed_s"],
                "val_final": float(res["val_loss"][-1]) if len(res["val_loss"]) else None,
            }
            if "eigenvalues" in res:
                # complex numpy → serializable (real, imag) pairs
                meta["eigenvalues"] = [[float(e.real), float(e.imag)]
                                       for e in res["eigenvalues"]]
                meta["timescales"]  = [float(t) for t in res["timescales"]]
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            sd_final = (res["chi_sd_history"][-1]
                        if len(res["chi_sd_history"]) else np.zeros(k))
            k_eff    = int((sd_final > 0.05).sum())
            print(f"iters={res['n_iter']:4d}  "
                  f"sd=[{', '.join(f'{s:.3f}' for s in sd_final)}]  "
                  f"k_eff={k_eff}  "
                  f"t={res['elapsed_s']:.0f}s")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Triple-well ─────────────────────────────────────────────────────────
    tw_path = os.path.join(DATA_DIR, "triple_well_koopman.npz")
    if os.path.exists(tw_path):
        tw = np.load(tw_path)
        run_dataset("triple_well",
                    feat         = tw["anchors"],
                    bursts       = tw["bursts"],
                    patch_splits = tw["patch_splits"])
    else:
        print(f"Triple-well data not found: {tw_path}")

    # ── ADP tau=5ps ──────────────────────────────────────────────────────────
    al_path = os.path.join(DATA_DIR, "alanine_koopman.npz")
    if os.path.exists(al_path):
        al = np.load(al_path)
        run_dataset("alanine_5ps",
                    feat         = al["anchors_feat"],
                    bursts       = al["bursts_feat"],
                    patch_splits = al["patch_splits"])
    else:
        print(f"ADP data not found: {al_path}")

    # ── ADP tau=0.1ps only ───────────────────────────────────────────────────
    al_path_0p1 = os.path.join(DATA_DIR, "alanine_0p1ps_koopman.npz")
    if os.path.exists(al_path) and os.path.exists(al_path_0p1):
        al_5ps  = np.load(al_path)
        al_0p1  = np.load(al_path_0p1)
        run_dataset("alanine_0p1ps",
                    feat         = al_5ps["anchors_feat"],
                    bursts       = al_0p1["bursts_feat"],
                    patch_splits = al_5ps["patch_splits"])
    else:
        print(f"\nalanine_0p1ps skipped — run 01_simulate_alanine_0p1ps.py first.")

    # ── ADP multi-tau (5 ps + 0.1 ps joint) ─────────────────────────────────
    # Joint dataset: same 1578 anchors, bursts concatenated along N_K axis.
    # 5 ps bursts capture the slow C7eq ↔ C7ax transition;
    # 0.1 ps bursts add high-resolution local geometry information.
    # The isotarget Koopman average is computed over all 40 burst endpoints,
    # which is equivalent to training on an empirical mixture of the two
    # Koopman operators.  True eigenfunctions are eigenfunctions at both
    # lag times, so the slow mode should still emerge.
    al_path_0p1 = os.path.join(DATA_DIR, "alanine_0p1ps_koopman.npz")
    if os.path.exists(al_path) and os.path.exists(al_path_0p1):
        al_5ps  = np.load(al_path)
        al_0p1  = np.load(al_path_0p1)
        bursts_joint = np.concatenate(
            [al_5ps["bursts_feat"], al_0p1["bursts_feat"]], axis=1
        )   # (1578, 40, 231)
        run_dataset("alanine_multitau",
                    feat         = al_5ps["anchors_feat"],
                    bursts       = bursts_joint,
                    patch_splits = al_5ps["patch_splits"])
    else:
        missing = []
        if not os.path.exists(al_path):     missing.append("alanine_koopman.npz")
        if not os.path.exists(al_path_0p1): missing.append("alanine_0p1ps_koopman.npz")
        print(f"\nMulti-tau skipped — missing: {', '.join(missing)}")
        print("  Run 01_simulate_alanine_0p1ps.py to generate 0.1 ps burst data.")
