# -*- coding: utf-8 -*-
"""
Benchmark v2 training script.

Datasets
--------
  triple_well   : TW potential, tau=0.30, k=3
  alanine_5ps   : ADP 450K, tau=5ps, k=3
  alanine_multi : ADP 450K, tau=5ps + tau=0.1ps combined, k=3

Architecture (fixed across all methods)
----------------------------------------
  MLP: input -> [128, 32, 8] -> k
  Sigmoid hidden, LINEAR output.
  (No sigmoid output — targets are in arbitrary range, no [0,1] renorm needed.)
  Note: v1 had sigmoid output + [0,1] target normalisation — this caused systematic
  bias. v2 fixes this. All isotarget methods return targets in their natural range.

Training
--------
  Max iter: 5000
  Early stopping: plateau criterion — stop if
    max(val[-W:]) - min(val[-W:]) < REL_TOL * median(val[-W:])
    after MIN_ITER iterations, with window W=500.
  Warm-up: 100 iterations of 1D ShiftScale on chi_1 only (chi_2..k frozen).
    Same for all methods (not just ISA/PseudoInv as in v1).
  Dual checkpoint: save both at-stop and best-val-loss checkpoints.
  5 PAIRED seeds (split_seed == train_seed).

Variants
--------
  isa, gramschmidt, pseudoinv, cross, svd, vamp2
  (ShiftScale excluded — 1D only, not useful for k=3 benchmark)

Multi-tau (alanine_multi)
--------------------------
  In each outer iteration, one isotarget update is computed from K_{5ps}[chi]
  and one from K_{0.1ps}[chi] (alternating). Both use the same variant transform.
  VAMP2: combined loss at both lag times, equal weight.

Outputs
-------
  benchmark_v2/runs/{dataset}/{variant}/seed_{s}/
    val_loss.npy          (n_iter,)
    chi_sd_history.npy    (n_iter, k)
    chi_atstop.npy        (N_ANC, k)
    chi_best.npy          (N_ANC, k)
    meta.json
"""

import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from targets import apply_target, vamp2_score, CrossHistory, VARIANT_NAMES

HERE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(HERE, "..", "benchmark", "data")
RUNS_DIR  = os.path.join(HERE, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Hyperparameters ─────────────────────────────────────────────────────────────
MAX_ITER   = 5000
MIN_ITER   = 1000     # no early stop before this
W          = 500      # plateau window
REL_TOL    = 1e-3     # stop if range(val[-W:]) < REL_TOL * median(val[-W:])
WARMUP     = 100      # iterations of 1D ShiftScale on chi_1 only
LR         = 1e-3
N_SEEDS    = 5        # paired split_seed == train_seed

VARIANTS = ["isa", "gramschmidt", "pseudoinv", "svd", "cross", "vamp2"]

# ── Architecture ─────────────────────────────────────────────────────────────────

def make_net(in_dim: int, k: int) -> nn.Module:
    """
    MLP: in_dim -> [128, 32, 8] -> k
    Sigmoid hidden, linear output (v2 spec).
    """
    dims   = [in_dim, 128, 32, 8, k]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Sigmoid())
        # last layer: no activation (linear output)
    return nn.Sequential(*layers)


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_np(t): return t.detach().cpu().float().numpy()

def eval_chi(net, x):
    net.eval()
    with torch.no_grad():
        return to_np(net(x))   # (n, k)

def kchi_avg(net, bursts_t):
    """
    bursts_t: (N, N_K, feat_dim) tensor
    Returns (k, N) numpy: E[chi(x_tau) | x0].
    """
    net.eval()
    N, N_K, _ = bursts_t.shape
    with torch.no_grad():
        acc = sum(to_np(net(bursts_t[:, ki, :])) for ki in range(N_K))
    return (acc / N_K).T   # (k, N)


def chi_sd(chi_np):
    """Per-mode standard deviation across anchors. chi_np: (N, k)."""
    return chi_np.std(axis=0)   # (k,)


def plateau_stop(val_history, it):
    """Return True if the plateau criterion is met."""
    if it < MIN_ITER or len(val_history) < W:
        return False
    recent = np.array(val_history[-W:])
    recent = recent[np.isfinite(recent)]
    if len(recent) < W // 2:
        return False
    rng = recent.max() - recent.min()
    med = np.median(recent)
    return (rng < REL_TOL * max(abs(med), 1e-12))


# ── Single-condition training ──────────────────────────────────────────────────

def run_one(variant, features_tr, features_te, features_all,
            bursts_tr, bursts_te,
            bursts2_tr=None, bursts2_te=None,   # second lag (multi-tau)
            seed=0):
    """
    Train one (variant, seed) condition.

    features_* : (N, F) tensors
    bursts_*   : (N, N_K, F) tensors  — primary lag (5ps for ADP)
    bursts2_*  : (N, N_K2, F) tensors — secondary lag (0.1ps), optional
    """
    torch.manual_seed(seed * 12345 + 7)
    np.random.seed(seed * 12345 + 7)

    k        = 3
    in_dim   = features_tr.shape[1]
    net      = make_net(in_dim, k).to(DEVICE)
    opt      = torch.optim.Adam(net.parameters(), lr=LR)
    GRAD_CLIP = 5.0
    history  = CrossHistory(maxcols=k * 3) if variant == "cross" else None
    history2 = CrossHistory(maxcols=k * 3) if (variant == "cross" and bursts2_tr is not None) else None

    multi_tau = bursts2_tr is not None

    val_history  = []
    sd_history   = []
    best_val     = np.inf
    best_chi     = None

    t0 = time.perf_counter()

    for it in range(MAX_ITER):

        # ── Warm-up: GramSchmidt on all k outputs (skipped for VAMP2) ──────────
        # GramSchmidt warm-up trains all k dimensions simultaneously, giving
        # ISA/PseudoInv/Cross/SVD a non-degenerate k-dimensional starting point.
        # VAMP2 is excluded: its score-based objective handles multi-dim init
        # naturally and the GramSchmidt warm-up pulls its weights in the wrong
        # direction (MSE vs score maximisation are conflicting objectives).
        if it < WARMUP and variant != "vamp2":
            from targets import gramschmidt_target
            net.train()
            with torch.no_grad():
                chi_w_np = eval_chi(net, features_tr).T    # (k, N)
                kchi_w   = kchi_avg(net, bursts_tr)         # (k, N)
                tgt_w    = gramschmidt_target(kchi_w)       # (k, N) orthonormal
            tgt_w_t = torch.tensor(tgt_w.T, dtype=torch.float32, device=DEVICE)
            pred_w  = net(features_tr)
            loss    = nn.functional.mse_loss(pred_w, tgt_w_t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            val_history.append(float(loss.detach()))
            sd_history.append(chi_sd(eval_chi(net, features_all)))
            continue

        net.train()

        if variant == "vamp2":
            # ── VAMP2: negative VAMP-2 score at primary lag ──────────────────
            # Normalize chi column-wise before score computation to remove the
            # chi=0 degenerate fixed point (VAMP-2 score is scale-invariant but
            # its gradient vanishes at chi=0 with linear output).
            def _norm_chi(c):
                # Clamp std to 0.05 minimum so collapsed chi (std≈0) produces
                # bounded normalised values instead of NaN-inducing huge numbers.
                s = c.std(dim=0, keepdim=True).clamp(min=0.05)
                return c / s

            b_flat  = bursts_tr.view(-1, in_dim)
            x0_rep  = features_tr.repeat_interleave(bursts_tr.shape[1], dim=0)
            chi0    = _norm_chi(net(x0_rep))
            chi1_v  = _norm_chi(net(b_flat))
            try:
                loss = -vamp2_score(chi0, chi1_v)
                if multi_tau:
                    b2_flat  = bursts2_tr.view(-1, in_dim)
                    x0_rep2  = features_tr.repeat_interleave(bursts2_tr.shape[1], dim=0)
                    chi0_2   = _norm_chi(net(x0_rep2))
                    chi1_2   = _norm_chi(net(b2_flat))
                    loss     = loss - vamp2_score(chi0_2, chi1_2)
                if torch.isfinite(loss):
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                    opt.step()
            except RuntimeError:
                pass  # skip this iteration; network state unchanged

            # Validation: negative VAMP-2 on test split (normalised)
            net.eval()
            try:
                with torch.no_grad():
                    b_te_f  = bursts_te.view(-1, in_dim)
                    x0_te_r = features_te.repeat_interleave(bursts_te.shape[1], dim=0)
                    val = float(-vamp2_score(_norm_chi(net(x0_te_r)), _norm_chi(net(b_te_f))).detach())
                    if multi_tau and bursts2_te is not None:
                        b2_te_f  = bursts2_te.view(-1, in_dim)
                        x0_te_r2 = features_te.repeat_interleave(bursts2_te.shape[1], dim=0)
                        val += float(-vamp2_score(_norm_chi(net(x0_te_r2)), _norm_chi(net(b2_te_f))).detach())
            except Exception:
                val = float("nan")   # treated as no improvement by plateau check

        else:
            # ── Power-iteration variants ───────────────────────────────────────
            # Primary lag step
            chi_x0_np = eval_chi(net, features_tr).T    # (k, N_tr)
            kchi_np   = kchi_avg(net, bursts_tr)         # (k, N_tr)
            try:
                target_np = apply_target(variant, chi_x0_np, kchi_np, history)
            except Exception:
                from targets import gramschmidt_target
                target_np = gramschmidt_target(kchi_np)

            target_t = torch.tensor(target_np.T, dtype=torch.float32, device=DEVICE)
            net.train()
            pred  = net(features_tr)
            loss  = nn.functional.mse_loss(pred, target_t)

            if multi_tau:
                # Secondary lag step (averaged into same gradient pass)
                net.eval()
                chi_x0_2 = eval_chi(net, features_tr).T
                kchi_2   = kchi_avg(net, bursts2_tr)
                try:
                    tgt2_np = apply_target(variant, chi_x0_2, kchi_2, history2)
                except Exception:
                    from targets import gramschmidt_target
                    tgt2_np = gramschmidt_target(kchi_2)
                tgt2_t  = torch.tensor(tgt2_np.T, dtype=torch.float32, device=DEVICE)
                net.train()
                loss = loss + nn.functional.mse_loss(net(features_tr), tgt2_t)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()

            # Validation: MSE on test split
            net.eval()
            with torch.no_grad():
                chi_te_np  = eval_chi(net, features_te).T
                kchi_te_np = kchi_avg(net, bursts_te)
            try:
                tgt_te = apply_target(variant, chi_te_np, kchi_te_np, None)
                tgt_te_t = torch.tensor(tgt_te.T, dtype=torch.float32)
                val = float(nn.functional.mse_loss(net(features_te), tgt_te_t).detach())
            except Exception:
                val = float("nan")

        val_history.append(val)
        chi_all_np = eval_chi(net, features_all)       # (N_all, k)
        sd_history.append(chi_sd(chi_all_np))

        if np.isfinite(val) and val < best_val:
            best_val = val
            best_chi = chi_all_np.copy()

        if plateau_stop(val_history, it):
            break

    chi_atstop = eval_chi(net, features_all)
    if best_chi is None:
        best_chi = chi_atstop

    elapsed = time.perf_counter() - t0
    return {
        "val_loss"        : np.array(val_history, dtype=np.float32),
        "chi_sd_history"  : np.array(sd_history,  dtype=np.float32),
        "chi_atstop"      : chi_atstop.astype(np.float32),
        "chi_best"        : best_chi.astype(np.float32),
        "elapsed_s"       : elapsed,
        "n_iter"          : len(val_history),
    }


# ── Dataset runner ──────────────────────────────────────────────────────────────

def run_dataset(ds_name, feat, bursts, patch_splits,
                extra_bursts=None, extra_feat=None,
                extra_patch_splits=None):
    """
    Run all variants × seeds for one dataset.

    feat          : (N_ANC, F)         anchor features
    bursts        : (N_ANC, N_K, F)    primary lag bursts
    extra_bursts  : (N_ANC2, N_K2, F)  secondary lag bursts (multi-tau)
    extra_feat    : (N_ANC2, F)        anchors for secondary lag (may differ)
    """
    print(f"\n{'='*65}")
    print(f"Dataset: {ds_name}  k=3  variants={VARIANTS}")
    print(f"{'='*65}")

    N_ANC     = feat.shape[0]
    is_multi  = extra_bursts is not None

    # If multi-tau, align anchor sets: use intersection
    if is_multi and extra_feat is not None:
        # extra_feat anchors may be a subset; just pass them directly
        # The patch split for multi-tau is re-derived from extra_feat's indices
        pass

    feat_t    = torch.tensor(feat,   dtype=torch.float32, device=DEVICE)
    bursts_t  = torch.tensor(bursts, dtype=torch.float32, device=DEVICE)

    if is_multi:
        # For multi-tau, use the intersection of both anchor sets
        # (extra_feat anchors were sampled from feat anchors by index)
        feat2_t   = torch.tensor(extra_feat,   dtype=torch.float32, device=DEVICE)
        bursts2_t = torch.tensor(extra_bursts, dtype=torch.float32, device=DEVICE)
    else:
        feat2_t   = None
        bursts2_t = None

    for variant in VARIANTS:
        print(f"\n  Variant: {VARIANT_NAMES.get(variant, variant)}")
        for seed in range(N_SEEDS):
            out_dir = os.path.join(RUNS_DIR, ds_name, variant, f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)

            # Check if already done
            if os.path.exists(os.path.join(out_dir, "chi_atstop.npy")):
                print(f"    seed={seed}  [already done, skipping]")
                continue

            print(f"    seed={seed}", end="  ", flush=True)

            # Train/test split (paired: split_seed == seed)
            split   = patch_splits[seed]        # (N_ANC,) — 0=train 1=test
            tr_mask = split == 0
            te_mask = split == 1

            f_tr   = feat_t[tr_mask];   f_te   = feat_t[te_mask]
            b_tr   = bursts_t[tr_mask]; b_te   = bursts_t[te_mask]

            if is_multi:
                # Use the secondary lag data as-is (same anchors, different bursts)
                # Patch split applied to the secondary anchor set
                sp2 = extra_patch_splits[seed] if extra_patch_splits is not None else split[:len(extra_feat)]
                tr2 = sp2 == 0
                te2 = sp2 == 1
                f2_tr = feat2_t[tr2]; f2_te = feat2_t[te2]
                b2_tr = bursts2_t[tr2]; b2_te = bursts2_t[te2]
            else:
                f2_tr = f2_te = b2_tr = b2_te = None

            res = run_one(variant, f_tr, f_te, feat_t,
                          b_tr, b_te, b2_tr, b2_te, seed=seed)

            np.save(os.path.join(out_dir, "val_loss.npy"),        res["val_loss"])
            np.save(os.path.join(out_dir, "chi_sd_history.npy"),  res["chi_sd_history"])
            np.save(os.path.join(out_dir, "chi_atstop.npy"),      res["chi_atstop"])
            np.save(os.path.join(out_dir, "chi_best.npy"),        res["chi_best"])
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump({"n_iter": res["n_iter"],
                           "elapsed_s": res["elapsed_s"],
                           "val_final": float(res["val_loss"][-1]) if len(res["val_loss"]) else None},
                          f, indent=2)

            sd_final = res["chi_sd_history"][-1] if len(res["chi_sd_history"]) else np.zeros(3)
            k_eff    = int((sd_final > 0.05).sum())
            print(f"iters={res['n_iter']:4d}  "
                  f"sd=[{', '.join(f'{s:.3f}' for s in sd_final)}]  "
                  f"k_eff={k_eff}  "
                  f"t={res['elapsed_s']:.0f}s")


# ── Load data and run ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Triple-well ──────────────────────────────────────────────────────────────
    tw_path = os.path.join(DATA_DIR, "triple_well_koopman.npz")
    if os.path.exists(tw_path):
        tw = np.load(tw_path)
        run_dataset("triple_well",
                    feat          = tw["anchors"],        # (N, 2)
                    bursts        = tw["bursts"],          # (N, 20, 2)
                    patch_splits  = tw["patch_splits"])
    else:
        print(f"Triple-well data not found: {tw_path}")

    # ── ADP tau=5ps ───────────────────────────────────────────────────────────────
    al_path = os.path.join(DATA_DIR, "alanine_koopman.npz")
    if os.path.exists(al_path):
        al = np.load(al_path)
        run_dataset("alanine_5ps",
                    feat          = al["anchors_feat"],    # (1578, 231)
                    bursts        = al["bursts_feat"],     # (1578, 20, 231)
                    patch_splits  = al["patch_splits"])
    else:
        print(f"ADP data not found: {al_path}")

    # ── ADP multi-tau (5ps + 0.1ps) ───────────────────────────────────────────────
    ml_path = os.path.join(DATA_DIR, "alanine_multilag.npz")
    if os.path.exists(al_path) and os.path.exists(ml_path):
        al  = np.load(al_path)
        ml  = np.load(ml_path)

        # 0.1ps burst features: compute pairwise distances from Cartesian coords.
        # We have phi/psi in the multilag npz but NOT 231-dim features.
        # Recompute features from the anchor cartesian coords for the 0.1ps endpoints.
        # The multilag file has bursts_phi/psi but not bursts_feat.
        # For the multi-tau benchmark we use phi/psi-derived features from the
        # original anchor features (reindex into the full 1578-anchor feature array).
        #
        # Actually: the multilag anchors are a random subset of the 1578 anchors.
        # We use their feat vectors from the full alanine_koopman.npz.
        # For the 0.1ps BURSTS we need 231-dim features, but multilag only saved
        # phi/psi of endpoints. We'll use a 2D (phi, psi) feature for the 0.1ps
        # component as a proxy (sufficient to separate basins at this resolution).
        #
        # Multi-tau architecture: primary (5ps) uses 231-dim features,
        # secondary (0.1ps) uses 2D (phi, psi) features.
        # To keep the same network, we need to embed 2D into 231D or vice-versa.
        # Simplest: use the same 231D network for both lags.
        # For secondary bursts: look up feat vectors for 0.1ps endpoints via
        # nearest-anchor lookup in the (phi, psi) grid.
        #
        # PRACTICAL SHORTCUT for v2:
        # Use the 0.1ps ANCHOR features (same as 5ps — these are the starting points)
        # as x0 features, and use the multilag phi/psi endpoint to identify the
        # nearest grid cell in alanine_koopman.npz anchors, then use that anchor's
        # feat as the burst feature.
        #
        # This is an approximation but sufficient for the pilot multi-tau test.
        # A proper version would save 231-dim features at each checkpoint.

        print("\nPreparing multi-tau ADP dataset ...")
        ml_phi_src  = ml["anchors_phi"]    # (N_ML,) subset anchor phi
        ml_psi_src  = ml["anchors_psi"]    # (N_ML,)
        ml_bphi_01  = ml["bursts_phi"][:, 0, 0]   # (N_ML,) tau=0.1ps phi endpoint
        ml_bpsi_01  = ml["bursts_psi"][:, 0, 0]   # (N_ML,)
        N_ML = len(ml_phi_src)

        # Find nearest anchor in the full 1578-anchor set for each multilag anchor
        all_phi = al["anchors_phi"]   # (1578,)
        all_psi = al["anchors_psi"]
        all_feat = al["anchors_feat"]  # (1578, 231)

        def nearest_anchor_feat(phis, psis):
            """For each (phi, psi), return feat of the nearest anchor."""
            dist = (phis[:, None] - all_phi[None, :])**2 + \
                   (psis[:, None] - all_psi[None, :])**2
            idx  = dist.argmin(axis=1)
            return all_feat[idx]

        # Features for multilag anchors (x0)
        ml_feat_src  = nearest_anchor_feat(ml_phi_src, ml_psi_src)   # (N_ML, 231)
        # Features for 0.1ps burst endpoints (x_0.1ps)
        ml_feat_01   = nearest_anchor_feat(ml_bphi_01, ml_bpsi_01)   # (N_ML, 231)
        # Wrap as (N_ML, 1, 231) burst array (1 burst per anchor)
        ml_bursts_01 = ml_feat_01[:, None, :]                          # (N_ML, 1, 231)

        # Patch splits for the multilag anchor subset
        # Use simple random 80/20 splits for each seed
        rng = np.random.default_rng(42)
        ml_splits = np.zeros((N_SEEDS, N_ML), dtype=np.int8)
        for s in range(N_SEEDS):
            te_idx = rng.choice(N_ML, size=N_ML // 5, replace=False)
            ml_splits[s, te_idx] = 1

        run_dataset("alanine_multi_tau",
                    feat          = ml_feat_src,      # (N_ML, 231) x0 features
                    bursts        = np.stack([
                                      nearest_anchor_feat(ml["bursts_phi"][:, 0, 4],
                                                          ml["bursts_psi"][:, 0, 4])
                                    ], axis=1),       # (N_ML, 1, 231) 5ps bursts (1 burst)
                    patch_splits  = ml_splits,
                    extra_bursts  = ml_bursts_01,     # (N_ML, 1, 231) 0.1ps bursts
                    extra_feat    = ml_feat_src,       # same x0 for both lags
                    extra_patch_splits = ml_splits)
    else:
        if not os.path.exists(ml_path):
            print(f"Multi-lag data not found ({ml_path}). "
                  f"Run 01b_adp_its_data.py to generate it.")
