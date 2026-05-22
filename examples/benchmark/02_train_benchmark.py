"""
Train all ISOKANN variants + VAMPnet baseline on triple-well and alanine dipeptide.

Architecture (fixed across all conditions, per benchmark spec)
--------------------------------------------------------------
  MLP: 3 hidden layers × 64 units, sigmoid activation throughout.
  Output dimension k: 3 for triple-well, 3 for AD.
  Optimizer: Adam, lr=1e-3.
  Full-batch (anchors are small).
  Epochs: 500, early stopping patience=50 on validation loss.
  5 training seeds × 5 split seeds = 25 runs per (condition, dataset).

Training mode per variant
-------------------------
  V1-V5 : power iteration — compute isotarget from (chi_x0, kchi_avg),
           then minimise MSE(chi(anchors), target) on train split.
  VAMP2  : gradient descent on negative VAMP-2 score directly.

Outputs
-------
  results/triple_well_results.npz
  results/alanine_results.npz

  Each contains per-run arrays (n_variants, n_split_seeds, n_train_seeds, ...):
    val_losses  (n_var, 5, 5, 500)  — val loss curve (NaN after early-stop epoch)
    best_epoch  (n_var, 5, 5)
    chi_train   (n_var, 5, 5, N_train, k)  — chi evaluated on train anchors
    chi_test    (n_var, 5, 5, N_test,  k)
    chi_all     (n_var, 5, 5, N_all,   k)  — all anchors
    phi_anchors (N_all,)  — for AD only
    psi_anchors (N_all,)
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from targets import apply_target, vamp2_score, CrossHistory, VARIANT_NAMES

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
HIDDEN_DIM   = 64
N_HIDDEN     = 3
EPOCHS       = 500
PATIENCE     = 50
LR           = 1e-3
N_SPLIT_SEEDS = 5
N_TRAIN_SEEDS = 5
# ISA/PseudoInv trivial-collapse fix: run this many epochs of GramSchmidt first
# so chi is non-uniform before the simplex/Schur inversion is applied.
WARMUP_EPOCHS = 50

DEVICE = torch.device("cpu")

VARIANTS = ["shiftscale", "isa", "gramschmidt", "pseudoinv", "cross", "vamp2"]


# ── Architecture ───────────────────────────────────────────────────────────────

def make_net(in_dim: int, k: int) -> nn.Module:
    """3×64 MLP with sigmoid throughout (benchmark spec)."""
    dims   = [in_dim] + [HIDDEN_DIM] * N_HIDDEN + [k]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def eval_chi(net: nn.Module, x: torch.Tensor) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return to_numpy(net(x))   # (n, k)


def kchi_avg(net: nn.Module, bursts_feat: torch.Tensor) -> np.ndarray:
    """
    Compute E[chi(x_tau) | x0] by averaging chi over the burst dimension.
    bursts_feat: (N_ANC, N_K, feat_dim)
    Returns (k, N_ANC) numpy array.
    """
    net.eval()
    N_ANC, N_K, _ = bursts_feat.shape
    acc = None
    for k_i in range(N_K):
        c = to_numpy(net(bursts_feat[:, k_i, :]))  # (N_ANC, k)
        acc = c if acc is None else acc + c
    return (acc / N_K).T   # (k, N_ANC)


def run_condition(variant: str, features: np.ndarray, bursts: np.ndarray,
                  patch_splits: np.ndarray, k: int,
                  split_seed: int, train_seed: int) -> dict:
    """
    Train one (variant, split_seed, train_seed) run.
    features : (N_ANC, feat_dim)
    bursts   : (N_ANC, N_K, feat_dim)
    Returns dict with val_losses, best_epoch, chi_all arrays.
    """
    torch.manual_seed(train_seed * 100 + split_seed * 10000)
    np.random.seed(train_seed * 100 + split_seed * 10000)

    split    = patch_splits[split_seed]        # (N_ANC,) 0=train 1=test
    tr_mask  = split == 0
    te_mask  = split == 1

    x_all   = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    x_tr    = x_all[tr_mask]
    x_te    = x_all[te_mask]
    b_all   = torch.tensor(bursts,   dtype=torch.float32, device=DEVICE)
    b_tr    = b_all[tr_mask]

    net     = make_net(features.shape[1], k).to(DEVICE)
    opt     = torch.optim.Adam(net.parameters(), lr=LR)

    # Gradient clipping: prevents ISA/PseudoInv target explosion at init
    GRAD_CLIP = 5.0

    history = CrossHistory(maxcols=k * 3) if variant == "cross" else None

    # ISA/PseudoInv: warm up with GramSchmidt to avoid trivial collapse at init
    warmup_var = "gramschmidt" if variant in ("isa", "pseudoinv") else None

    val_losses  = np.full(EPOCHS, np.nan)
    best_val    = np.inf
    patience_ct = 0
    best_epoch  = 0
    _in_warmup  = warmup_var is not None   # tracks whether we are still warming up

    for ep in range(EPOCHS):
        net.train()
        active_var = warmup_var if (warmup_var and ep < WARMUP_EPOCHS) else variant

        # At the warm-up → main-variant transition, reset early-stopping state so
        # the main variant gets a full PATIENCE budget of its own.
        if _in_warmup and ep == WARMUP_EPOCHS:
            _in_warmup  = False
            best_val    = np.inf
            patience_ct = 0

        if active_var == "vamp2":
            # VAMP-2: subsample pairs from flattened (anchor, burst) pairs
            b_flat = b_tr.view(-1, features.shape[1])   # (N_tr*N_K, feat)
            x0_rep = x_tr.repeat_interleave(bursts.shape[1], dim=0)
            chi0   = net(x0_rep)          # (N_tr*N_K, k)
            chi1   = net(b_flat)
            loss   = -vamp2_score(chi0, chi1)
            opt.zero_grad(); loss.backward(); opt.step()

            # Validation: VAMP-2 score on test split
            net.eval()
            with torch.no_grad():
                b_te    = b_all[te_mask]
                b_te_f  = b_te.view(-1, features.shape[1])
                x0_te_r = x_te.repeat_interleave(bursts.shape[1], dim=0)
                val_loss = float(-vamp2_score(net(x0_te_r), net(b_te_f)).detach())

        else:
            # Power iteration variants: compute isotarget then MSE
            chi_x0_np = to_numpy(net(x_tr)).T    # (k, N_tr)
            kchi_np   = kchi_avg(net, b_tr)       # (k, N_tr)

            try:
                target_np = apply_target(active_var, chi_x0_np, kchi_np, history)
            except (ValueError, np.linalg.LinAlgError):
                from targets import gramschmidt_target
                target_np = gramschmidt_target(kchi_np)

            # Normalise targets to [0, 1] per row so they are compatible with
            # sigmoid outputs.  ISA and PseudoInv need clamping first because
            # they can produce huge values when chi is near-uniform at init.
            # GramSchmidt/Cross return values in [-1,1] / arbitrary; normalising
            # them consistently is correct and necessary for sigmoid networks.
            if active_var in ("isa", "pseudoinv"):
                target_np = np.clip(target_np, -5.0, 5.0)
            t_min = target_np.min(1, keepdims=True)
            t_max = target_np.max(1, keepdims=True)
            span  = t_max - t_min
            target_np = np.where(span > 1e-6,
                                 (target_np - t_min) / span, 0.5)

            target = torch.tensor(target_np.T, dtype=torch.float32, device=DEVICE)  # (N_tr, k)
            net.train()
            pred = net(x_tr)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()

            # Validation loss: MSE on test anchors against their own isotarget
            # During warm-up use the active (warm-up) variant for consistency.
            net.eval()
            with torch.no_grad():
                chi_te_np  = to_numpy(net(x_te)).T
                kchi_te_np = kchi_avg(net, b_all[te_mask])
            try:
                tgt_te = apply_target(active_var, chi_te_np, kchi_te_np, None)
                if active_var in ("isa", "pseudoinv"):
                    tgt_te = np.clip(tgt_te, -5.0, 5.0)
                t_min = tgt_te.min(1, keepdims=True)
                t_max = tgt_te.max(1, keepdims=True)
                span  = t_max - t_min
                tgt_te = np.where(span > 1e-6, (tgt_te - t_min) / span, 0.5)
                tgt_te_t = torch.tensor(tgt_te.T, dtype=torch.float32)
                val_loss = float(nn.functional.mse_loss(net(x_te), tgt_te_t))
            except Exception:
                val_loss = float("nan")

        val_losses[ep] = val_loss

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val    = val_loss
            best_epoch  = ep
            patience_ct = 0
        else:
            patience_ct += 1

        if patience_ct >= PATIENCE:
            print(f"      early stop ep={ep}, best_val={best_val:.5f}")
            break

    chi_all = eval_chi(net, x_all)   # (N_ANC, k)

    return {
        "val_losses" : val_losses,      # (500,)
        "best_epoch" : best_epoch,
        "chi_all"    : chi_all,         # (N_ANC, k)
    }


# ── Dataset runner ─────────────────────────────────────────────────────────────

def run_dataset(name: str, npz_path: str, k: int,
                feat_key: str, burst_key: str):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  k={k}  variants={VARIANTS}")
    print(f"{'='*60}")

    data    = np.load(npz_path)
    features     = data[feat_key].astype(np.float32)    # (N_ANC, F)
    bursts_feat  = data[burst_key].astype(np.float32)   # (N_ANC, N_K, F)
    patch_splits = data["patch_splits"]                  # (5, N_ANC)
    N_ANC = features.shape[0]

    # result containers
    n_v = len(VARIANTS)
    val_store  = np.full((n_v, N_SPLIT_SEEDS, N_TRAIN_SEEDS, EPOCHS), np.nan)
    epoch_store = np.zeros((n_v, N_SPLIT_SEEDS, N_TRAIN_SEEDS), dtype=int)
    chi_store   = np.zeros((n_v, N_SPLIT_SEEDS, N_TRAIN_SEEDS, N_ANC, k), dtype=np.float32)

    for v_i, variant in enumerate(VARIANTS):
        k_eff = 1 if variant == "shiftscale" else k
        print(f"\n  Variant: {VARIANT_NAMES[variant]}  k={k_eff}")
        for ss in range(N_SPLIT_SEEDS):
            for ts in range(N_TRAIN_SEEDS):
                print(f"    split={ss} train={ts}", end="  ", flush=True)
                res = run_condition(variant, features, bursts_feat,
                                    patch_splits, k_eff, ss, ts)
                val_store[v_i, ss, ts]    = res["val_losses"]
                epoch_store[v_i, ss, ts]  = res["best_epoch"]
                chi_all = res["chi_all"]
                # Pad to k if variant was k_eff=1
                if chi_all.shape[1] < k:
                    pad = np.zeros((N_ANC, k - chi_all.shape[1]), dtype=np.float32)
                    chi_all = np.concatenate([chi_all, pad], axis=1)
                chi_store[v_i, ss, ts] = chi_all
                print(f"best_val={val_store[v_i,ss,ts,res['best_epoch']]:.4f} ep={res['best_epoch']}")

    save_dict = dict(
        variants    = np.array(VARIANTS, dtype=object),
        val_losses  = val_store,
        best_epoch  = epoch_store,
        chi_all     = chi_store,
        patch_splits = patch_splits,
        k           = np.array([k]),
    )
    # Forward extra arrays for AD
    for key in ["anchors_phi", "anchors_psi", "grid_phi", "grid_psi"]:
        if key in data:
            save_dict[key] = data[key]

    out_path = os.path.join(RESULTS_DIR, f"{name}_results.npz")
    np.savez(out_path, **save_dict)
    print(f"\nSaved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tw_path = os.path.join(DATA_DIR, "triple_well_koopman.npz")
    al_path = os.path.join(DATA_DIR, "alanine_koopman.npz")

    if os.path.exists(tw_path):
        run_dataset("triple_well", tw_path, k=3,
                    feat_key="anchors", burst_key="bursts")
    else:
        print(f"Triple-well data not found: {tw_path}")
        print("Run 00_simulate_triple_well.py first.")

    if os.path.exists(al_path):
        run_dataset("alanine", al_path, k=3,
                    feat_key="anchors_feat", burst_key="bursts_feat")
    else:
        print(f"Alanine data not found: {al_path}")
        print("Run 01_simulate_alanine.py first.")
