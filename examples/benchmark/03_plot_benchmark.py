"""
Generate benchmark panels from training results.

Panel 1: Convergence curves (val loss vs epoch, all seeds per variant)
Panel 2: Story panel
  2a. Empirical isocommittor sanity figure
  2b. AU-ROC boxplots (Hungarian assignment to well labels)
  2c. Separatrix MAE boxplot
  2d. Gradient-norm boxplot
Panel 3: Learned slow modes (best model per condition)
"""

from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from targets import VARIANT_NAMES

# ── Color palette (one per variant) ───────────────────────────────────────────
VARIANT_COLORS = {
    "shiftscale"  : "#1f77b4",   # blue
    "isa"         : "#ff7f0e",   # orange
    "gramschmidt" : "#2ca02c",   # green
    "pseudoinv"   : "#d62728",   # red
    "cross"       : "#9467bd",   # purple
    "vamp2"       : "#8c564b",   # brown
}
VARIANT_LABELS = {k: v for k, v in VARIANT_NAMES.items()}


def savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def hungarian_auc(chi_all: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign chi columns to label columns via Hungarian algorithm maximising
    total AU-ROC.  Returns (auc_matrix k×K, best_assignment array).

    chi_all : (N, k)
    labels  : (N, K)  binary columns (0/1)
    """
    k = chi_all.shape[1]
    K = labels.shape[1]
    n = min(k, K)
    auc_mat = np.zeros((k, K))
    for i in range(k):
        for j in range(K):
            # Try both sign conventions
            if labels[:, j].sum() > 0:
                a_pos = roc_auc_score(labels[:, j], chi_all[:, i])
                a_neg = roc_auc_score(labels[:, j], -chi_all[:, i])
                auc_mat[i, j] = max(a_pos, a_neg)
    # Maximise assignment
    row_ind, col_ind = linear_sum_assignment(-auc_mat[:n, :n])
    return auc_mat, col_ind


def empirical_committor(anchors: np.ndarray, bursts: np.ndarray,
                        basin_masks: list[np.ndarray]) -> np.ndarray:
    """
    For each anchor, compute probability that a burst ends in each basin.
    anchors : (N, d), bursts : (N, K, d)
    basin_masks : list of (N*K,) bool arrays over flattened bursts
    Returns (N, n_basins) float array.
    """
    N, K, d = bursts.shape
    bursts_f = bursts.reshape(-1, d)
    probs = np.zeros((N, len(basin_masks)), dtype=np.float32)
    for j, mask in enumerate(basin_masks):
        probs[:, j] = mask.reshape(N, K).mean(axis=1)
    return probs


def gradient_norm_grid(net, x_grid: np.ndarray) -> np.ndarray:
    """Average ||∇chi||_2 over a uniform grid (not training anchors)."""
    import torch
    x = torch.tensor(x_grid, dtype=torch.float32, requires_grad=True)
    y = net(x)
    g = torch.autograd.grad(y.sum(), x)[0]
    return g.detach().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Triple-well plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_triple_well(res_path: str, data_path: str):
    print("\n-- Triple-well panels --")
    res  = np.load(res_path, allow_pickle=True)
    data = np.load(data_path)

    variants     = list(res["variants"])
    chi_all      = res["chi_all"]          # (n_var, 5, 5, N_ANC, k)
    val_losses   = res["val_losses"]       # (n_var, 5, 5, 500)
    best_epoch   = res["best_epoch"]
    patch_splits = res["patch_splits"]
    k_val        = int(res["k"][0])

    anchors = data["anchors"]              # (N_ANC, 2)
    bursts  = data["bursts"]               # (N_ANC, N_K, 2)
    wells   = data["wells"]                # (3, 2)
    N_ANC   = len(anchors)

    # Well basin labels: cell in basin i if within 0.5 of well i
    def well_mask_flat(wi):
        b_flat = bursts.reshape(-1, 2)
        return np.linalg.norm(b_flat - wells[wi], axis=1) < 0.6

    labels = np.column_stack([well_mask_flat(i) for i in range(3)])  # (N_ANC*N_K, 3)
    # Average to anchor level: prob of ending in well i
    label_anchors = labels.reshape(N_ANC, bursts.shape[1], 3).mean(1)  # (N_ANC, 3)

    # Separatrix proxy: anchors where max prob < 0.6
    sep_mask = label_anchors.max(axis=1) < 0.6

    # ── Panel 1: Convergence ────────────────────────────────────────────────
    n_v = len(variants)
    fig, axes = plt.subplots(1, n_v, figsize=(n_v * 3.5, 3.5))
    if n_v == 1: axes = [axes]
    for v_i, (var, ax) in enumerate(zip(variants, axes)):
        color = VARIANT_COLORS.get(var, "gray")
        for ss in range(5):
            for ts in range(5):
                vl = val_losses[v_i, ss, ts]
                ep = best_epoch[v_i, ss, ts]
                vl_plot = vl.copy(); vl_plot[ep+1:] = np.nan
                ax.plot(np.where(np.isfinite(vl_plot), vl_plot, np.nan),
                        alpha=0.25, color=color, lw=0.8)
                ax.plot(ep, vl[ep], "*", color=color, ms=6, alpha=0.8)
        ax.set_title(VARIANT_LABELS.get(var, var), fontsize=9)
        ax.set_xlabel("Epoch")
        if var != "vamp2":
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        if v_i == 0: ax.set_ylabel("Val loss")
    plt.suptitle("Triple-well — Panel 1: Convergence", fontsize=11)
    plt.tight_layout()
    savefig(fig, "triple_well_panel1_convergence.png")

    # ── Pre-panel sanity: empirical isocommittor ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    basin_names = ["Well A\n(-1.2,0)", "Well B\n(1.2,0)", "Well C\n(0,1.5)"]
    for j, ax in enumerate(axes):
        sc_ = ax.scatter(anchors[:, 0], anchors[:, 1], c=label_anchors[:, j],
                         cmap="Blues", vmin=0, vmax=1, s=8, rasterized=True)
        plt.colorbar(sc_, ax=ax, shrink=0.8)
        ax.scatter(wells[:, 0], wells[:, 1], marker="*", s=120, c="gold",
                   edgecolors="black", zorder=5)
        ax.scatter(anchors[sep_mask, 0], anchors[sep_mask, 1],
                   s=5, c="red", alpha=0.5, label=f"sep ({sep_mask.sum()})")
        ax.set_title(f"P(burst→{basin_names[j]})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.suptitle("Triple-well — Pre-panel sanity: empirical committor", fontsize=11)
    plt.tight_layout()
    savefig(fig, "triple_well_sanity_committor.png")
    print(f"  Separatrix cells: {sep_mask.sum()}/{N_ANC}")

    # ── Panel 2b: AU-ROC boxplots ──────────────────────────────────────────
    labels_bin = (label_anchors > 0.5).astype(int)  # hard binary labels per anchor
    auc_data   = {var: [] for var in variants}

    for v_i, var in enumerate(variants):
        for ss in range(5):
            for ts in range(5):
                chi = chi_all[v_i, ss, ts]   # (N_ANC, k)
                try:
                    auc_mat, assign = hungarian_auc(chi, labels_bin)
                    assigned_aucs   = [auc_mat[i, assign[i]] for i in range(min(k_val, 3))]
                    auc_data[var].append(np.mean(assigned_aucs))
                except Exception:
                    auc_data[var].append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(variants))
    for pos, var in zip(positions, variants):
        vals = [v for v in auc_data[var] if np.isfinite(v)]
        color = VARIANT_COLORS.get(var, "gray")
        if vals:
            bp = ax.boxplot(vals, positions=[pos], widths=0.5,
                           patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.7),
                           medianprops=dict(color="black", lw=2))
    ax.set_xticks(positions)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=30, ha="right")
    ax.set_ylabel("Mean AU-ROC (Hungarian assignment)")
    ax.set_title("Triple-well — Panel 2b: AU-ROC per variant")
    ax.axhline(0.9, ls="--", c="gray", lw=1, label="0.9 threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig, "triple_well_panel2b_auroc.png")

    # ── Panel 2c: Separatrix MAE ───────────────────────────────────────────
    sep_mae = {var: [] for var in variants}
    for v_i, var in enumerate(variants):
        for ss in range(5):
            for ts in range(5):
                chi = chi_all[v_i, ss, ts]   # (N_ANC, k)
                chi_sep = chi[sep_mask]       # should be near 0.5 per column
                mae = np.mean(np.abs(chi_sep - 0.5))
                sep_mae[var].append(mae)

    fig, ax = plt.subplots(figsize=(8, 4))
    for pos, var in zip(positions, variants):
        vals  = sep_mae[var]
        color = VARIANT_COLORS.get(var, "gray")
        ax.boxplot(vals, positions=[pos], widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="black", lw=2))
    ax.set_xticks(positions); ax.set_xticklabels(
        [VARIANT_LABELS.get(v,v) for v in variants], rotation=30, ha="right")
    ax.set_ylabel("MAE from 0.5 on separatrix")
    ax.set_title("Triple-well — Panel 2c: Separatrix score")
    plt.tight_layout()
    savefig(fig, "triple_well_panel2c_sep.png")

    # ── Panel 3: Learned slow modes (best run per variant) ─────────────────
    fig, axes = plt.subplots(n_v, k_val, figsize=(k_val * 3.5, n_v * 3))
    if n_v == 1: axes = axes.reshape(1, -1)

    xg = np.linspace(data["grid_lo"][0], data["grid_hi"][0], 80)
    yg = np.linspace(data["grid_lo"][1], data["grid_hi"][1], 80)

    for v_i, var in enumerate(variants):
        # Best run = lowest median val loss across split/train seeds
        med_val = np.nanmedian(val_losses[v_i].reshape(25, -1), axis=1)
        best_flat = int(np.nanargmin(med_val))
        ss, ts    = divmod(best_flat, 5)
        chi       = chi_all[v_i, ss, ts]   # (N_ANC, k)

        for ki in range(k_val):
            ax = axes[v_i, ki]
            sc_ = ax.scatter(anchors[:,0], anchors[:,1], c=chi[:,ki],
                             cmap="coolwarm", vmin=0, vmax=1, s=6, rasterized=True)
            plt.colorbar(sc_, ax=ax, shrink=0.7)
            ax.scatter(wells[:,0], wells[:,1], marker="*", s=120,
                       c="gold", edgecolors="black", zorder=5)
            if v_i == 0: ax.set_title(f"χ_{ki+1}")
            if ki == 0:  ax.set_ylabel(VARIANT_LABELS.get(var, var), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Triple-well — Panel 3: Learned slow modes (best seed per variant)", fontsize=11)
    plt.tight_layout()
    savefig(fig, "triple_well_panel3_modes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Alanine dipeptide plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_alanine(res_path: str, data_path: str):
    print("\n-- Alanine dipeptide panels --")
    res  = np.load(res_path, allow_pickle=True)
    data = np.load(data_path)

    variants     = list(res["variants"])
    chi_all      = res["chi_all"]
    val_losses   = res["val_losses"]
    best_epoch   = res["best_epoch"]
    k_val        = int(res["k"][0])

    phi_anc = data["anchors_phi"]           # (N_ANC,)
    psi_anc = data["anchors_psi"]
    bursts_phi = data["bursts_phi"]         # (N_ANC, N_K)
    bursts_psi = data["bursts_psi"]
    N_ANC = len(phi_anc)
    N_K   = bursts_phi.shape[1]

    # Basin definitions (degrees → radians)
    def in_c7eq(phi, psi):
        return ((phi > np.radians(-100)) & (phi < np.radians(-60)) &
                (psi > np.radians( 60))  & (psi < np.radians(100)))
    def in_c7ax(phi, psi):
        return ((phi > np.radians(40)) & (phi < np.radians(80)) &
                (psi > np.radians(-100)) & (psi < np.radians(-60)))
    def in_c7eq2(phi, psi):
        return ((phi > np.radians(-170)) & (phi < np.radians(-130)) &
                (psi > np.radians(140)) & (psi < np.radians(180)))

    # Empirical committor on burst endpoints
    bp_flat = bursts_phi.ravel()
    bq_flat = bursts_psi.ravel()
    labels_flat = np.column_stack([
        in_c7eq(bp_flat,  bq_flat).astype(float),
        in_c7ax(bp_flat,  bq_flat).astype(float),
        in_c7eq2(bp_flat, bq_flat).astype(float),
    ])   # (N_ANC*N_K, 3)
    label_anchors = labels_flat.reshape(N_ANC, N_K, 3).mean(1)  # (N_ANC, 3)
    sep_mask      = label_anchors.max(axis=1) < 0.6

    # ── Sanity: empirical committor on φ,ψ ────────────────────────────────
    basin_names = ["C7eq", "C7ax", "C7eq'"]
    fig, axes   = plt.subplots(1, 3, figsize=(13, 4))
    for j, ax in enumerate(axes):
        sc_ = ax.scatter(np.degrees(phi_anc), np.degrees(psi_anc),
                         c=label_anchors[:, j], cmap="Blues",
                         vmin=0, vmax=1, s=8, rasterized=True)
        plt.colorbar(sc_, ax=ax, shrink=0.8, label="Committor prob")
        ax.scatter(np.degrees(phi_anc[sep_mask]), np.degrees(psi_anc[sep_mask]),
                   s=5, c="red", alpha=0.4)
        ax.set_xlabel("φ (°)"); ax.set_ylabel("ψ (°)")
        ax.set_title(f"P(burst→{basin_names[j]})")
    plt.suptitle("Alanine dipeptide — sanity: empirical committor", fontsize=11)
    plt.tight_layout()
    savefig(fig, "alanine_sanity_committor.png")
    print(f"  Separatrix cells: {sep_mask.sum()}/{N_ANC}")

    # ── Panel 1: Convergence ───────────────────────────────────────────────
    n_v = len(variants)
    fig, axes = plt.subplots(1, n_v, figsize=(n_v * 3.5, 3.5))
    if n_v == 1: axes = [axes]
    for v_i, (var, ax) in enumerate(zip(variants, axes)):
        color = VARIANT_COLORS.get(var, "gray")
        for ss in range(5):
            for ts in range(5):
                vl = val_losses[v_i, ss, ts]
                ep = best_epoch[v_i, ss, ts]
                vl_p = vl.copy(); vl_p[ep+1:] = np.nan
                ax.plot(vl_p, alpha=0.25, color=color, lw=0.8)
                ax.plot(ep, vl[ep], "*", color=color, ms=6, alpha=0.8)
        ax.set_title(VARIANT_LABELS.get(var, var), fontsize=9)
        ax.set_xlabel("Epoch")
        # VAMP2 val_loss is negative; log-scale only for positive-loss variants
        if var != "vamp2":
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        if v_i == 0: ax.set_ylabel("Val loss")
    plt.suptitle("Alanine dipeptide — Panel 1: Convergence", fontsize=11)
    plt.tight_layout()
    savefig(fig, "alanine_panel1_convergence.png")

    # ── Panel 2b: AU-ROC ──────────────────────────────────────────────────
    # At 450 K / 5 ps lag the system is so dynamic that no anchor reaches
    # p > 0.5 for any basin.  Use median-split labels (top 50% per basin)
    # so AU-ROC is still interpretable as "does chi rank basin-rich anchors
    # above basin-poor anchors?".
    labels_bin = np.zeros(label_anchors.shape, dtype=int)
    for j in range(label_anchors.shape[1]):
        labels_bin[:, j] = (label_anchors[:, j] > np.median(label_anchors[:, j])).astype(int)
    print(f"  Alanine basin positives (median-split): {labels_bin.sum(0)}")

    auc_data   = {var: [] for var in variants}
    for v_i, var in enumerate(variants):
        for ss in range(5):
            for ts in range(5):
                chi = chi_all[v_i, ss, ts]
                try:
                    auc_mat, assign = hungarian_auc(chi, labels_bin)
                    mean_auc = np.mean([auc_mat[i, assign[i]] for i in range(min(k_val,3))])
                    auc_data[var].append(mean_auc)
                except Exception:
                    auc_data[var].append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(n_v)
    for pos, var in zip(positions, variants):
        vals  = [v for v in auc_data[var] if np.isfinite(v)]
        color = VARIANT_COLORS.get(var, "gray")
        if vals:
            ax.boxplot(vals, positions=[pos], widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.7),
                       medianprops=dict(color="black", lw=2))
    ax.set_xticks(positions); ax.set_xticklabels(
        [VARIANT_LABELS.get(v,v) for v in variants], rotation=30, ha="right")
    ax.set_ylabel("Mean AU-ROC (Hungarian assignment)")
    ax.set_title("Alanine dipeptide — Panel 2b: AU-ROC per variant")
    ax.axhline(0.9, ls="--", c="gray", lw=1, label="0.9")
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig, "alanine_panel2b_auroc.png")

    # ── Panel 2c: Separatrix MAE ──────────────────────────────────────────
    sep_mae = {var: [] for var in variants}
    for v_i, var in enumerate(variants):
        for ss in range(5):
            for ts in range(5):
                chi_sep = chi_all[v_i, ss, ts][sep_mask]
                sep_mae[var].append(np.mean(np.abs(chi_sep - 0.5)))

    fig, ax = plt.subplots(figsize=(8, 4))
    for pos, var in zip(positions, variants):
        color = VARIANT_COLORS.get(var, "gray")
        ax.boxplot(sep_mae[var], positions=[pos], widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="black", lw=2))
    ax.set_xticks(positions); ax.set_xticklabels(
        [VARIANT_LABELS.get(v,v) for v in variants], rotation=30, ha="right")
    ax.set_ylabel("MAE from 0.5 on separatrix cells")
    ax.set_title("Alanine dipeptide — Panel 2c: Separatrix score")
    plt.tight_layout()
    savefig(fig, "alanine_panel2c_sep.png")

    # ── Panel 3: Learned slow modes on φ,ψ (best seed per variant) ────────
    fig, axes = plt.subplots(n_v, k_val, figsize=(k_val * 3.5, n_v * 3))
    if n_v == 1: axes = axes.reshape(1, -1)
    for v_i, var in enumerate(variants):
        med_val   = np.nanmedian(val_losses[v_i].reshape(25, -1), axis=1)
        best_flat = int(np.nanargmin(med_val))
        ss, ts    = divmod(best_flat, 5)
        chi       = chi_all[v_i, ss, ts]
        for ki in range(k_val):
            ax = axes[v_i, ki]
            c_col = chi[:, ki]
            # Adaptive colorscale: show actual range so small variations are visible
            c_min, c_max = float(c_col.min()), float(c_col.max())
            c_span = c_max - c_min
            if c_span < 1e-4:   # collapsed/trivial: fixed scale
                vm, vx = 0.0, 1.0
            else:
                vm, vx = c_min, c_max
            sc_ = ax.scatter(np.degrees(phi_anc), np.degrees(psi_anc),
                             c=c_col, cmap="coolwarm",
                             vmin=vm, vmax=vx, s=6, rasterized=True)
            cb = plt.colorbar(sc_, ax=ax, shrink=0.7)
            cb.set_label(f"[{vm:.2f},{vx:.2f}]", fontsize=6)
            if v_i == 0: ax.set_title(f"χ_{ki+1}")
            if ki == 0:  ax.set_ylabel(VARIANT_LABELS.get(var, var), fontsize=8)
            ax.set_xlabel("φ (°)"); ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("AD — Panel 3: Learned slow modes (best seed per variant)", fontsize=11)
    plt.tight_layout()
    savefig(fig, "alanine_panel3_modes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tw_res  = os.path.join(RESULTS_DIR, "triple_well_results.npz")
    tw_data = os.path.join(DATA_DIR,    "triple_well_koopman.npz")
    al_res  = os.path.join(RESULTS_DIR, "alanine_results.npz")
    al_data = os.path.join(DATA_DIR,    "alanine_koopman.npz")

    if os.path.exists(tw_res):
        plot_triple_well(tw_res, tw_data)
    else:
        print(f"Triple-well results not found: {tw_res}")

    if os.path.exists(al_res):
        plot_alanine(al_res, al_data)
    else:
        print(f"Alanine results not found: {al_res}")
