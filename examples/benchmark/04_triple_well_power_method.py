"""
Benchmark the LARRY ISOKANN implementation (power_method_multi / ChiNetMultiRaw)
on the triple-well dataset and compare to the isotarget variants from Study 02.

Architecture: ChiNetMultiRaw, hidden=[512,256,128], sigmoid output  (LARRY default)
vs Study 02:  3×64 sigmoid (benchmark spec)

This answers: does the SVD-deflation power method work as well as the
isotarget-based variants on a system where ground truth is known?

Outputs
-------
  results/triple_well_power_method.npz
  figures/triple_well_power_method_comparison.png
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from amore.isokann import ChiNetMultiRaw, power_method_multi

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)

K           = 3        # 3 basins → k=3 eigenfunctions
N_POWER_ITER = 80
EPOCHS_PER_ITER = 400
LR          = 2e-3
LR_DECAY    = 0.97
N_SEEDS     = 5
DEVICE      = torch.device("cpu")


def best_auc(scores, labels):
    if labels.sum() < 3 or labels.sum() == len(labels): return float("nan")
    return max(roc_auc_score(labels, scores), roc_auc_score(labels, -scores))


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading triple-well data …")
data = np.load(os.path.join(DATA_DIR, "triple_well_koopman.npz"))
anchors = data["anchors"].astype(np.float32)     # (N_ANC, 2)
bursts  = data["bursts"].astype(np.float32)      # (N_ANC, 20, 2)
wells   = data["wells"].astype(np.float32)       # (3, 2)
N_ANC   = len(anchors)
print(f"  {N_ANC} anchors, bursts shape: {bursts.shape}")

# Flatten bursts → Koopman pairs (src = anchor repeated, dst = burst endpoint)
N_K = bursts.shape[1]
x0_all = np.repeat(anchors, N_K, axis=0)         # (N_ANC*N_K, 2)
x1_all = bursts.reshape(-1, 2)                    # (N_ANC*N_K, 2)

x0_t = torch.tensor(x0_all, dtype=torch.float32, device=DEVICE)
x1_t = torch.tensor(x1_all, dtype=torch.float32, device=DEVICE)
x_all_t = torch.tensor(anchors, dtype=torch.float32, device=DEVICE)

# Well labels for AU-ROC
def well_label(wi):
    return (np.linalg.norm(anchors - wells[wi], axis=1) < 0.6).astype(int)
labels = np.column_stack([well_label(i) for i in range(3)])  # (N_ANC, 3)

# SD as collapse detector
def report_sd(chi_np):
    sd = chi_np.std(axis=0)
    status = ["✓" if s > 0.05 else "✗ FLAT" for s in sd]
    return sd, status

# ── Run power_method_multi for N_SEEDS seeds ──────────────────────────────────
chi_seeds = []
auc_seeds = []

print(f"\nRunning power_method_multi (k={K}, {N_POWER_ITER} iters × {EPOCHS_PER_ITER} epochs) …")

for seed in range(N_SEEDS):
    torch.manual_seed(seed * 137)
    np.random.seed(seed * 137)

    net = ChiNetMultiRaw(in_dim=2, k=K, hidden=[512, 256, 128]).to(DEVICE)
    result = power_method_multi(
        net, x0_t, x1_t,
        n_iter=N_POWER_ITER,
        epochs_per_iter=EPOCHS_PER_ITER,
        lr=LR, lr_decay=LR_DECAY,
        verbose=False,
    )

    net.eval()
    with torch.no_grad():
        chi = net(x_all_t).cpu().numpy()   # (N_ANC, K)

    sd, status = report_sd(chi)
    auc_per_well = [max(best_auc(chi[:, m], labels[:, w]) for m in range(K))
                    for w in range(3)]
    mean_auc = np.nanmean(auc_per_well)

    chi_seeds.append(chi)
    auc_seeds.append(auc_per_well)
    print(f"  Seed {seed}: loss={result['losses'][-1]:.5f}  "
          f"SD={sd.round(3)}  {status}  "
          f"mean_AUC={mean_auc:.3f}  per_well={[f'{a:.3f}' for a in auc_per_well]}")

chi_seeds = np.array(chi_seeds)   # (N_SEEDS, N_ANC, K)
auc_seeds = np.array(auc_seeds)   # (N_SEEDS, 3)

print(f"\nSummary over {N_SEEDS} seeds:")
print(f"  Mean AUC per well: {auc_seeds.mean(0).round(3)}")
print(f"  Median mean AUC:   {np.median(auc_seeds.mean(1)):.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(os.path.join(RESULTS_DIR, "triple_well_power_method.npz"),
         chi_seeds=chi_seeds, auc_seeds=auc_seeds, wells=wells)

# ── Load isotarget results for comparison ─────────────────────────────────────
iso_path = os.path.join(RESULTS_DIR, "triple_well_results.npz")
has_iso  = os.path.exists(iso_path)

if has_iso:
    res = np.load(iso_path, allow_pickle=True)
    variants  = list(res["variants"])
    chi_iso   = res["chi_all"]        # (n_var, 5, 5, N_ANC, k)
    val_losses = res["val_losses"]

    # Compute mean AUC per variant (best seed)
    iso_aucs = {}
    for v_i, var in enumerate(variants):
        med_val = np.nanmedian(val_losses[v_i].reshape(25, -1), axis=1)
        best_flat = int(np.nanargmin(med_val))
        ss, ts = divmod(best_flat, 5)
        chi_v = chi_iso[v_i, ss, ts]   # (N_ANC, k)
        k_v   = chi_v.shape[1]
        aucs  = [max(best_auc(chi_v[:, m], labels[:, w]) for m in range(k_v))
                 for w in range(3)]
        iso_aucs[var] = aucs

# ── Comparison plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
well_names = ["Well A\n(-1.2,0)", "Well B\n(1.2,0)", "Well C\n(0,1.5)"]

for w_i, (ax, wname) in enumerate(zip(axes, well_names)):
    # power_method_multi results
    pm_aucs = auc_seeds[:, w_i]
    ax.boxplot([pm_aucs], positions=[0], widths=0.5,
               patch_artist=True, boxprops=dict(facecolor="#9467bd", alpha=0.7),
               medianprops=dict(color="black", lw=2))
    ax.text(0, pm_aucs.mean(), f"{pm_aucs.mean():.3f}", ha="center",
            va="bottom", fontsize=8, color="#9467bd")

    # Isotarget variants
    if has_iso:
        ISOTARGET_COLORS = {
            "shiftscale":"#1f77b4","isa":"#ff7f0e","gramschmidt":"#2ca02c",
            "pseudoinv":"#d62728","cross":"#9467bd","vamp2":"#8c564b",
        }
        for pos, (var, aucs) in enumerate(iso_aucs.items(), start=1):
            ax.bar(pos, aucs[w_i], 0.6,
                   color=ISOTARGET_COLORS.get(var, "gray"), alpha=0.7,
                   label=var if w_i == 0 else "")
            ax.text(pos, aucs[w_i] + 0.005, f"{aucs[w_i]:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_title(f"AU-ROC — {wname}")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, ls="--", c="gray", lw=1)
    ax.set_xticks([0] + list(range(1, 1+len(iso_aucs))) if has_iso else [0])
    xlabels = ["power\nmethod"] + (list(iso_aucs.keys()) if has_iso else [])
    ax.set_xticklabels(xlabels, fontsize=7, rotation=30, ha="right")

if has_iso:
    axes[0].legend(fontsize=7, loc="lower left")

plt.suptitle(f"Triple-well: power_method_multi (LARRY impl) vs isotarget variants\n"
             f"power_method: k={K}, [512,256,128] arch, {N_SEEDS} seeds", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "triple_well_power_method_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: triple_well_power_method_comparison.png")

# ── Best-seed chi plot ────────────────────────────────────────────────────────
best_seed_idx = int(np.argmax(auc_seeds.mean(1)))
chi_best = chi_seeds[best_seed_idx]   # (N_ANC, K)
sd_best, _ = report_sd(chi_best)

fig, axes = plt.subplots(1, K, figsize=(K*4, 3.5))
for ki, ax in enumerate(axes):
    sc = ax.scatter(anchors[:, 0], anchors[:, 1], c=chi_best[:, ki],
                    cmap="coolwarm", vmin=0, vmax=1, s=8, rasterized=True)
    plt.colorbar(sc, ax=ax)
    ax.scatter(wells[:, 0], wells[:, 1], marker="*", s=120, c="gold",
               edgecolors="black", zorder=5)
    ax.set_title(f"χ_{ki+1}  SD={sd_best[ki]:.3f}")
plt.suptitle(f"power_method_multi — best seed (mean AUC={auc_seeds[best_seed_idx].mean():.3f})", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "triple_well_power_method_modes.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: triple_well_power_method_modes.png")

print("\nDone.")
