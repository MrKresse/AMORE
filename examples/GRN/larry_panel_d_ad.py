"""
AD Panel D — (chi_1, chi_2) scatter for each surviving variant,
coloured by literature phi/psi basin membership.

Uses alanine_results.npz from the benchmark.
Expected result: chi functions are near-trivial at 450 K / 5 ps lag,
so basin separation will be absent — this is the negative control.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = os.path.dirname(__file__)
BENCH  = os.path.join(BASE, "..", "benchmark", "AMORE", "examples", "benchmark")
# Correct path to benchmark results
BENCH  = os.path.join(os.path.dirname(BASE), "..", "..",
                      "AMORE", "examples", "benchmark")
BENCH  = os.path.abspath(os.path.join(BASE, "..", "..", "examples", "benchmark"))

RESULTS = os.path.join(BENCH, "results")
DATA    = os.path.join(BENCH, "data")
FIGS    = os.path.join(BENCH, "figures")
os.makedirs(FIGS, exist_ok=True)

print(f"Looking for AD results in: {RESULTS}")

al_path  = os.path.join(RESULTS, "alanine_results.npz")
dat_path = os.path.join(DATA,    "alanine_koopman.npz")

if not os.path.exists(al_path):
    print(f"AD results not found: {al_path}")
    exit(0)

res  = np.load(al_path, allow_pickle=True)
data = np.load(dat_path)

variants   = list(res["variants"])
chi_all    = res["chi_all"]          # (n_var, 5, 5, N_ANC, k)
val_losses = res["val_losses"]
k_val      = int(res["k"][0])

phi_anc = data["anchors_phi"]        # (N_ANC,)
psi_anc = data["anchors_psi"]
N_ANC   = len(phi_anc)

# Basin definitions (radians)
def in_c7eq(phi, psi):
    return ((phi > np.radians(-100)) & (phi < np.radians(-60)) &
            (psi > np.radians(60))  & (psi < np.radians(100)))
def in_c7ax(phi, psi):
    return ((phi > np.radians(40)) & (phi < np.radians(80)) &
            (psi > np.radians(-100)) & (psi < np.radians(-60)))
def in_c7eq2(phi, psi):
    return ((phi > np.radians(-170)) & (phi < np.radians(-130)) &
            (psi > np.radians(140)) & (psi < np.radians(180)))

basin_masks = [in_c7eq(phi_anc, psi_anc),
               in_c7ax(phi_anc, psi_anc),
               in_c7eq2(phi_anc, psi_anc)]
basin_names = ["C7eq", "C7ax", "C7eq'"]
basin_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
neither_mask = ~(basin_masks[0] | basin_masks[1] | basin_masks[2])

VARIANT_NAMES = {
    "shiftscale": "V1-ShiftScale", "isa": "V2-ISA",
    "gramschmidt": "V3-GramSchmidt", "pseudoinv": "V4-PseudoInv",
    "cross": "V5-Cross", "vamp2": "B-VAMP2",
}

# Pick best seed per variant (min median val loss)
n_var = len(variants)

fig, axes = plt.subplots(n_var, 3, figsize=(10, n_var * 3))
if n_var == 1: axes = axes.reshape(1, -1)

for v_i, var in enumerate(variants):
    # Best seed
    med_val = np.nanmedian(val_losses[v_i].reshape(25, -1), axis=1)
    best_flat = int(np.nanargmin(med_val))
    ss, ts = divmod(best_flat, 5)
    chi = chi_all[v_i, ss, ts]               # (N_ANC, k)

    # Effective chi range
    c1_range = chi[:, 0].max() - chi[:, 0].min() if k_val > 0 else 0
    c2_range = chi[:, 1].max() - chi[:, 1].min() if k_val > 1 else 0

    label_name = VARIANT_NAMES.get(var, var)

    for b_i, (mask, bname, bcol) in enumerate(zip(basin_masks, basin_names, basin_colors)):
        ax = axes[v_i, b_i]
        if k_val >= 2:
            ax.scatter(chi[neither_mask, 0], chi[neither_mask, 1],
                       s=2, c="#cccccc", alpha=0.3, rasterized=True, label="other")
            ax.scatter(chi[mask, 0], chi[mask, 1],
                       s=8, c=bcol, alpha=0.7, rasterized=True, label=bname)
            ax.set_xlabel(f"χ₁ [{chi[:,0].min():.2f},{chi[:,0].max():.2f}]", fontsize=8)
            ax.set_ylabel(f"χ₂ [{chi[:,1].min():.2f},{chi[:,1].max():.2f}]", fontsize=8)
        else:
            ax.text(0.5, 0.5, "k=1 only", transform=ax.transAxes, ha="center")
        ax.set_title(f"{label_name}\n{bname} ({mask.sum()} cells)", fontsize=8)

plt.suptitle("AD Panel D — (χ₁, χ₂) scatter per variant, coloured by basin\n"
             "(450 K, 5 ps lag: expect no basin separation — negative control)",
             fontsize=10)
plt.tight_layout()
out_path = os.path.join(FIGS, "panel_d_ad_chi_scatter.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# Summary
print("\nAD chi variation summary:")
for v_i, var in enumerate(variants):
    med_val = np.nanmedian(val_losses[v_i].reshape(25, -1), axis=1)
    best_flat = int(np.nanargmin(med_val))
    ss, ts = divmod(best_flat, 5)
    chi = chi_all[v_i, ss, ts]
    ranges = [chi[:, m].max() - chi[:, m].min() for m in range(min(k_val, 3))]
    print(f"  {VARIANT_NAMES.get(var,var):20s}  chi ranges: {[f'{r:.4f}' for r in ranges]}")
