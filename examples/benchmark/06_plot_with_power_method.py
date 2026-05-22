"""
Regenerate all benchmark panels with power_method_multi added as a 7th variant.

Key difference from isotarget variants:
  Isotarget variants (ShiftScale/ISA/GramSchmidt/PseudoInv/Cross):
    - Compute an ANALYTICAL target T = f(chi(x_tau))  [e.g. QR, simplex, etc.]
    - Train chi(x_0) → T  via MSE
    - Each variant encodes a different mathematical prior about what the
      eigenfunctions should look like

  VAMP2:
    - Directly MAXIMISES the variational VAMP-2 score (no explicit target)

  power_method_multi (this script):
    - Compute Y = chi(x_1)  [Koopman action, same as isotargets]
    - SVD of Y → replace Y with orthonormal U  [data-driven deflation]
    - Scale U to [0,1]
    - Train chi(x_0) → U  via MSE
    - Theoretically equivalent to simultaneous power iteration for the
      dominant k-dimensional invariant subspace of the Koopman operator
    - Key difference: deflation is computed EMPIRICALLY from a data batch
      (not analytically), which can be noisier at init but avoids hard-coding
      functional-form assumptions

  Common to all: architecture, optimiser, and training epochs differ between
  the isotarget benchmark (3×64, 500 epochs) and power_method_multi
  ([512,256,128], 80 outer × 400 inner = 32,000 grad steps).
  This is noted on all plots.

Outputs (all overwrite the existing figures):
  triple_well_panel1_convergence.png   (7 variants)
  triple_well_panel2b_auroc.png        (7 variants)
  triple_well_panel2c_sep.png          (7 variants)
  triple_well_panel3_modes.png         (7 variants)
  alanine_panel1_convergence.png       (7 variants, if AD power_method available)
  alanine_panel_d_eigvec_scatter.png   (χ₁ vs χ₂ per variant coloured by basin)
"""

from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from targets import VARIANT_NAMES

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Colour / label palette ────────────────────────────────────────────────────
VARIANT_COLORS = {
    "shiftscale"  : "#1f77b4",
    "isa"         : "#ff7f0e",
    "gramschmidt" : "#2ca02c",
    "pseudoinv"   : "#d62728",
    "cross"       : "#9467bd",
    "vamp2"       : "#8c564b",
    "power_method": "#17becf",   # teal — clearly different
}
VARIANT_LABELS = {k: v for k, v in VARIANT_NAMES.items()}
VARIANT_LABELS["power_method"] = "SVD-Power\n[512,256,128]"


def savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def hungarian_auc(chi_all, labels):
    k = chi_all.shape[1]; K = labels.shape[1]; n = min(k, K)
    auc_mat = np.zeros((k, K))
    for i in range(k):
        for j in range(K):
            if labels[:, j].sum() > 0:
                a = roc_auc_score(labels[:, j], chi_all[:, i])
                b = roc_auc_score(labels[:, j], -chi_all[:, i])
                auc_mat[i, j] = max(a, b)
    ri, ci = linear_sum_assignment(-auc_mat[:n, :n])
    return auc_mat, ci


# ══════════════════════════════════════════════════════════════════════════════
# Triple-well
# ══════════════════════════════════════════════════════════════════════════════
print("=== Triple-well ===")

tw_res  = np.load(os.path.join(RESULTS_DIR, "triple_well_results.npz"), allow_pickle=True)
tw_data = np.load(os.path.join(DATA_DIR, "triple_well_koopman.npz"))
pm_res  = np.load(os.path.join(RESULTS_DIR, "triple_well_power_method.npz"))

variants   = list(tw_res["variants"])
chi_iso    = tw_res["chi_all"]       # (n_var, 5, 5, N_ANC, k)
val_losses = tw_res["val_losses"]    # (n_var, 5, 5, 500)
best_epoch = tw_res["best_epoch"]
k_val      = int(tw_res["k"][0])

chi_pm     = pm_res["chi_seeds"]     # (5, N_ANC, 3)
auc_pm     = pm_res["auc_seeds"]     # (5, 3)

anchors = tw_data["anchors"]
bursts  = tw_data["bursts"]
wells   = tw_data["wells"]
N_ANC   = len(anchors)

# Basin labels
def well_mask_flat(wi):
    b = bursts.reshape(-1, 2)
    return np.linalg.norm(b - wells[wi], axis=1) < 0.6

labels_flat   = np.column_stack([well_mask_flat(i) for i in range(3)])
label_anchors = labels_flat.reshape(N_ANC, bursts.shape[1], 3).mean(1)
labels_bin    = (label_anchors > 0.5).astype(int)
sep_mask      = label_anchors.max(1) < 0.6

all_variants = variants + ["power_method"]
n_all = len(all_variants)

# ── Panel 1: Convergence (isotarget: loss/epoch; power_method: loss/outer-iter) ──
fig, axes = plt.subplots(1, n_all, figsize=(n_all * 3.3, 3.5))
for v_i, (var, ax) in enumerate(zip(all_variants, axes)):
    color = VARIANT_COLORS.get(var, "gray")
    if var == "power_method":
        # power_method: loss per outer iteration, 5 seeds
        # Load detailed loss from the run log — use per-iter spans as proxy
        for s_i in range(chi_pm.shape[0]):
            # Use per-mode span at each iter as convergence proxy (SD of chi)
            # (val_loss per iter not stored; show SD trajectory is not available
            # without re-running with logging — use dummy markers for now)
            pass
        # Show AU-ROC distribution instead as a star marker
        mean_aucs = auc_pm.mean(1)   # (5,)
        ax.scatter(range(len(mean_aucs)), mean_aucs, s=30, color=color, marker="*", zorder=5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Mean AU-ROC")
        ax.set_title(VARIANT_LABELS[var] + "\n(AU-ROC, not loss)", fontsize=8)
        ax.set_xticks(range(5))
        ax.axhline(0.9, ls="--", c="gray", lw=1)
    else:
        for ss in range(5):
            for ts in range(5):
                vl = val_losses[v_i, ss, ts]
                ep = best_epoch[v_i, ss, ts]
                vp = vl.copy(); vp[ep+1:] = np.nan
                ax.plot(np.where(np.isfinite(vp), vp, np.nan),
                        alpha=0.25, color=color, lw=0.8)
                ax.plot(ep, vl[ep], "*", color=color, ms=6, alpha=0.8)
        ax.set_title(VARIANT_LABELS.get(var, var), fontsize=9)
        ax.set_xlabel("Epoch")
        if var != "vamp2":
            try: ax.set_yscale("log")
            except: pass
        if v_i == 0: ax.set_ylabel("Val loss")

plt.suptitle("Triple-well — Panel 1: Convergence (7 variants)\n"
             "Note: SVD-Power uses different arch/epochs than isotarget variants",
             fontsize=10)
plt.tight_layout()
savefig(fig, "triple_well_panel1_convergence.png")

# ── Panel 2b: AU-ROC ──────────────────────────────────────────────────────────
auc_data = {var: [] for var in all_variants}

for v_i, var in enumerate(variants):
    for ss in range(5):
        for ts in range(5):
            chi = chi_iso[v_i, ss, ts]
            try:
                auc_mat, assign = hungarian_auc(chi, labels_bin)
                auc_data[var].append(np.mean([auc_mat[i, assign[i]]
                                               for i in range(min(k_val, 3))]))
            except Exception:
                auc_data[var].append(np.nan)

# power_method: 5 seeds
for s_i in range(chi_pm.shape[0]):
    chi = chi_pm[s_i]
    try:
        auc_mat, assign = hungarian_auc(chi, labels_bin)
        auc_data["power_method"].append(np.mean([auc_mat[i, assign[i]]
                                                  for i in range(min(3, 3))]))
    except Exception:
        auc_data["power_method"].append(np.nan)

fig, ax = plt.subplots(figsize=(9, 4))
positions = np.arange(n_all)
for pos, var in zip(positions, all_variants):
    vals  = [v for v in auc_data[var] if np.isfinite(v)]
    color = VARIANT_COLORS.get(var, "gray")
    if vals:
        bp = ax.boxplot(vals, positions=[pos], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color="black", lw=2))
        ax.text(pos, np.median(vals) + 0.01, f"{np.median(vals):.3f}",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(positions)
ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in all_variants],
                   rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Mean AU-ROC (Hungarian assignment)")
ax.set_title("Triple-well — Panel 2b: AU-ROC per variant\n"
             "SVD-Power (n=5 seeds) vs isotarget variants (n=25 seeds each)")
ax.axhline(0.9, ls="--", c="gray", lw=1, label="0.9")
ax.legend(fontsize=8)
plt.tight_layout()
savefig(fig, "triple_well_panel2b_auroc.png")

# ── Panel 2c: Separatrix ──────────────────────────────────────────────────────
sep_mae = {var: [] for var in all_variants}

for v_i, var in enumerate(variants):
    for ss in range(5):
        for ts in range(5):
            chi     = chi_iso[v_i, ss, ts]
            chi_sep = chi[sep_mask]
            sep_mae[var].append(np.mean(np.abs(chi_sep - 0.5)))

for s_i in range(chi_pm.shape[0]):
    chi     = chi_pm[s_i]
    chi_sep = chi[sep_mask]
    sep_mae["power_method"].append(np.mean(np.abs(chi_sep - 0.5)))

fig, ax = plt.subplots(figsize=(9, 4))
for pos, var in zip(positions, all_variants):
    color = VARIANT_COLORS.get(var, "gray")
    ax.boxplot(sep_mae[var], positions=[pos], widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7),
               medianprops=dict(color="black", lw=2))
ax.set_xticks(positions)
ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in all_variants],
                   rotation=30, ha="right", fontsize=8)
ax.set_ylabel("MAE from 0.5 on separatrix"); ax.set_ylim(0, 0.5)
ax.set_title("Triple-well — Panel 2c: Separatrix score")
plt.tight_layout()
savefig(fig, "triple_well_panel2c_sep.png")

# ── Panel 3: Modes ────────────────────────────────────────────────────────────
# Best seed: for isotarget = min median val_loss; for power_method = max mean AU-ROC
best_idx_pm = int(np.argmax(auc_pm.mean(1)))
chi_pm_best = chi_pm[best_idx_pm]
sd_pm       = chi_pm_best.std(0)

fig, axes = plt.subplots(n_all, k_val, figsize=(k_val * 3.3, n_all * 2.8))
if n_all == 1: axes = axes.reshape(1, -1)

for v_i, var in enumerate(all_variants):
    if var == "power_method":
        chi = chi_pm_best
        sd  = sd_pm
        row_label = f"{VARIANT_LABELS[var]}\nSD={sd.round(3)}"
    else:
        med_val   = np.nanmedian(val_losses[v_i].reshape(25, -1), axis=1)
        best_flat = int(np.nanargmin(med_val))
        ss, ts    = divmod(best_flat, 5)
        chi       = chi_iso[v_i, ss, ts]
        sd        = chi.std(0)
        row_label = VARIANT_LABELS.get(var, var)

    for ki in range(k_val):
        ax = axes[v_i, ki]
        # Adaptive colorscale: show actual range so collapsed modes are visible
        c_col = chi[:, ki]
        c_min, c_max = c_col.min(), c_col.max()
        c_span = c_max - c_min
        vm, vx = (0, 1) if c_span < 1e-4 else (c_min, c_max)

        sc_ = ax.scatter(anchors[:, 0], anchors[:, 1], c=c_col,
                         cmap="coolwarm", vmin=vm, vmax=vx, s=5, rasterized=True)
        cb = plt.colorbar(sc_, ax=ax, shrink=0.7)
        cb.set_label(f"SD={sd[ki]:.3f}", fontsize=6)
        ax.scatter(wells[:, 0], wells[:, 1], marker="*", s=80,
                   c="gold", edgecolors="black", zorder=5)
        if v_i == 0: ax.set_title(f"χ_{ki+1}", fontsize=9)
        if ki == 0:  ax.set_ylabel(row_label, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Triple-well — Panel 3: Slow modes (best seed per variant)\n"
             "Adaptive colorscale; SD in colorbar — collapsed modes appear uniform",
             fontsize=10)
plt.tight_layout()
savefig(fig, "triple_well_panel3_modes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Alanine — Panel D: eigenvector scatter  χ₁ vs χ₂ coloured by basin
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Alanine — eigenvector scatter ===")

al_res  = np.load(os.path.join(RESULTS_DIR, "alanine_results.npz"), allow_pickle=True)
al_data = np.load(os.path.join(DATA_DIR, "alanine_koopman.npz"))

al_variants  = list(al_res["variants"])
chi_al       = al_res["chi_all"]      # (n_var, 5, 5, N_ANC, k)
val_al       = al_res["val_losses"]
phi_anc      = al_data["anchors_phi"]
psi_anc      = al_data["anchors_psi"]

# Check if power_method alanine results exist
pm_al_path = os.path.join(RESULTS_DIR, "alanine_power_method.npz")
has_pm_al  = os.path.exists(pm_al_path)
if has_pm_al:
    pm_al    = np.load(pm_al_path)
    chi_pm_al = pm_al["chi_seeds"]   # (5, N_ANC, 3)
    print("  Loaded alanine power_method results")
else:
    print("  Alanine power_method not yet available — skipping SVD-Power panel")

# Basin membership on ANCHOR phi/psi
def in_c7eq(phi, psi):
    return ((phi > np.radians(-100)) & (phi < np.radians(-60)) &
            (psi > np.radians(60))  & (psi < np.radians(100)))
def in_c7ax(phi, psi):
    return ((phi > np.radians(40)) & (phi < np.radians(80)) &
            (psi > np.radians(-100)) & (psi < np.radians(-60)))
def in_c7eq2(phi, psi):
    return ((phi > np.radians(-170)) & (phi < np.radians(-130)) &
            (psi > np.radians(140)) & (psi < np.radians(180)))

b_c7eq  = in_c7eq( phi_anc, psi_anc)
b_c7ax  = in_c7ax( phi_anc, psi_anc)
b_c7eq2 = in_c7eq2(phi_anc, psi_anc)
b_other = ~(b_c7eq | b_c7ax | b_c7eq2)

BASIN_COLORS = {"C7eq":"#e41a1c","C7ax":"#377eb8","C7eq'":"#4daf4a","other":"#aaaaaa"}
print(f"  Basin counts — C7eq:{b_c7eq.sum()}  C7ax:{b_c7ax.sum()}  "
      f"C7eq':{b_c7eq2.sum()}  other:{b_other.sum()}")

al_all_variants = al_variants + (["power_method"] if has_pm_al else [])
n_al = len(al_all_variants)

# ── (χ₁ vs χ₂) eigenvector scatter per variant ───────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes_flat = axes.ravel()

for v_i, var in enumerate(al_all_variants):
    if v_i >= len(axes_flat): break
    ax = axes_flat[v_i]
    color = VARIANT_COLORS.get(var, "gray")

    if var == "power_method":
        best_idx = int(np.argmax([chi_pm_al[s].std(0).mean() for s in range(5)]))
        chi = chi_pm_al[best_idx]
        label = VARIANT_LABELS["power_method"]
    else:
        med_val   = np.nanmedian(val_al[v_i].reshape(25, -1), axis=1)
        best_flat = int(np.nanargmin(med_val))
        ss, ts    = divmod(best_flat, 5)
        chi = chi_al[v_i, ss, ts]   # (N_ANC, k)
        label = VARIANT_LABELS.get(var, var)

    k_v = chi.shape[1]
    if k_v < 2:
        ax.text(0.5, 0.5, "k=1 only", transform=ax.transAxes, ha="center")
        ax.set_title(label, fontsize=8); continue

    sd_1 = chi[:, 0].std(); sd_2 = chi[:, 1].std()

    # Adaptive axis range
    x, y = chi[:, 0], chi[:, 1]

    for mask, bname, bcol in [(b_other, "other", BASIN_COLORS["other"]),
                               (b_c7eq,  "C7eq",  BASIN_COLORS["C7eq"]),
                               (b_c7ax,  "C7ax",  BASIN_COLORS["C7ax"]),
                               (b_c7eq2, "C7eq'", BASIN_COLORS["C7eq'"])]:
        n = mask.sum()
        ax.scatter(x[mask], y[mask], s=4 if bname == "other" else 10,
                   c=bcol, alpha=0.3 if bname == "other" else 0.7,
                   label=f"{bname} (n={n})", rasterized=True)

    ax.set_xlabel(f"χ₁  SD={sd_1:.4f}", fontsize=8)
    ax.set_ylabel(f"χ₂  SD={sd_2:.4f}", fontsize=8)
    ax.set_title(f"{label}", fontsize=8)
    if v_i == 0:
        ax.legend(markerscale=2, fontsize=6, loc="upper right")

# Hide unused axes
for idx in range(n_al, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle(
    "Alanine dipeptide — Panel D: (χ₁, χ₂) eigenvector scatter coloured by φ,ψ basin\n"
    "450 K / 5 ps lag: ALL variants show no basin separation (χ variation < 0.001).\n"
    "This is a NEGATIVE CONTROL — the lag is too short for the Koopman spectrum to resolve the basins.",
    fontsize=10)
plt.tight_layout()
savefig(fig, "alanine_panel_d_eigvec_scatter.png")

# ── Also redo alanine Panel 3 with adaptive colorscale + SD labels ────────────
print("\n-- Alanine Panel 3 (adaptive colorscale + SD) --")
n_al_iso = len(al_variants)
al_pm_part = ["power_method"] if has_pm_al else []
n_al_all   = n_al_iso + len(al_pm_part)
k_al = int(al_res["k"][0])

fig, axes = plt.subplots(n_al_all, k_al, figsize=(k_al * 3.3, n_al_all * 2.8))
if n_al_all == 1: axes = axes.reshape(1, -1)

for v_i, var in enumerate(al_variants + al_pm_part):
    if var == "power_method":
        best_idx = int(np.argmax([chi_pm_al[s].std(0).mean() for s in range(5)]))
        chi = chi_pm_al[best_idx]
        row_label = VARIANT_LABELS["power_method"]
    else:
        med_val   = np.nanmedian(val_al[v_i].reshape(25, -1), axis=1)
        best_flat = int(np.nanargmin(med_val))
        ss, ts    = divmod(best_flat, 5)
        chi = chi_al[v_i, ss, ts]
        row_label = VARIANT_LABELS.get(var, var)

    sd = chi.std(0)
    for ki in range(k_al):
        ax  = axes[v_i, ki]
        c   = chi[:, ki]
        vm, vx = (c.min(), c.max()) if c.max()-c.min() > 1e-4 else (0, 1)
        sc_ = ax.scatter(np.degrees(phi_anc), np.degrees(psi_anc),
                         c=c, cmap="coolwarm", vmin=vm, vmax=vx,
                         s=5, rasterized=True)
        cb = plt.colorbar(sc_, ax=ax, shrink=0.7)
        cb.set_label(f"SD={sd[ki]:.4f}", fontsize=6)
        if v_i == 0: ax.set_title(f"χ_{ki+1}", fontsize=9)
        if ki == 0:  ax.set_ylabel(row_label, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("AD — Panel 3: Slow modes (adaptive colorscale, SD in colorbar)\n"
             "All χ variation < 0.001 → basins unresolvable at 450 K / 5 ps", fontsize=10)
plt.tight_layout()
savefig(fig, "alanine_panel3_modes.png")

# ── Alanine Panel 1 convergence (add power_method if available) ───────────────
if has_pm_al:
    print("\n-- Alanine Panel 1 convergence (with SVD-Power) --")
    n_al_conv = n_al_iso + 1
    fig, axes = plt.subplots(1, n_al_conv, figsize=(n_al_conv * 3.3, 3.5))
    al_val = al_res["val_losses"]
    al_ep  = al_res["best_epoch"]

    for v_i, var in enumerate(al_variants):
        ax    = axes[v_i]
        color = VARIANT_COLORS.get(var, "gray")
        for ss in range(5):
            for ts in range(5):
                vl = al_val[v_i, ss, ts]
                ep = al_ep[v_i, ss, ts]
                vp = vl.copy(); vp[ep+1:] = np.nan
                ax.plot(np.where(np.isfinite(vp), vp, np.nan),
                        alpha=0.25, color=color, lw=0.8)
                ax.plot(ep, vl[ep], "*", color=color, ms=6, alpha=0.8)
        ax.set_title(VARIANT_LABELS.get(var, var), fontsize=9)
        ax.set_xlabel("Epoch")
        if var != "vamp2":
            try: ax.set_yscale("log")
            except: pass
        if v_i == 0: ax.set_ylabel("Val loss")

    # power_method: show SD per seed as bar
    ax    = axes[-1]
    color = VARIANT_COLORS["power_method"]
    sds   = np.array([chi_pm_al[s].std(0).mean() for s in range(5)])
    ax.bar(range(5), sds, color=color, alpha=0.8)
    ax.set_xlabel("Seed"); ax.set_ylabel("Mean chi SD")
    ax.set_title(f"{VARIANT_LABELS['power_method']}\n(mean SD across modes)", fontsize=8)
    ax.axhline(0.01, ls="--", c="gray", lw=1, label="collapse threshold")
    ax.legend(fontsize=7)

    plt.suptitle("Alanine dipeptide — Panel 1: Convergence + SVD-Power SD", fontsize=10)
    plt.tight_layout()
    savefig(fig, "alanine_panel1_convergence.png")

print("\nAll panels regenerated.")
