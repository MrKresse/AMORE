"""
Discrete transfer operator baseline for LARRY.

Algorithm
---------
1. k-means cluster cells in 40D PCA space → N_CLUSTERS discrete states
2. Assign each Koopman pair (src, dst) to cluster pair (ci, cj)
3. Build count matrix C[ci, cj]; row-normalise → stochastic T
4. Eigendecompose T; left eigenvectors = discrete Koopman eigenfunctions
5. Assign each cell its cluster's eigenvector value
6. Report:
   - Eigenvalue spectrum (implied timescales)
   - Pearson r against NeuMon fate bias (Tier-2 hard split)
   - AU-ROC against fate labels (Tier-1 sanity)
   - SD of each eigenvector (collapse detection)
   - Compare to ISOKANN neural-network chi

This is a non-parametric baseline — no training, no neural network.
Any ISOKANN result not substantially exceeding this is suspect.

Output
------
  output/benchmark/discrete_transfer_eigenvalues.png
  output/benchmark/discrete_transfer_fate.png
  output/benchmark/discrete_transfer_umap.png
  output/benchmark/discrete_transfer_results.csv
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from scipy.linalg import eig as scipy_eig
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

BASE  = os.path.dirname(__file__)
DATA  = os.path.join(BASE, "data")
OUT   = os.path.join(BASE, "output")
BENCH = os.path.join(OUT, "benchmark")
os.makedirs(BENCH, exist_ok=True)

N_CLUSTERS = 500    # discrete states in PCA space
EPS        = 1e-3   # uniform mixing to handle empty rows
N_EIG      = 20     # eigenvectors to compute
SEED       = 42
FATES      = ["Mast","Baso","Meg","Erythroid","Lymphoid",
              "Neutrophil","Monocyte","Eos","pDC","Ccr7_DC"]
FATE_COLS  = [f"progenitor_{f}" for f in FATES]


def best_auc(scores, labels):
    if labels.sum() < 3 or labels.sum() == len(labels): return float("nan")
    return max(roc_auc_score(labels, scores), roc_auc_score(labels, -scores))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load
# ══════════════════════════════════════════════════════════════════════════════
print("Loading …")
import anndata
adata   = anndata.read_h5ad(os.path.join(DATA, "larry_processed.h5ad"))
obs     = adata.obs.copy()
X_pca   = adata.obsm["X_pca"].astype(np.float32)    # (49116, 40)
X_umap  = adata.obsm["X_umap"].astype(np.float32)
src     = np.load(os.path.join(DATA, "larry_src.npy"))  # (206942,)
dst     = np.load(os.path.join(DATA, "larry_dst.npy"))

state_info = obs["state_info"].values.astype(str)
time_info  = obs["time_info"].astype(str).values
day2_mask  = time_info == "2"

nm_bias = obs["NeuMon_fate_bias"].values.astype(float)
nm_mask = obs["NeuMon_mask"].values.astype(bool) & day2_mask

fate_labels = {f: obs[c].values.astype(float)
               for f, c in zip(FATES, FATE_COLS) if c in obs.columns}

# Also load ISOKANN chi for comparison
chi_iso = np.load(os.path.join(OUT, "multi_chi_all.npy"))  # (49116, 15)

print(f"  {X_pca.shape[0]} cells, {len(src)} pairs")


# ══════════════════════════════════════════════════════════════════════════════
# 2. k-means discretisation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nk-means clustering (k={N_CLUSTERS}) …")
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=3, batch_size=4096)
km.fit(X_pca)
cell_cluster = km.labels_.astype(int)   # (49116,)
print(f"  Cluster sizes: min={np.bincount(cell_cluster).min()}  "
      f"max={np.bincount(cell_cluster).max()}  "
      f"mean={np.bincount(cell_cluster).mean():.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build transition matrix
# ══════════════════════════════════════════════════════════════════════════════
print("Building transition matrix …")
ci = cell_cluster[src]
cj = cell_cluster[dst]

C = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.float64)
np.add.at(C, (ci, cj), 1.0)

row_sums = C.sum(axis=1, keepdims=True)
occupied = row_sums[:, 0] > 0
T        = np.zeros_like(C)
T[occupied] = C[occupied] / row_sums[occupied]
T = (1.0 - EPS) * T + EPS / N_CLUSTERS   # uniform mixing

print(f"  Occupied rows: {occupied.sum()} / {N_CLUSTERS}")
np.save(os.path.join(BENCH, "discrete_T.npy"), T)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Eigendecomposition
# ══════════════════════════════════════════════════════════════════════════════
print("Eigendecomposing T …")
vals, lvecs = scipy_eig(T.T)            # left eigenvectors: π T = λ π
order = np.argsort(vals.real)[::-1]
vals  = vals[order].real
lvecs = lvecs[:, order].real            # (N_CLUSTERS, N_EIG)

print(f"  Top {N_EIG} eigenvalues: {vals[:N_EIG].round(4)}")

# Implied timescales: -1 / log(λ) (in units of Koopman lag)
with np.errstate(divide="ignore", invalid="ignore"):
    its = np.where(vals[1:N_EIG] > 0, -1.0 / np.log(np.abs(vals[1:N_EIG])), np.inf)
print(f"  Implied timescales: {its[:10].round(3)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Assign eigenvector values to cells
# ══════════════════════════════════════════════════════════════════════════════
# Each cell inherits its cluster's eigenvector value
chi_disc = lvecs[cell_cluster, :N_EIG]    # (49116, N_EIG)

# SD of each eigenvector on all cells — collapse check
ev_sd = chi_disc.std(axis=0)
print(f"\n  Eigenvector SD (collapse check):")
for i in range(min(N_EIG, 10)):
    flag = "✓" if ev_sd[i] > 0.01 else "✗ FLAT"
    print(f"    EV {i+1:2d}: SD={ev_sd[i]:.4f}  {flag}")

# Compare ISOKANN chi SD
chi_iso_sd = chi_iso.std(axis=0)
print(f"\n  ISOKANN chi SD (first 10):")
for i in range(min(15, 10)):
    flag = "✓" if chi_iso_sd[i] > 0.01 else "✗ FLAT"
    print(f"    chi {i+1:2d}: SD={chi_iso_sd[i]:.4f}  {flag}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Tier-2: Pearson r on Neu-vs-Mono hard split
# ══════════════════════════════════════════════════════════════════════════════
print("\nTier 2 — Neu-vs-Mono hard split:")
bias_valid = nm_bias[nm_mask]

best_r_disc = 0.0; best_ev = 0
for i in range(N_EIG):
    r, _ = pearsonr(chi_disc[nm_mask, i], bias_valid)
    if abs(r) > abs(best_r_disc):
        best_r_disc, best_ev = r, i

best_r_iso = 0.0; best_iso_col = 0
for i in range(15):
    r, _ = pearsonr(chi_iso[nm_mask, i], bias_valid)
    if abs(r) > abs(best_r_iso):
        best_r_iso, best_iso_col = r, i

print(f"  Discrete T EV {best_ev+1}:  Pearson r = {best_r_disc:.3f}")
print(f"  ISOKANN chi {best_iso_col+1}: Pearson r = {best_r_iso:.3f}")
print(f"  Literature state-only range: 0.26–0.50")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Tier-1: AU-ROC per fate
# ══════════════════════════════════════════════════════════════════════════════
print("\nTier 1 — AU-ROC per fate:")
rows = []
for fate in FATES:
    if fate not in fate_labels: continue
    labels_d2 = fate_labels[fate][day2_mask].astype(int)
    if labels_d2.sum() < 5: continue

    disc_d2 = chi_disc[day2_mask]
    iso_d2  = chi_iso[day2_mask]

    auc_disc = max(best_auc(disc_d2[:, i], labels_d2) for i in range(N_EIG))
    auc_iso  = max(best_auc(iso_d2[:, i],  labels_d2) for i in range(15))
    rows.append({"fate": fate, "auc_discrete_T": auc_disc, "auc_isokann": auc_iso})
    print(f"  {fate:15s}  discrete_T={auc_disc:.3f}  ISOKANN={auc_iso:.3f}")

df_compare = pd.DataFrame(rows).set_index("fate")
df_compare.to_csv(os.path.join(BENCH, "discrete_transfer_results.csv"))


# ══════════════════════════════════════════════════════════════════════════════
# 8. Plots
# ══════════════════════════════════════════════════════════════════════════════
STATE_COLORS = {
    "undiff":"#aaaaaa","Neutrophil":"#4daf4a","Monocyte":"#f781bf",
    "Baso":"#ff7f00","Mast":"#e41a1c","Meg":"#984ea3",
    "Erythroid":"#a65628","Lymphoid":"#377eb8","Eos":"#999999",
    "Neu_Mon":"#8dd3c7","Ccr7_DC":"#ffff33","pDC":"#e0e0e0",
}

# ── 8a: Eigenvalue spectrum ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(range(1, N_EIG+1), vals[:N_EIG], "o-", lw=1.5, ms=5)
ax.axhline(1, ls="--", c="gray", lw=1)
ax.set_xlabel("Index"); ax.set_ylabel("Eigenvalue")
ax.set_title(f"Discrete T eigenvalues\n(k={N_CLUSTERS} clusters, ε={EPS})")
ax.set_xticks(range(1, N_EIG+1, 2))

ax = axes[1]
ax.bar(range(1, min(N_EIG, 10)), its[:9], color="steelblue", alpha=0.8)
ax.set_xlabel("Mode"); ax.set_ylabel("Implied timescale (× lag units)")
ax.set_title("Implied timescales")
plt.tight_layout()
fig.savefig(os.path.join(BENCH, "discrete_transfer_eigenvalues.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: discrete_transfer_eigenvalues.png")

# ── 8b: Scatter of EV2 vs EV3, coloured by cell state ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pairs_to_plot = [(1, 2), (1, 3), (2, 3)]  # 0-indexed modes (skip EV1=stationary)
for ax, (i, j) in zip(axes, pairs_to_plot):
    for state, c in STATE_COLORS.items():
        mask = state_info == state
        ax.scatter(chi_disc[mask, i], chi_disc[mask, j],
                   s=1, c=c, alpha=0.3, label=state, rasterized=True)
    ax.set_xlabel(f"EV {i+1}"); ax.set_ylabel(f"EV {j+1}")
    ax.set_title(f"EV{i+1} vs EV{j+1}")
axes[0].legend(markerscale=4, fontsize=6, ncol=2, bbox_to_anchor=(0,1), loc="lower left")
plt.suptitle(f"Discrete transfer operator — pairwise eigenvector scatter\n"
             f"(k={N_CLUSTERS} PCA clusters, no neural network)", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(BENCH, "discrete_transfer_ev_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: discrete_transfer_ev_scatter.png")

# ── 8c: Neu-vs-Mono scatter comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, scores, label, r_val in [
    (axes[0], chi_disc[nm_mask, best_ev],  f"Discrete T EV {best_ev+1}", best_r_disc),
    (axes[1], chi_iso[nm_mask, best_iso_col], f"ISOKANN chi {best_iso_col+1}", best_r_iso),
]:
    ax.scatter(scores, bias_valid, s=3, alpha=0.3, c="steelblue")
    ax.set_xlabel(label); ax.set_ylabel("Neu/(Neu+Mono) fate bias")
    ax.set_title(f"Pearson r = {r_val:.3f}")
plt.suptitle("Tier-2: Neu-vs-Mono — discrete T vs ISOKANN", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(BENCH, "discrete_transfer_neu_mono.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: discrete_transfer_neu_mono.png")

# ── 8d: AU-ROC bar chart comparison ─────────────────────────────────────────
if len(df_compare) > 0:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df_compare))
    w = 0.35
    ax.bar(x - w/2, df_compare["auc_discrete_T"], w, label="Discrete T", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, df_compare["auc_isokann"],    w, label="ISOKANN k=13", color="tomato", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df_compare.index, rotation=35, ha="right")
    ax.set_ylabel("AU-ROC"); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.axhline(0.9, ls="--", c="green", lw=1, alpha=0.5, label="0.9 sanity threshold")
    ax.set_title("AU-ROC: discrete transfer operator vs ISOKANN (day-2 fate prediction)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(BENCH, "discrete_transfer_auroc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: discrete_transfer_auroc.png")

# ── 8e: SD comparison (collapse check) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.5))
x = np.arange(1, N_EIG+1)
ax.bar(x - 0.2, ev_sd[:N_EIG], 0.4, label="Discrete T EV", color="steelblue", alpha=0.8)
ax.bar(np.arange(1, 16) + 0.2, chi_iso_sd, 0.4, label="ISOKANN chi", color="tomato", alpha=0.8)
ax.axhline(0.01, ls="--", c="gray", lw=1, label="SD=0.01 collapse threshold")
ax.set_xlabel("Mode index"); ax.set_ylabel("Standard deviation")
ax.set_title("Collapse check: SD of discrete T eigenvectors vs ISOKANN chi")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(BENCH, "discrete_transfer_sd.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: discrete_transfer_sd.png")

# ── 8f: EV2/EV3 on original UMAP ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, ev_idx, title in [
    (axes[0], 1, "Discrete T — EV 2 on original UMAP"),
    (axes[1], 2, "Discrete T — EV 3 on original UMAP"),
]:
    v = chi_disc[:, ev_idx]
    vmax = np.abs(v).max()
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=v, s=1,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(f"Discrete transfer operator eigenvectors on original UMAP\n"
             f"(k={N_CLUSTERS} PCA clusters, top eigenvalues: {vals[1:3].round(3)})", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(BENCH, "discrete_transfer_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: discrete_transfer_umap.png")

print(f"\nDiscrete transfer operator complete.")
print(f"\n=== SUMMARY ===")
print(f"Tier-2 Pearson r:  Discrete T={best_r_disc:.3f}  ISOKANN={best_r_iso:.3f}  "
      f"Literature state-only=0.26–0.50")
print(f"\nTier-1 AU-ROC:")
print(df_compare.round(3).to_string())
