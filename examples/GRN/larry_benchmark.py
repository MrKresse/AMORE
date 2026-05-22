"""
LARRY benchmark: Study A (dimension scan), Panel D (chi-space UMAP),
kNN baseline, Neu-vs-Mono hard split.

Uses existing multi_chi_all.npy — no retraining required for these studies.

Outputs
-------
  output/benchmark/study_a_auroc.png         — AUC-ROC vs k for each fate
  output/benchmark/study_a_auroc.csv         — numerical table
  output/benchmark/panel_d_umap.png          — chi-space UMAP coloured by state/cluster
  output/benchmark/panel_d_comparison.png    — original vs chi-space UMAP side-by-side
  output/benchmark/knn_comparison.png        — kNN vs ISOKANN AUC-ROC
  output/benchmark/neu_mono_split.png        — Neu/Mono hard-split scatter
  output/benchmark/LARRY_BENCHMARK_RESULTS.md
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data")
OUT    = os.path.join(BASE, "output")
BENCH  = os.path.join(OUT, "benchmark")
os.makedirs(BENCH, exist_ok=True)

# ── Fate columns in adata.obs (progenitor labels for day-2 cells) ─────────────
FATES = ["Mast", "Baso", "Meg", "Erythroid", "Lymphoid",
         "Neutrophil", "Monocyte", "Eos", "pDC", "Ccr7_DC"]
FATE_COLS = [f"progenitor_{f}" for f in FATES]
K_SCAN = [2, 3, 4, 5, 6, 8, 10, 12, 13]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def pcca_rotation(chi_mat: np.ndarray):
    """Simplified PCCA+ via successive max-distance vertex selection."""
    n, k = chi_mat.shape
    chi_n = chi_mat - chi_mat.min(0)
    chi_n = chi_n / (chi_n.sum(1, keepdims=True) + 1e-8)

    vertex_idx = [int(np.argmax(np.linalg.norm(chi_n - chi_n.mean(0), axis=1)))]
    for _ in range(k - 1):
        dists = np.min(np.stack([np.linalg.norm(chi_n - chi_n[v], axis=1)
                                  for v in vertex_idx]), axis=0)
        vertex_idx.append(int(np.argmax(dists)))

    vertex_idx = np.array(vertex_idx)
    C = chi_n[vertex_idx]
    try:
        A = np.linalg.inv(C)
        membership = chi_n @ A
        membership = np.clip(membership, 0, None)
        membership = membership / (membership.sum(1, keepdims=True) + 1e-8)
    except np.linalg.LinAlgError:
        membership = chi_n
    return membership, vertex_idx


def best_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC, trying both sign conventions; 0.0 if only one class."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    a = roc_auc_score(labels, scores)
    b = roc_auc_score(labels, -scores)
    return max(a, b)


def hungarian_auc(membership: np.ndarray, label_mat: np.ndarray):
    """
    Assign membership columns to fate columns maximising total AUC.
    Returns per-fate best AUC and the assignment array.
    membership : (N, k)
    label_mat  : (N, F) binary
    """
    from scipy.optimize import linear_sum_assignment
    k, F = membership.shape[1], label_mat.shape[1]
    n = min(k, F)
    auc_mat = np.full((k, F), np.nan)
    for i in range(k):
        for j in range(F):
            auc_mat[i, j] = best_auc(membership[:, i], label_mat[:, j])
    finite = np.nan_to_num(auc_mat, nan=0.0)
    row_idx, col_idx = linear_sum_assignment(-finite[:n, :n])
    assignment = dict(zip(col_idx, row_idx))   # fate_idx -> membership_idx
    return auc_mat, assignment


def savefig(fig, name):
    path = os.path.join(BENCH, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
import anndata
adata = anndata.read_h5ad(os.path.join(DATA, "larry_processed.h5ad"))
chi_all = np.load(os.path.join(OUT, "multi_chi_all.npy"))    # (49116, 15)
ev      = np.load(os.path.join(OUT, "multi_eigenvalues.npy"))
X_pca   = adata.obsm["X_pca"].astype(np.float32)             # (49116, 40)
X_umap  = adata.obsm["X_umap"].astype(np.float32)            # (49116, 2)
obs     = adata.obs.copy()

N_CELLS = chi_all.shape[0]
print(f"  {N_CELLS} cells, {chi_all.shape[1]} eigenfunctions")
print(f"  Eigenvalues: {ev.round(3)}")

# ── Day-2 progenitor mask ─────────────────────────────────────────────────────
# time_info may be int or string depending on h5ad serialisation
time_vals_raw = obs["time_info"].values
print(f"  time_info dtype={obs['time_info'].dtype}  unique={sorted(set(str(v) for v in time_vals_raw[:20]))}")
day2_mask = obs["time_info"].astype(str).values == "2"            # (49116,)
day2_idx  = np.where(day2_mask)[0]
print(f"  Day-2 cells: {day2_mask.sum()}")

# Fate labels for day-2 cells
fate_labels = {}
for fate, col in zip(FATES, FATE_COLS):
    if col in obs.columns:
        vals = obs[col].values.astype(float)
        fate_labels[fate] = vals   # defined for all cells, non-zero only for day-2

# Count positives
for fate in FATES:
    n_pos = (fate_labels[fate][day2_mask] > 0).sum()
    print(f"    {fate}: {n_pos} day-2 progenitors")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Study A — PCCA+(k) dimension scan
# ══════════════════════════════════════════════════════════════════════════════
print("\nStudy A: PCCA+ dimension scan …")

results_a = []   # list of dicts

for k in K_SCAN:
    print(f"  k={k} …", end=" ", flush=True)
    chi_k = chi_all[:, :k]
    membership, _ = pcca_rotation(chi_k)     # (49116, k)

    # Restrict to day-2 cells with at least one fate label
    mem_d2 = membership[day2_mask]           # (N_day2, k)

    row = {"k": k}
    for fate in FATES:
        labels_d2 = fate_labels[fate][day2_mask].astype(int)
        if labels_d2.sum() < 5:
            row[fate] = float("nan")
            continue
        # Best AUC over all membership columns (greedy, not Hungarian, for speed)
        aucs = [best_auc(mem_d2[:, i], labels_d2) for i in range(k)]
        row[fate] = float(np.nanmax(aucs))
    results_a.append(row)
    fates_with_data = [f for f in FATES if not np.isnan(row.get(f, np.nan))]
    mean_auc = np.nanmean([row[f] for f in fates_with_data])
    print(f"mean AUC={mean_auc:.3f}")

df_a = pd.DataFrame(results_a).set_index("k")
df_a.to_csv(os.path.join(BENCH, "study_a_auroc.csv"))
print(df_a.round(3).to_string())

# ── Plot Study A ──────────────────────────────────────────────────────────────
FATE_COLORS = {
    "Mast": "#e41a1c", "Baso": "#ff7f00", "Meg": "#984ea3",
    "Erythroid": "#a65628", "Lymphoid": "#377eb8", "Neutrophil": "#4daf4a",
    "Monocyte": "#f781bf", "Eos": "#999999", "pDC": "#ffff33", "Ccr7_DC": "#8dd3c7",
}
fig, ax = plt.subplots(figsize=(8, 5))
for fate in FATES:
    vals = df_a[fate].values
    if np.all(np.isnan(vals)):
        continue
    ax.plot(K_SCAN, vals, marker="o", ms=5,
            color=FATE_COLORS.get(fate, "gray"), label=fate, lw=1.5)
ax.axvline(4,  ls="--", c="gray", lw=1, alpha=0.7, label="spectral gap (k=4)")
ax.axvline(13, ls=":",  c="gray", lw=1, alpha=0.7, label="user k=13")
ax.axhline(0.7, ls="--", c="black", lw=0.8, alpha=0.5)
ax.set_xlabel("k (number of Koopman modes)")
ax.set_ylabel("Best AU-ROC (max over membership columns)")
ax.set_title("Study A — Fate prediction vs ISOKANN dimension k")
ax.legend(fontsize=7, ncol=2, loc="lower right")
ax.set_xticks(K_SCAN)
ax.set_ylim(0.4, 1.02)
plt.tight_layout()
savefig(fig, "study_a_auroc.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. kNN baseline (no dynamics)
# ══════════════════════════════════════════════════════════════════════════════
print("\nkNN baseline …")

# Use day-2 cells with at least one fate label
X_d2 = X_pca[day2_mask]   # (N_day2, 40)

knn_results = {}
iso_results = {}   # ISOKANN at k=13 for same fates

# PCCA+ at k=13 once
print("  Computing PCCA+ at k=13 for ISOKANN comparison …")
mem_13, _ = pcca_rotation(chi_all[:, :13])
mem_13_d2 = mem_13[day2_mask]

for fate in FATES:
    labels_d2 = fate_labels[fate][day2_mask].astype(int)
    if labels_d2.sum() < 10:
        knn_results[fate] = float("nan")
        iso_results[fate] = float("nan")
        continue

    # kNN: 5-fold cross-validated AUC
    knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for tr, te in cv.split(X_d2, labels_d2):
        knn.fit(X_d2[tr], labels_d2[tr])
        proba = knn.predict_proba(X_d2[te])
        if proba.shape[1] < 2:
            continue
        fold_aucs.append(roc_auc_score(labels_d2[te], proba[:, 1]))
    knn_results[fate] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    # ISOKANN k=13: best membership column
    iso_aucs = [best_auc(mem_13_d2[:, i], labels_d2) for i in range(13)]
    iso_results[fate] = float(np.nanmax(iso_aucs))

    print(f"  {fate}: kNN={knn_results[fate]:.3f}  ISOKANN={iso_results[fate]:.3f}")

# ── Plot kNN vs ISOKANN ────────────────────────────────────────────────────────
knn_vals  = [knn_results[f] for f in FATES]
iso_vals  = [iso_results[f]  for f in FATES]
colors    = [FATE_COLORS.get(f, "gray") for f in FATES]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(FATES))
w = 0.35
bars1 = ax.bar(x - w/2, knn_vals, w, label="kNN-PCA (baseline)", color="steelblue", alpha=0.8)
bars2 = ax.bar(x + w/2, iso_vals,  w, label="ISOKANN k=13",       color="tomato",    alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(FATES, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("AU-ROC"); ax.set_ylim(0, 1.05)
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_title("kNN-PCA baseline vs ISOKANN k=13 (day-2 fate prediction)")
ax.legend()
plt.tight_layout()
savefig(fig, "knn_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Neu-vs-Mono hard split
# ══════════════════════════════════════════════════════════════════════════════
print("\nNeu-vs-Mono hard split …")

nm_bias = obs["NeuMon_fate_bias"].values.astype(float)
nm_mask = obs["NeuMon_mask"].values.astype(bool)

# Restrict to day-2 cells with a defined Neu/Mono outcome
valid = day2_mask & nm_mask
print(f"  Valid Neu/Mono day-2 cells: {valid.sum()}")

if valid.sum() > 10:
    bias_valid   = nm_bias[valid]            # continuous [0,1]: 1=pure Neu, 0=pure Mono
    mem_valid    = mem_13[valid]             # (N, 13) membership at k=13
    chi_valid    = chi_all[valid]            # (N, 15) raw eigenfunctions

    # Best Pearson r over all membership columns
    best_r, best_p, best_col = 0.0, 1.0, 0
    for i in range(13):
        r, p = pearsonr(mem_valid[:, i], bias_valid)
        if abs(r) > abs(best_r):
            best_r, best_p, best_col = r, p, i

    print(f"  Best membership col={best_col}, Pearson r={best_r:.3f}, p={best_p:.2e}")

    # Also try raw chi
    best_r_chi, best_col_chi = 0.0, 0
    for i in range(15):
        r, _ = pearsonr(chi_valid[:, i], bias_valid)
        if abs(r) > abs(best_r_chi):
            best_r_chi, best_col_chi = r, i
    print(f"  Best raw chi col={best_col_chi}, Pearson r={best_r_chi:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, scores, label, r_val in [
        (axes[0], mem_valid[:, best_col],  f"Membership col {best_col}", best_r),
        (axes[1], chi_valid[:, best_col_chi], f"Chi col {best_col_chi}", best_r_chi),
    ]:
        ax.scatter(scores, bias_valid, s=4, alpha=0.3, c="steelblue")
        ax.set_xlabel(label)
        ax.set_ylabel("Neu/(Neu+Mono) fate bias")
        ax.set_title(f"Pearson r = {r_val:.3f}")
    plt.suptitle("Neu-vs-Mono hard split (day-2 progenitors with Neu/Mono fate)", y=1.02)
    plt.tight_layout()
    savefig(fig, "neu_mono_split.png")
else:
    print("  Not enough cells for Neu-vs-Mono analysis.")
    best_r, best_p, best_col = float("nan"), float("nan"), -1


# ══════════════════════════════════════════════════════════════════════════════
# 5. Panel D — UMAP in chi-space
# ══════════════════════════════════════════════════════════════════════════════
print("\nPanel D: UMAP in chi-space …")
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        metric="euclidean", random_state=42, verbose=False)
    chi_umap = reducer.fit_transform(chi_all[:, :13])
    np.save(os.path.join(OUT, "chi_umap.npy"), chi_umap)
    print("  UMAP computed.")
except ImportError:
    print("  umap-learn not available — using PCA of chi as fallback")
    from sklearn.decomposition import PCA
    chi_umap = PCA(n_components=2).fit_transform(chi_all[:, :13])

# Cluster label: argmax of membership (at k=13)
cluster_label = np.argmax(mem_13, axis=1)

# Color palette for cell states
STATE_COLORS = {
    "undiff": "#aaaaaa", "Neutrophil": "#4daf4a", "Monocyte": "#f781bf",
    "Baso": "#ff7f00", "Mast": "#e41a1c", "Meg": "#984ea3",
    "Erythroid": "#a65628", "Lymphoid": "#377eb8", "Eos": "#999999",
    "Neu_Mon": "#8dd3c7", "Ccr7_DC": "#ffff33", "pDC": "#e0e0e0",
}
state_vals = obs["state_info"].values
time_vals  = obs["time_info"].values
CMAP_TIME  = {2: "#d73027", 4: "#fc8d59", 6: "#4575b4"}

# ── 5a: Side-by-side original vs chi-space UMAP ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, coords, title in [
    (axes[0], X_umap,  "Original UMAP (PCA features)"),
    (axes[1], chi_umap, "Chi-space UMAP (13 Koopman modes)"),
]:
    for state, c in STATE_COLORS.items():
        mask = state_vals == state
        ax.scatter(coords[mask, 0], coords[mask, 1], s=1, c=c,
                   label=state, alpha=0.5, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

axes[0].legend(markerscale=4, fontsize=7, ncol=2,
               loc="upper left", framealpha=0.7)
plt.suptitle("Panel D — Original vs Chi-space UMAP coloured by cell state", fontsize=12)
plt.tight_layout()
savefig(fig, "panel_d_comparison.png")

# ── 5b: Chi-space UMAP with PCCA+ cluster + time ─────────────────────────────
CLUSTER_COLORS = plt.cm.tab20.colors
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# By PCCA+ cluster
ax = axes[0]
for ci in range(13):
    mask = cluster_label == ci
    ax.scatter(chi_umap[mask, 0], chi_umap[mask, 1], s=1, alpha=0.4,
               c=[CLUSTER_COLORS[ci % 20]], label=f"m{ci}", rasterized=True)
ax.set_title("PCCA+ cluster (k=13, argmax membership)")
ax.legend(markerscale=4, fontsize=7, ncol=2, loc="upper left", framealpha=0.7)
ax.set_xticks([]); ax.set_yticks([])

# By time
ax = axes[1]
for t, c in CMAP_TIME.items():
    mask = time_vals == t
    ax.scatter(chi_umap[mask, 0], chi_umap[mask, 1], s=1, alpha=0.4,
               c=c, label=f"Day {t}", rasterized=True)
ax.set_title("Time point")
ax.legend(markerscale=4, fontsize=8, loc="upper left", framealpha=0.7)
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Panel D — Chi-space UMAP: PCCA+ clusters & time", fontsize=12)
plt.tight_layout()
savefig(fig, "panel_d_clusters.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Summarise Study A as table
# ══════════════════════════════════════════════════════════════════════════════

# Find best k per fate
print("\nSummary: best k per fate")
for fate in FATES:
    vals = df_a[fate].values
    if np.all(np.isnan(vals)):
        continue
    best_k_idx = np.nanargmax(vals)
    print(f"  {fate:15s}  best k={K_SCAN[best_k_idx]}  AUC={vals[best_k_idx]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Save numerical summary
# ══════════════════════════════════════════════════════════════════════════════

summary = {
    "study_a": df_a.to_dict(),
    "knn": knn_results,
    "isokann_k13": iso_results,
    "neu_mono_r": float(best_r) if "best_r" in dir() else float("nan"),
    "neu_mono_p": float(best_p) if "best_p" in dir() else float("nan"),
    "neu_mono_col": int(best_col) if "best_col" in dir() else -1,
}
np.save(os.path.join(BENCH, "benchmark_summary.npy"), summary, allow_pickle=True)

print("\nBenchmark complete. Figures saved to output/benchmark/")
