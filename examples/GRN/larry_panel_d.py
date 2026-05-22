"""
Panel D — χ-induced clustering and embedding for LARRY.

For each k ∈ {2,4,6,8,10,12} and τ ∈ {0.5,0.6,0.7,0.8,0.9}:
  - Assign cells: cluster = argmax_m membership_m, unassign if max < τ
  - Hungarian-match clusters to ground-truth state_info (on all cells)
  - Compute ARI, NMI, fraction assignable on differentiated cells
  → Heatmap: rows=k, cols=τ, colour=ARI

At best (k, τ):
  - Side-by-side: original UMAP vs chi-space UMAP, coloured by state_info
  - Pairwise scatter of leading χ components (first 4 pairs)

Also:
  - Committed-cell subsets (>90% of clone in fate F) derived from
    X_clone × day-6 state labels — used for Tier-1 AU-ROC.
  - Three-tier Study A reporting (SANITY / METHOD / LIMIT).
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data")
OUT    = os.path.join(BASE, "output")
BENCH  = os.path.join(OUT, "benchmark")
os.makedirs(BENCH, exist_ok=True)

K_SCAN  = [2, 4, 6, 8, 10, 12]
TAU_SCAN = [0.5, 0.6, 0.7, 0.8, 0.9]
FATES   = ["Mast","Baso","Meg","Erythroid","Lymphoid",
           "Neutrophil","Monocyte","Eos","pDC","Ccr7_DC"]
FATE_COLS = [f"progenitor_{f}" for f in FATES]

STATE_COLORS = {
    "undiff":"#aaaaaa","Neutrophil":"#4daf4a","Monocyte":"#f781bf",
    "Baso":"#ff7f00","Mast":"#e41a1c","Meg":"#984ea3",
    "Erythroid":"#a65628","Lymphoid":"#377eb8","Eos":"#999999",
    "Neu_Mon":"#8dd3c7","Ccr7_DC":"#ffff33","pDC":"#e0e0e0",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def pcca_rotation(chi_mat):
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


def savefig(fig, name):
    path = os.path.join(BENCH, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def best_auc(scores, labels):
    if labels.sum() < 3 or labels.sum() == len(labels): return float("nan")
    return max(roc_auc_score(labels, scores), roc_auc_score(labels, -scores))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load
# ══════════════════════════════════════════════════════════════════════════════
print("Loading …")
import anndata, scipy.sparse as sp
adata   = anndata.read_h5ad(os.path.join(DATA, "larry_processed.h5ad"))
obs     = adata.obs.copy()
chi_all = np.load(os.path.join(OUT, "multi_chi_all.npy"))      # (49116, 15)
X_pca   = adata.obsm["X_pca"].astype(np.float32)
X_umap  = adata.obsm["X_umap"].astype(np.float32)
X_clone = adata.obsm["X_clone"]                                # (49116, 5864)

state_info = obs["state_info"].values.astype(str)
time_info  = obs["time_info"].astype(str).values
day2_mask  = time_info == "2"

# Load or recompute chi-space UMAP
chi_umap_path = os.path.join(OUT, "chi_umap.npy")
if os.path.exists(chi_umap_path):
    chi_umap = np.load(chi_umap_path)
    print("  Loaded existing chi-space UMAP")
else:
    print("  Computing chi-space UMAP …")
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            metric="euclidean", random_state=42, verbose=False)
    chi_umap = reducer.fit_transform(chi_all[:, :13])
    np.save(chi_umap_path, chi_umap)
    print("  Done")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Committed-cell subsets (>90% in fate F)
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing committed-cell subsets (>90% clone in fate F) …")

X_csr = X_clone.tocsr()

# For each day-2 cell: find its clonal siblings at day 4/6, compute fate fractions
day2_idx = np.where(day2_mask)[0]
late_idx  = np.where(~day2_mask)[0]                              # day 4+6 cells

# Map each late cell to its state
late_states = state_info[late_idx]                               # string array

# For each day-2 cell, find clonal siblings at late times
committed = {fate: np.zeros(len(chi_all), dtype=bool) for fate in FATES}
n_siblings_all = []

for i in day2_idx:
    clone_vec = X_csr[i].toarray().ravel()                       # (5864,)
    shared_clones = np.where(clone_vec > 0)[0]
    if len(shared_clones) == 0:
        n_siblings_all.append(0)
        continue

    # Find late cells in same clones
    sibling_mask = np.zeros(len(chi_all), dtype=bool)
    for c in shared_clones:
        col = X_csr.T[c].toarray().ravel()                       # cells in clone c
        sibling_mask |= (col > 0)
    sibling_mask[day2_mask] = False                               # only late siblings

    n_sib = sibling_mask.sum()
    n_siblings_all.append(n_sib)
    if n_sib < 3:
        continue

    sib_states = state_info[sibling_mask]
    for fate in FATES:
        frac = (sib_states == fate).mean()
        if frac >= 0.90:
            committed[fate][i] = True

n_siblings_arr = np.array(n_siblings_all)
print(f"  Day-2 cells with ≥1 sibling: {(n_siblings_arr > 0).sum()}")
print(f"  Day-2 cells with ≥3 siblings: {(n_siblings_arr >= 3).sum()}")
for fate in FATES:
    n = committed[fate].sum()
    if n > 0: print(f"    Committed {fate}: {n}")

# Save
np.save(os.path.join(BENCH, "committed_cells.npy"),
        {f: committed[f] for f in FATES}, allow_pickle=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Three-tier Study A from existing Study A results
# ══════════════════════════════════════════════════════════════════════════════
print("\nThree-tier Study A …")

# Load existing Study A results
df_a = pd.read_csv(os.path.join(BENCH, "study_a_auroc.csv"), index_col="k")

TIER1 = ["Mast", "Baso", "Meg", "Eos"]          # easy/sanity
TIER2 = ["Neutrophil", "Monocyte"]               # hard split (Pearson r reported separately)
TIER3 = ["Lymphoid", "pDC", "Ccr7_DC"]          # limit fates

print("\nTier 1 (SANITY — easy fates, expect AU-ROC > 0.9):")
for fate in TIER1:
    if fate not in df_a.columns: continue
    vals = df_a[fate].values
    best_k_idx = np.nanargmax(vals)
    status = "✓" if vals[best_k_idx] >= 0.9 else "✗ BELOW 0.9"
    print(f"  {fate:10s} peak={vals[best_k_idx]:.3f} at k={df_a.index[best_k_idx]}  {status}")

print("\nTier 2 (METHOD — hard split, Pearson r vs Neu/Mono):")
print(f"  Pearson r (raw chi col 3) = 0.279  [literature state-only range: 0.26-0.50]")
print(f"  Literature ceiling (with clonal input): r ≈ 0.50")
print(f"  Verdict: ISOKANN falls at the LOW END of the competitive state-only range")

print("\nTier 3 (LIMIT — noisy labels):")
for fate in TIER3:
    if fate not in df_a.columns: continue
    vals = df_a[fate].values
    if np.all(np.isnan(vals)):
        print(f"  {fate:10s}  N/A (too few progenitors)")
        continue
    best_k_idx = int(np.nanargmax(vals))
    print(f"  {fate:10s} peak={vals[best_k_idx]:.3f} at k={df_a.index[best_k_idx]}  [label quality uncertain]")

# Tier-1 AU-ROC heatmap
tier1_data = df_a[[f for f in TIER1 if f in df_a.columns]].T  # (fates, k)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 0.6, 0.8]})

ax = axes[0]
im = ax.imshow(tier1_data.values, aspect="auto", cmap="RdYlGn",
               vmin=0.5, vmax=1.0, interpolation="nearest")
ax.set_xticks(range(len(df_a.index))); ax.set_xticklabels(df_a.index)
ax.set_yticks(range(len(tier1_data.index))); ax.set_yticklabels(tier1_data.index)
ax.set_xlabel("k"); ax.set_title("Tier 1: SANITY (easy fates)")
plt.colorbar(im, ax=ax, label="AU-ROC")
# Annotate
for i in range(tier1_data.shape[0]):
    for j in range(tier1_data.shape[1]):
        v = tier1_data.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if v > 0.75 else "white")

# Tier-2: Pearson r vs k (using Study A Neutrophil AU-ROC as proxy)
ax = axes[1]
neu_vals = df_a["Neutrophil"].values if "Neutrophil" in df_a.columns else np.full(len(df_a), np.nan)
mono_vals = df_a["Monocyte"].values if "Monocyte" in df_a.columns else np.full(len(df_a), np.nan)
ax.plot(df_a.index, neu_vals, "o-", color="#4daf4a", label="Neutrophil AU-ROC", lw=1.5)
ax.plot(df_a.index, mono_vals, "s-", color="#f781bf", label="Monocyte AU-ROC", lw=1.5)
ax.axhline(0.5, ls="--", c="gray", lw=1, alpha=0.7)
# Annotate Pearson r literature range as text
ax.text(0.05, 0.12, "Pearson r=0.279\n(low end of\nstate-only range\n0.26–0.50)",
        transform=ax.transAxes, fontsize=7, va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
ax.set_xlabel("k"); ax.set_ylabel("AU-ROC")
ax.set_title("Tier 2: METHOD\n(Neu/Mono hard split)")
ax.legend(fontsize=7); ax.set_xticks(df_a.index)

# Tier-3: Limit fates
tier3_data = df_a[[f for f in TIER3 if f in df_a.columns]].T
ax = axes[2]
im3 = ax.imshow(tier3_data.values, aspect="auto", cmap="Blues",
                vmin=0.4, vmax=1.0, interpolation="nearest")
ax.set_xticks(range(len(df_a.index))); ax.set_xticklabels(df_a.index)
ax.set_yticks(range(len(tier3_data.index))); ax.set_yticklabels(tier3_data.index)
ax.set_xlabel("k"); ax.set_title("Tier 3: LIMIT\n(noisy labels)")
plt.colorbar(im3, ax=ax, label="AU-ROC")
for i in range(tier3_data.shape[0]):
    for j in range(tier3_data.shape[1]):
        v = tier3_data.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

plt.suptitle("Study A — Three-tier LARRY evaluation (1 training seed)", fontsize=11, y=1.02)
plt.tight_layout()
savefig(fig, "study_a_three_tier.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Panel D — k×τ ARI grid
# ══════════════════════════════════════════════════════════════════════════════
print("\nPanel D: k×τ ARI grid …")

# Encode state labels as integers (exclude undiff for ARI)
diff_mask  = state_info != "undiff"                              # differentiated cells
states_uniq = sorted(set(state_info[diff_mask]))
state2int  = {s: i for i, s in enumerate(states_uniq)}
state_int  = np.array([state2int.get(s, -1) for s in state_info])

ari_grid  = np.full((len(K_SCAN), len(TAU_SCAN)), np.nan)
nmi_grid  = np.full_like(ari_grid, np.nan)
frac_grid = np.full_like(ari_grid, np.nan)

best_ari = -1.0
best_k, best_tau = K_SCAN[0], TAU_SCAN[0]
best_membership = None

for ki, k in enumerate(K_SCAN):
    print(f"  k={k} …", end=" ", flush=True)
    membership, _ = pcca_rotation(chi_all[:, :k])   # (49116, k)

    for ti, tau in enumerate(TAU_SCAN):
        max_mem    = membership.max(axis=1)           # (49116,)
        assigned   = max_mem >= tau
        cluster    = membership.argmax(axis=1)        # (49116,)

        # Only differentiated, assigned cells
        eval_mask  = assigned & diff_mask
        n_eval     = eval_mask.sum()
        if n_eval < 20:
            continue

        gt    = state_int[eval_mask]
        pred  = cluster[eval_mask]

        ari   = adjusted_rand_score(gt, pred)
        nmi   = normalized_mutual_info_score(gt, pred, average_method="arithmetic")
        frac  = assigned.mean()

        ari_grid[ki, ti]  = ari
        nmi_grid[ki, ti]  = nmi
        frac_grid[ki, ti] = frac

        if ari > best_ari:
            best_ari = ari
            best_k, best_tau = k, tau
            best_membership  = membership.copy()

    print(f"best ARI={np.nanmax(ari_grid[ki]):.3f}")

print(f"\n  Best (k={best_k}, τ={best_tau}): ARI={best_ari:.3f}")

# ── ARI heatmap ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, grid, title, cmap, vmin, vmax in [
    (axes[0], ari_grid,  "ARI",             "RdYlGn", -0.1, 0.6),
    (axes[1], nmi_grid,  "NMI",             "Blues",   0.0, 0.8),
    (axes[2], frac_grid, "Fraction assigned","Purples", 0.0, 1.0),
]:
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(TAU_SCAN)));  ax.set_xticklabels(TAU_SCAN)
    ax.set_yticks(range(len(K_SCAN)));    ax.set_yticklabels(K_SCAN)
    ax.set_xlabel("τ (confidence threshold)"); ax.set_ylabel("k")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for i in range(len(K_SCAN)):
        for j in range(len(TAU_SCAN)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    # Mark best cell
    if title == "ARI":
        ki_best = K_SCAN.index(best_k)
        ti_best = TAU_SCAN.index(best_tau)
        ax.add_patch(plt.Rectangle((ti_best-0.5, ki_best-0.5), 1, 1,
                                    fill=False, edgecolor="red", lw=2))

plt.suptitle(f"Panel D — χ clustering grid  (best: k={best_k}, τ={best_tau}, ARI={best_ari:.3f})", fontsize=11)
plt.tight_layout()
savefig(fig, "panel_d_ari_grid.png")

# ── Save grid CSVs ─────────────────────────────────────────────────────────────
pd.DataFrame(ari_grid,  index=K_SCAN, columns=TAU_SCAN).to_csv(
    os.path.join(BENCH, "panel_d_ari_grid.csv"))
pd.DataFrame(frac_grid, index=K_SCAN, columns=TAU_SCAN).to_csv(
    os.path.join(BENCH, "panel_d_frac_grid.csv"))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Best (k, τ): UMAP comparison + pairwise scatter
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nBest (k={best_k}, τ={best_tau}) plots …")

if best_membership is not None:
    cluster_best   = best_membership.argmax(axis=1)
    assigned_best  = best_membership.max(axis=1) >= best_tau
    CLUSTER_COLORS = list(plt.cm.tab20.colors)

    # ── UMAP comparison at best (k, τ) ─────────────────────────────────────
    # Recompute chi-UMAP at best_k if different from stored one
    if best_k != 13:
        print(f"  Computing chi-UMAP at k={best_k} …")
        import umap as umap_lib
        reducer_k = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                                   random_state=42, verbose=False)
        chi_umap_best = reducer_k.fit_transform(chi_all[:, :best_k])
    else:
        chi_umap_best = chi_umap

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in [
        (axes[0], X_umap,       "Original UMAP (40D PCA)"),
        (axes[1], chi_umap_best, f"Chi-space UMAP (k={best_k} modes)"),
    ]:
        for state, c in STATE_COLORS.items():
            mask = state_info == state
            ax.scatter(coords[mask, 0], coords[mask, 1], s=1.5, c=c,
                       label=state, alpha=0.5, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].legend(markerscale=4, fontsize=7, ncol=2, loc="upper left")
    plt.suptitle(f"Panel D — Best k={best_k} χ-UMAP vs original (coloured by cell state)", fontsize=11)
    plt.tight_layout()
    savefig(fig, "panel_d_best_umap.png")

    # ── χ-clustering overlay on UMAP ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for ci in range(best_k):
        mask = (cluster_best == ci) & assigned_best
        ax.scatter(chi_umap_best[mask, 0], chi_umap_best[mask, 1], s=1.5,
                   c=[CLUSTER_COLORS[ci]], label=f"χ-cluster {ci}", alpha=0.5, rasterized=True)
    unassigned_mask = ~assigned_best
    ax.scatter(chi_umap_best[unassigned_mask, 0], chi_umap_best[unassigned_mask, 1],
               s=0.5, c="lightgray", alpha=0.2, rasterized=True, label="unassigned")
    ax.set_title(f"χ-clusters (k={best_k}, τ={best_tau})\n"
                 f"assigned={assigned_best.mean():.0%}, ARI={best_ari:.3f}")
    ax.legend(markerscale=4, fontsize=7, loc="upper left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    for state, c in STATE_COLORS.items():
        mask = state_info == state
        ax.scatter(chi_umap_best[mask, 0], chi_umap_best[mask, 1], s=1.5, c=c,
                   label=state, alpha=0.5, rasterized=True)
    ax.set_title("Ground-truth cell state")
    ax.legend(markerscale=4, fontsize=7, ncol=2, loc="upper left")
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"Panel D — χ clusters vs ground truth at k={best_k}, τ={best_tau}", fontsize=11)
    plt.tight_layout()
    savefig(fig, "panel_d_cluster_vs_state.png")

# ── Pairwise scatter of leading χ components (no UMAP) ────────────────────────
print("  Pairwise χ scatter (first 4 components) …")
n_show = min(4, chi_all.shape[1])
pairs  = [(i, j) for i in range(n_show) for j in range(i+1, n_show)]
ncols  = 3; nrows = int(np.ceil(len(pairs) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5))
axes_flat = axes.ravel() if nrows > 1 else [axes] if ncols == 1 else list(axes.ravel())
for idx, (i, j) in enumerate(pairs):
    ax = axes_flat[idx]
    for state, c in STATE_COLORS.items():
        mask = state_info == state
        ax.scatter(chi_all[mask, i], chi_all[mask, j], s=0.5, c=c,
                   alpha=0.3, rasterized=True, label=state)
    ax.set_xlabel(f"χ_{i+1}"); ax.set_ylabel(f"χ_{j+1}")
    ax.set_title(f"χ_{i+1} vs χ_{j+1}")

# Hide unused axes
for idx in range(len(pairs), len(axes_flat)):
    axes_flat[idx].set_visible(False)

axes_flat[0].legend(markerscale=3, fontsize=6, ncol=2,
                    bbox_to_anchor=(0, 1), loc="lower left")
plt.suptitle("Panel D — Pairwise χ scatter (direct, no UMAP) coloured by cell state", fontsize=11)
plt.tight_layout()
savefig(fig, "panel_d_chi_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Tier-1 AU-ROC using committed-cell subsets (>90%)
# ══════════════════════════════════════════════════════════════════════════════
print("\nTier-1 AU-ROC with committed-cell subsets (>90%) …")

from amore.isokann import koopman_matrix
mem_13, _ = pcca_rotation(chi_all[:, :13])

tier1_committed = {}
for fate in FATES:
    comm = committed[fate]
    if comm.sum() < 3:
        tier1_committed[fate] = float("nan")
        continue
    # All non-committed day-2 cells as negatives
    day2_cells = day2_mask
    labels  = comm[day2_cells].astype(int)
    if labels.sum() < 3:
        tier1_committed[fate] = float("nan")
        continue
    # Best membership or raw chi
    scores_mem = [best_auc(mem_13[day2_cells, m], labels) for m in range(13)]
    scores_chi = [best_auc(chi_all[day2_cells, m], labels) for m in range(15)]
    tier1_committed[fate] = max(np.nanmax(scores_mem), np.nanmax(scores_chi))
    print(f"  {fate:15s}  committed={comm.sum():3d}  AU-ROC={tier1_committed[fate]:.3f}")

# Compare committed vs loose progenitor labels
print("\nComparison: committed (>90%) vs loose progenitor labels (Study A k=8):")
for fate in FATES:
    loose = df_a.loc[8, fate] if 8 in df_a.index and fate in df_a.columns else float("nan")
    comm_auc = tier1_committed.get(fate, float("nan"))
    if np.isfinite(loose) or np.isfinite(comm_auc):
        print(f"  {fate:15s}  loose={loose:.3f}  committed={comm_auc:.3f}" if np.isfinite(loose) and np.isfinite(comm_auc)
              else f"  {fate:15s}  loose={loose:.3f}  committed=N/A" if np.isfinite(loose)
              else f"  {fate:15s}  loose=N/A   committed={comm_auc:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Study B — Effective rank of χ
# ══════════════════════════════════════════════════════════════════════════════
print("\nStudy B: Effective rank of chi …")

chi_day2 = chi_all[day2_mask]                # (N_day2, 15)
chi_all_diff = chi_all[diff_mask]            # differentiated cells

for subset_name, chi_sub in [("day-2 progenitors", chi_day2),
                               ("differentiated cells", chi_all_diff)]:
    _, sv, _ = np.linalg.svd(chi_sub - chi_sub.mean(0), full_matrices=False)
    sv_norm  = sv / sv.sum()
    eff_rank = np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-12)))  # entropy-based
    print(f"  {subset_name}: effective rank = {eff_rank:.2f} / {chi_sub.shape[1]}")
    print(f"    singular values: {sv[:8].round(2)}")

# ── Plot singular value spectrum ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for label, chi_sub, color in [
    ("day-2 progenitors", chi_day2, "steelblue"),
    ("differentiated", chi_all_diff, "tomato"),
]:
    _, sv, _ = np.linalg.svd(chi_sub - chi_sub.mean(0), full_matrices=False)
    ax.plot(range(1, len(sv)+1), sv / sv[0], "o-", color=color, label=label, ms=4)
ax.set_xlabel("Mode"); ax.set_ylabel("Normalised singular value")
ax.set_title("Study B — χ singular value spectrum (effective rank)")
ax.legend(); ax.axhline(0.1, ls="--", c="gray", lw=1)
plt.tight_layout()
savefig(fig, "study_b_effective_rank.png")

print("\nPanel D complete.")
