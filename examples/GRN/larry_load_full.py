"""
Load the full LARRY 130k-cell dataset via cospar and extract Koopman pairs
with two key improvements over larry_load.py:

1. Balanced pair sampling:  each transition type (undiff->Erythroid, undiff->Baso, ...)
   is capped at MAX_PAIRS_PER_TYPE to prevent NeuMon pairs from dominating training.

2. Clone holdout (80/20 split):  clones are split before pair extraction so the
   held-out 20% of clones never appear as training pairs.  This allows rigorous
   out-of-sample AUC evaluation on day-2 cells from held-out clones, ruling out
   any concern that day-2 cells in training pairs inflate the fate-prediction AUC.

Dataset
-------
cospar.datasets.hematopoiesis_130K() — the full LARRY experiment:
  ~130k cells x 25k genes  |  time points: 2, 4, 6  |  5864+ clones

Outputs (data/)
--------------
  larry_full_processed.h5ad  — full AnnData
  larry_full_pca.npy          — PCA, (n_cells, 40)
  larry_full_x0_train.npy     — train pairs x0
  larry_full_x1_train.npy     — train pairs x1
  larry_full_x0_test.npy      — held-out clone pairs (for AUC validation)
  larry_full_x1_test.npy
  larry_full_train_clones.npy — which clone indices are in train set
  larry_full_test_clones.npy  — which clone indices are in test set
  larry_full_obs.csv          — cell metadata
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_PCS              = 40
T_PAIRS            = [("2", "4"), ("2", "6"), ("4", "6")]  # time-lag pairs
TRAIN_FRACTION     = 0.80    # fraction of clones for training
MAX_PAIRS_PER_TYPE = 5_000   # cap per (src_state, dst_state) combination
SEED               = 42

rng = np.random.default_rng(SEED)

# ── 1. Load ─────────────────────────────────────────────────────────────────────
print("Loading LARRY 130k dataset via cospar ...")
try:
    import cospar as cs
    adata = cs.datasets.hematopoiesis_130K()
    print(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
except Exception as e:
    raise RuntimeError(
        f"Failed to load via cospar: {e}\n"
        "Make sure cospar>=0.5 is installed: pip install cospar"
    )

# Print metadata columns for inspection
print("\nobs columns:")
for col in adata.obs.columns:
    vals = adata.obs[col].dropna().unique()
    print(f"  {col:<30s}  {sorted(str(v) for v in vals[:5])}")

print("\nobsm keys:")
for k in adata.obsm:
    print(f"  {k:<20s}  {adata.obsm[k].shape}")


# ── 2. Preprocessing ────────────────────────────────────────────────────────────
print("\nPreprocessing ...")

if "X_pca" not in adata.obsm:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger")
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=N_PCS, use_highly_variable=True)
    print(f"  PCA recomputed: {adata.obsm['X_pca'].shape}")
else:
    print(f"  Using existing PCA: {adata.obsm['X_pca'].shape}")

# Use X_emb as UMAP stand-in (avoid expensive UMAP recompute)
umap_key = next((k for k in ["X_umap","X_emb"] if k in adata.obsm), None)
if umap_key and umap_key != "X_umap":
    adata.obsm["X_umap"] = adata.obsm[umap_key]
elif "X_umap" not in adata.obsm:
    adata.obsm["X_umap"] = adata.obsm["X_pca"][:, :2]

X_pca = adata.obsm["X_pca"][:, :N_PCS].astype(np.float32)
print(f"  PCA (used): {X_pca.shape}")

# Quick overview plot
TIME_COL  = next((c for c in ["time_info","time","day"] if c in adata.obs), None)
STATE_COL = next((c for c in ["state_info","cell_type","celltype"] if c in adata.obs), None)

if TIME_COL and STATE_COL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    emb = adata.obsm["X_umap"]
    times = adata.obs[TIME_COL].astype(str).values
    states = adata.obs[STATE_COL].astype(str).values

    for t, col in [("2","steelblue"),("4","orange"),("6","crimson")]:
        m = times == t
        axes[0].scatter(emb[m,0], emb[m,1], s=0.5, alpha=0.3, color=col,
                        label=f"day {t} (n={m.sum():,})", rasterized=True)
    axes[0].legend(fontsize=7, markerscale=5); axes[0].set_title("Time point")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    cats = sorted(set(states))
    cmap = plt.get_cmap("tab20", len(cats))
    for i, s in enumerate(cats):
        m = states == s
        axes[1].scatter(emb[m,0], emb[m,1], s=0.5, alpha=0.3, color=cmap(i),
                        label=s, rasterized=True)
    axes[1].legend(fontsize=5, markerscale=8, ncol=2)
    axes[1].set_title("Cell state"); axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "larry_full_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: larry_full_overview.png")

    for t in sorted(set(times)):
        print(f"  day {t}: {(times==t).sum():,} cells")


# ── 3. Clone-based train/test split ─────────────────────────────────────────────
print(f"\nClone-based 80/20 train/test split ...")

assert "X_clone" in adata.obsm, "X_clone not found in obsm"
C = adata.obsm["X_clone"]
if hasattr(C, "toarray"):
    C = C.toarray()
C = np.array(C, dtype=np.float32)  # (n_cells, n_clones)
n_clones = C.shape[1]

# Find clones that appear in any two time points (usable for pairs)
time_arr = adata.obs[TIME_COL].astype(str).values if TIME_COL else np.full(len(adata), "?")
mt_clones = []
for j in range(n_clones):
    cells_j = np.where(C[:, j] != 0)[0]
    t_j = set(time_arr[cells_j])
    if len(t_j) >= 2:
        mt_clones.append(j)

mt_clones = np.array(mt_clones)
print(f"  Multi-timepoint clones: {len(mt_clones):,} / {n_clones:,}")

# Shuffle and split
shuffled = rng.permutation(mt_clones)
n_train  = int(len(shuffled) * TRAIN_FRACTION)
train_clones = shuffled[:n_train]
test_clones  = shuffled[n_train:]
print(f"  Train clones: {len(train_clones):,}  |  Test clones: {len(test_clones):,}")


# ── 4. Vectorised pair extraction with balancing ─────────────────────────────────

def extract_pairs_balanced(clone_set, time_pairs, C, time_arr, X_pca, state_arr,
                            max_per_type=MAX_PAIRS_PER_TYPE, label=""):
    """Extract (x0, x1, state_src, state_dst) pairs from a clone set.

    Balancing: after extraction, cap each (state_src, state_dst) combination
    at max_per_type by random subsampling.
    """
    all_src, all_dst = [], []
    all_ss, all_sd  = [], []  # source/dest state labels

    for t_early, t_late in time_pairs:
        early_mask = time_arr == t_early
        late_mask  = time_arr == t_late
        early_idx  = np.where(early_mask)[0]
        late_idx   = np.where(late_mask)[0]

        # Vectorised clone pair extraction using sparse-style column slices
        for j in clone_set:
            e_cells = early_idx[C[early_idx, j] != 0]
            l_cells = late_idx[ C[late_idx,  j] != 0]
            if len(e_cells) == 0 or len(l_cells) == 0:
                continue
            # Cartesian product
            ee, ll = np.meshgrid(e_cells, l_cells, indexing="ij")
            all_src.extend(ee.ravel().tolist())
            all_dst.extend(ll.ravel().tolist())
            if state_arr is not None:
                all_ss.extend([state_arr[i] for i in ee.ravel()])
                all_sd.extend([state_arr[i] for i in ll.ravel()])

    src = np.array(all_src, dtype=np.int64)
    dst = np.array(all_dst, dtype=np.int64)
    print(f"  {label}: {len(src):,} pairs before balancing")

    # Balance: cap each (src_state, dst_state) pair type
    if state_arr is not None and max_per_type is not None and len(src) > 0:
        ss = np.array(all_ss); sd = np.array(all_sd)
        keep = []
        type_counts = {}
        for st_s, st_d in zip(ss, sd):
            key = (st_s, st_d)
            type_counts[key] = type_counts.get(key, 0) + 1
        # Subsample per type
        from collections import defaultdict
        type_indices = defaultdict(list)
        for i, (s, d) in enumerate(zip(ss, sd)):
            type_indices[(s,d)].append(i)
        for key, idxs in type_indices.items():
            if len(idxs) > max_per_type:
                chosen = rng.choice(idxs, max_per_type, replace=False)
            else:
                chosen = idxs
            keep.extend(chosen)
        keep = np.array(keep)
        src = src[keep]; dst = dst[keep]; ss = ss[keep]; sd = sd[keep]
        print(f"  {label}: {len(src):,} pairs after balancing (cap={max_per_type}/type)")

        print(f"  Top transition types:")
        counts = {}
        for s, d in zip(ss, sd):
            counts[(s,d)] = counts.get((s,d), 0) + 1
        for (s,d), c in sorted(counts.items(), key=lambda x: -x[1])[:8]:
            print(f"    {s:<15s} -> {d:<15s}: {c:,}")

    return src, dst

state_arr = adata.obs[STATE_COL].astype(str).values if STATE_COL else None

print("\nExtracting TRAIN pairs ...")
src_tr, dst_tr = extract_pairs_balanced(
    train_clones, T_PAIRS, C, time_arr, X_pca, state_arr, label="TRAIN")

print("\nExtracting TEST pairs ...")
src_te, dst_te = extract_pairs_balanced(
    test_clones, T_PAIRS, C, time_arr, X_pca, state_arr,
    max_per_type=None, label="TEST")   # no balancing on test set

x0_tr = X_pca[src_tr]; x1_tr = X_pca[dst_tr]
x0_te = X_pca[src_te]; x1_te = X_pca[dst_te]

print(f"\nFinal pair counts:")
print(f"  Train: {len(src_tr):,}  |  Test: {len(src_te):,}")


# ── 5. Save ──────────────────────────────────────────────────────────────────────
print("\nSaving ...")
adata.write_h5ad(os.path.join(DATA_DIR, "larry_full_processed.h5ad"))
np.save(os.path.join(DATA_DIR, "larry_full_pca.npy"),       X_pca)
np.save(os.path.join(DATA_DIR, "larry_full_x0_train.npy"),  x0_tr)
np.save(os.path.join(DATA_DIR, "larry_full_x1_train.npy"),  x1_tr)
np.save(os.path.join(DATA_DIR, "larry_full_x0_test.npy"),   x0_te)
np.save(os.path.join(DATA_DIR, "larry_full_x1_test.npy"),   x1_te)
np.save(os.path.join(DATA_DIR, "larry_full_train_clones.npy"), train_clones)
np.save(os.path.join(DATA_DIR, "larry_full_test_clones.npy"),  test_clones)

if state_arr is not None:
    adata.obs[[c for c in [TIME_COL, STATE_COL] if c]].to_csv(
        os.path.join(DATA_DIR, "larry_full_obs.csv"))

print(f"\nAll outputs in {DATA_DIR}/")
print(f"  larry_full_processed.h5ad   — {adata.n_obs:,} cells")
print(f"  larry_full_pca.npy          — {X_pca.shape}")
print(f"  train pairs: x0={x0_tr.shape}  x1={x1_tr.shape}")
print(f"  test  pairs: x0={x0_te.shape}  x1={x1_te.shape}")
print(f"\nNext: run larry_isokann_multi_full.py")
