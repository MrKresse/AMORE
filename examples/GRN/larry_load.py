"""
LARRY hematopoiesis: data loading, preprocessing, and Koopman-pair extraction.

Weinreb, Rodriguez-Fraticelli, Camargo, Klein (Science 2020, doi:10.1126/science.aaw3381)
~48k mouse hematopoietic progenitor cells with expressed-barcode lineage tracing,
sampled at days 2, 4, and 6.  Multi-time-point clones supply (ancestor, descendant)
pairs that play the role of Koopman pairs in ISOKANN.

Requirements
------------
    pip install cospar scanpy anndata matplotlib

The cospar package downloads and caches a curated 48k-cell subset of LARRY automatically.
For the full 300k-cell dataset use GEO accession GSE140802.

Outputs (written to ./data/)
-----------------------------
    larry_processed.h5ad  – AnnData with PCA and UMAP
    larry_pca.npy          – PCA embedding, shape (n_cells, 50)
    larry_x0.npy           – ancestor features (day T_EARLY), shape (n_pairs, 50)
    larry_x1.npy           – descendant features (day T_LATE),  shape (n_pairs, 50)
    larry_src.npy          – integer cell indices for x0
    larry_dst.npy          – integer cell indices for x1
    larry_obs.csv          – cell metadata (time, clone, state)
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")   # non-interactive — never call plt.close("all")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Column names as stored by cospar -- printed below in case they differ in your version
TIME_COL  = "time_info"
CLONE_COL = "clone_id"
STATE_COL = "state_info"

# Time-point labels to form pairs from.  Cospar encodes them as integers or strings;
# adjust to match what the print-out below shows.
T_EARLY = "2"   # day-2 label
T_LATE  = "4"   # day-4 label

N_HVG   = 2000  # highly-variable genes for PCA
N_PCS   = 40    # PCA components (cospar provides 40)


# ── 1. Load ────────────────────────────────────────────────────────────────────

print("Loading LARRY hematopoiesis data via cospar …")
try:
    import cospar as cs
    adata = cs.datasets.hematopoiesis()
    print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
except ImportError as exc:
    raise ImportError(
        "cospar is required for automatic data download.\n"
        "  pip install cospar\n\n"
        "Alternatively download from GEO (GSE140802) and load manually:\n"
        "  adata = sc.read_10x_h5('path/to/filtered_feature_bc_matrix.h5')"
    ) from exc

# Show what metadata columns are available — adjust TIME_COL / CLONE_COL above if needed
print("\n── obs columns ──────────────────────────────────────────────────────────")
for col in adata.obs.columns:
    uniq = adata.obs[col].dropna().unique()
    sample = sorted(str(v) for v in uniq[:6])
    print(f"  {col:<30s}  {sample}")

print("\n── obsm keys ────────────────────────────────────────────────────────────")
for k in adata.obsm:
    print(f"  {k:<20s}  shape {adata.obsm[k].shape}")


# ── 2. Preprocessing ──────────────────────────────────────────────────────────

print("\nPreprocessing …")

# cospar returns fully processed data: log-normalised X, PCA and UMAP already in obsm.
# We reuse those embeddings directly instead of recomputing, which would require
# going back to raw counts (unavailable here) and would risk double-normalisation.

if "X_pca" not in adata.obsm:
    # Raw counts path — only reached when loading from GEO directly
    if adata.raw is not None:
        adata = adata.raw.to_adata()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # seurat_v3 flavour expects raw counts; use 'cell_ranger' on log-normalised data
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="cell_ranger")
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=N_PCS, use_highly_variable=True)
    sc.pp.neighbors(adata, n_pcs=N_PCS)
    sc.tl.umap(adata)
    print(f"  PCA recomputed:  {adata.obsm['X_pca'].shape}")
else:
    print(f"  Using existing PCA: {adata.obsm['X_pca'].shape}")
    # Check for UMAP under various key names cospar might use
    umap_candidates = ["X_umap", "X_emb", "X_draw_graph_fa", "X_diffmap"]
    umap_key = next((k for k in umap_candidates if k in adata.obsm), None)
    if umap_key and umap_key != "X_umap":
        adata.obsm["X_umap"] = adata.obsm[umap_key]
    if "X_umap" not in adata.obsm:
        # Use PCA1/2 as a fast stand-in — avoids multi-minute UMAP on CPU
        print("  No UMAP found — using PC1/PC2 as 2-D embedding (fast stand-in)")
        adata.obsm["X_umap"] = adata.obsm["X_pca"][:, :2]

# Trim PCA to N_PCS components if cospar stored more
adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :N_PCS]
print(f"  PCA shape (used): {adata.obsm['X_pca'].shape}")
print(f"  UMAP shape:       {adata.obsm['X_umap'].shape}")


# ── 3. Quick exploratory plots ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

if STATE_COL in adata.obs:
    sc.pl.umap(adata, color=STATE_COL, ax=axes[0], show=False, title="Cell state")
if TIME_COL in adata.obs:
    sc.pl.umap(adata, color=TIME_COL,  ax=axes[1], show=False, title="Time point")

plt.tight_layout()
fig.savefig(os.path.join(DATA_DIR, "larry_overview.png"), dpi=150)
plt.close("all")
print(f"  Saved: {DATA_DIR}/larry_overview.png")


# ── 4. Clone-pair extraction ──────────────────────────────────────────────────
# Cospar stores clone membership as a cell×clone binary matrix in obsm['X_clone'],
# not as an obs column.  Each column j encodes one barcode; X_clone[i,j] != 0
# means cell i carries barcode j.

print(f"\nExtracting clone pairs (day2->day4 AND day2->day6 for longer-range dynamics) ...")

if TIME_COL not in adata.obs:
    raise KeyError(f"'{TIME_COL}' not found. Available: {list(adata.obs.columns)}")

time_arr   = adata.obs[TIME_COL].astype(str).values
mask_d2    = time_arr == "2"
mask_d4    = time_arr == "4"
mask_d6    = time_arr == "6"
print(f"  Cells: day2={mask_d2.sum():,}  day4={mask_d4.sum():,}  day6={mask_d6.sum():,}")

def extract_pairs_from_clone_matrix(C, src_time_mask, dst_time_mask, label):
    src_idx = np.where(src_time_mask)[0]
    dst_idx = np.where(dst_time_mask)[0]
    s_list, d_list = [], []
    for j in range(C.shape[1]):
        s_cells = src_idx[C[src_idx, j] != 0]
        d_cells = dst_idx[C[dst_idx, j] != 0]
        if len(s_cells) == 0 or len(d_cells) == 0:
            continue
        for i in s_cells:
            for k in d_cells:
                s_list.append(int(i)); d_list.append(int(k))
    n_clones = sum(1 for j in range(C.shape[1])
                   if (C[src_idx, j] != 0).any() and (C[dst_idx, j] != 0).any())
    print(f"  {label}: {len(s_list):,} pairs from {n_clones:,} clones")
    return s_list, d_list

if "X_clone" in adata.obsm:
    C = adata.obsm["X_clone"]
    if hasattr(C, "toarray"):
        C = C.toarray()
    C = np.array(C, dtype=np.float32)

    s24, d24 = extract_pairs_from_clone_matrix(C, mask_d2, mask_d4, "day2->day4 (short lag)")
    s26, d26 = extract_pairs_from_clone_matrix(C, mask_d2, mask_d6, "day2->day6 (long lag )")
    s46, d46 = extract_pairs_from_clone_matrix(C, mask_d4, mask_d6, "day4->day6 (mid  lag )")

    # Combine all lags — use all raw clone pairs, no reweighting
    src_list = s24 + s26 + s46
    dst_list = d24 + d26 + d46
    print(f"  Combined: {len(src_list):,} pairs total")

elif CLONE_COL in adata.obs:
    # Fallback: clone ID as obs column
    obs = adata.obs.copy()
    src_list, dst_list = [], []
    for clone, grp in obs.groupby(CLONE_COL):
        e = grp.index[grp[TIME_COL].astype(str) == T_EARLY]
        l = grp.index[grp[TIME_COL].astype(str) == T_LATE]
        e_idx = [obs.index.get_loc(x) for x in e]
        l_idx = [obs.index.get_loc(x) for x in l]
        for i in e_idx:
            for k in l_idx:
                src_list.append(i); dst_list.append(k)
else:
    raise KeyError(
        "No clone information found. Expected 'X_clone' in obsm or "
        f"'{CLONE_COL}' in obs.\nobs columns: {list(adata.obs.columns)}"
    )

src = np.array(src_list, dtype=np.int64)
dst = np.array(dst_list, dtype=np.int64)

X = adata.obsm["X_pca"].astype(np.float32)
x0 = X[src]
x1 = X[dst]

print(f"  Total pairs: {len(src):,}")
print(f"  Feature dim: {X.shape[1]}")

# ── 5. Save ───────────────────────────────────────────────────────────────────

adata.write_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
np.save(os.path.join(DATA_DIR, "larry_pca.npy"), X)
np.save(os.path.join(DATA_DIR, "larry_x0.npy"),  x0)
np.save(os.path.join(DATA_DIR, "larry_x1.npy"),  x1)
np.save(os.path.join(DATA_DIR, "larry_src.npy"), src)
np.save(os.path.join(DATA_DIR, "larry_dst.npy"), dst)
save_cols = [c for c in [TIME_COL, STATE_COL] if c in adata.obs.columns]
adata.obs[save_cols].to_csv(os.path.join(DATA_DIR, "larry_obs.csv"))

print(f"\nAll outputs written to {DATA_DIR}/")
print("  larry_processed.h5ad  – full AnnData")
print(f"  larry_pca.npy          – PCA of all cells,   {X.shape}")
print(f"  larry_x0.npy           – ancestor features,  {x0.shape}")
print(f"  larry_x1.npy           – descendant features,{x1.shape}")
print("\nNext: run larry_isokann.py")
