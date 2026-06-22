"""
02 — (re)compute CR2 cross-boundary correctness (CBC) for the RealTimeKernel.

Split out from 01 so it can be re-run cheaply: it rebuilds a PrecomputedKernel
from the saved RealTimeKernel transition matrix (artifacts/T.npz) instead of
recomputing WOT + GPCCA. The progenitor population has NaN cluster_name in the
raw data; CR2 relabels it "progenitors" (cluster_name_full). CBC needs a valid
source label, so we apply the same relabel and also rewrite cell_type.npy so the
ISOKANN-side CBC in 04 uses the identical labels.

Runs in cr2-py310. CBC definition is cellrank's kernel.cbc (same as the CR2
cbc_ptk_vs_vk benchmark): cross-boundary correctness from transition mass flow,
per source cell, using the X_pca representation and the distances graph.
"""
from __future__ import annotations
import os, sys
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData
from cellrank.kernels import PrecomputedKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

A = C.ARTIFACTS

T = sp.load_npz(os.path.join(A, "T.npz")).tocsr()
X = np.load(os.path.join(A, "features.npy")).astype(np.float32)
cell_type = np.load(os.path.join(A, "cell_type.npy"), allow_pickle=True).astype(object)
cell_type[cell_type == "nan"] = "progenitors"          # match CR2 cluster_name_full

adata = AnnData(X=sp.csr_matrix(X))                    # dummy X; rep is X_pca
adata.obsm["X_pca"] = X
adata.obs["cluster_name"] = cell_type
adata.obs["cluster_name"] = adata.obs["cluster_name"].astype("category")
sc.pp.neighbors(adata, use_rep="X_pca",
                n_pcs=C.NEIGHBORS_N_PCS, n_neighbors=C.NEIGHBORS_N_NEIGHBORS)

pk = PrecomputedKernel(T, adata=adata)

src = "progenitors"
boundaries = [(src, t) for t in C.TERMINAL_STATES] + [("early_thymus", t) for t in ["cTEC", "mTEC"]]
cbc = {}
cats = set(adata.obs["cluster_name"].cat.categories)
for s, t in boundaries:
    if s not in cats or t not in cats:
        print(f"  skip {s}->{t} (label absent)"); continue
    arr = np.asarray(pk.cbc(source=s, target=t, cluster_key="cluster_name", rep="X_pca"))
    cbc[f"{s}->{t}"] = arr
    print(f"  CBC {s}->{t}: n={arr.shape[0]} mean={np.nanmean(arr):.3f}")

np.savez(os.path.join(A, "cr2_cbc.npz"), **cbc)
np.save(os.path.join(A, "cell_type.npy"), cell_type.astype(str))  # consistent labels downstream
print("Saved cr2_cbc.npz and relabeled cell_type.npy")
