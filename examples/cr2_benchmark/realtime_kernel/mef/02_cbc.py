"""
02 — (re)compute CR2 cross-boundary correctness (CBC) for the RealTimeKernel.

Split out from 01 so it can be re-run cheaply: it rebuilds a PrecomputedKernel from
the saved transition matrix (artifacts/T.npz) instead of re-running GPCCA. Unlike the
pharyngeal benchmark, the MEF cell_sets labels are clean (the source population is the
real label "MEF/other", no NaN), so no relabeling is needed.

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
cell_type = np.load(os.path.join(A, "cell_type.npy"), allow_pickle=True).astype(str)

adata = AnnData(X=sp.csr_matrix(X))                    # dummy X; rep is X_pca
adata.obsm["X_pca"] = X
adata.obs["cell_sets"] = cell_type
adata.obs["cell_sets"] = adata.obs["cell_sets"].astype("category")
sc.pp.neighbors(adata, use_rep="X_pca",
                n_pcs=C.NEIGHBORS_N_PCS, n_neighbors=C.NEIGHBORS_N_NEIGHBORS,
                random_state=C.NEIGHBORS_RANDOM_STATE)

pk = PrecomputedKernel(T, adata=adata)

cats = set(adata.obs["cell_sets"].cat.categories)
boundaries = [(C.PROGENITOR_LABEL, t) for t in C.TERMINAL_STATES if t in cats]
cbc = {}
for s, t in boundaries:
    if s not in cats or t not in cats:
        print(f"  skip {s}->{t} (label absent)"); continue
    arr = np.asarray(pk.cbc(source=s, target=t, cluster_key="cell_sets", rep="X_pca"))
    cbc[f"{s}->{t}"] = arr
    print(f"  CBC {s}->{t}: n={arr.shape[0]} mean={np.nanmean(arr):.3f}")

np.savez(os.path.join(A, "cr2_cbc.npz"), **cbc)
print("Saved cr2_cbc.npz")
