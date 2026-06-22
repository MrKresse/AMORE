"""
01 — Reproduce CellRank2 (PseudotimeKernel "hematopoiesis" analysis, NeurIPS 2021
human bone-marrow) and emit the artifacts both arms consume.

Runs in the `cr2-py310` env (cellrank 2.0.7). Follows
cellrank2_reproducibility/scripts/pseudotime_kernel/hematopoiesis/dpt.py verbatim
for every PINNED step (see config.py): cell-type subset -> neighbours on the MultiVI
latent -> diffusion pseudotime from the HSC root -> PseudotimeKernel (soft scheme) ->
GPCCA (schur 20, macrostates 6, the 4 terminal states, fate probabilities). It adds
only (a) artifact emission for the ISOKANN arm, and (b) the CR2 head-to-head numbers
(fate probs, drivers, TSI, purity).

Unlike the WOT RealTimeKernel benchmarks, the transition matrix here is built live by
the PseudotimeKernel from the kNN graph + DPT, so we construct it exactly as dpt.py
does and read T off the kernel. ISOKANN's features are the 50 PCs computed on the
dataset's own shipped HVG mask (var["hvg_multiVI"]) — see config.py.

Emitted to artifacts/ (same contract as realtime_kernel/mef/):
    features.npy          (N,50)   X_pca on HVGs — the ISOKANN features
    T.npz                 (N,N)    PseudotimeKernel transition matrix (row-stochastic)
    pca_loadings.npy      (50,G)   PCA components on the HVG gene universe
    pca_mean.npy          (G,)     per-HVG mean (for the gradient chain rule)
    hvg_genes.npy         (G,)     HVG gene names (the driver gene universe)
    hvg_expr.npy          (N,G)    HVG expression (log-norm; z-scored downstream by nets)
    cell_type.npy         (N,)     l2_cell_type
    cluster_fine.npy      (N,)     l2_cell_type (artifact parity)
    day.npy               (N,)     dpt_pseudotime (the "time" coordinate here)
    umap.npy              (N,2)    CR2 UMAP embedding (shipped X_umap on the subset)
    cr2_fate_probs.npy    (N,4)    GPCCA fate/absorption probs (col order in json)
    cr2_macrostates.npy   (N,)     macrostate assignment
    cr2_drivers.csv                pDC_corr / pval / rank for every gene (if available)
    cr2_tsi.json                   TSI score
    cr2_purity.json                macrostate + terminal-state purity
    cr2_config_resolved.json       N, n_hvg, terminal order, fate columns, boundaries
"""

from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import scipy.sparse as sp

import scanpy as sc
import cellrank as cr
from cellrank.estimators import GPCCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
# SLEPc/PETSc-free sparse Schur backend for GPCCA on Windows (no SLEPc build).
# Must be imported AFTER cellrank/pygpcca so it can rebind their sorted_schur.
import _sparse_schur_patch  # noqa: E402,F401

sc.settings.verbosity = 1
cr.settings.verbosity = 2
np.random.seed(0)


# ── small inlined CR2 helper (avoid importing cr2 pkg -> pulls dynamo) ─────────
def get_state_purity(adata, estimator, states: str, obs_col: str) -> dict:
    states = getattr(estimator, states)
    max_obs = (
        pd.DataFrame({"states": states, "obs_col": adata.obs[obs_col]})[~states.isnull()]
        .groupby(["states", "obs_col"]).size().reset_index()
        .rename(columns={0: "group_counts"})[["states", "group_counts"]]
        .groupby("states").max()["group_counts"]
    )
    return (max_obs / states.value_counts()).to_dict()


def main():
    print("Loading gex_preprocessed.h5ad ...", flush=True)
    adata = sc.read(C.HEMATO_RAW)
    print(f"  full shape: {adata.shape}", flush=True)

    # ── cell-type subset (PINNED) ─────────────────────────────────────────────
    adata = adata[adata.obs[C.CELLTYPE_KEY].isin(C.CELLTYPES_TO_KEEP), :].copy()
    print(f"  subset shape: {adata.shape}", flush=True)
    print(f"  {C.CELLTYPE_KEY}: {adata.obs[C.CELLTYPE_KEY].value_counts().to_dict()}", flush=True)

    # ── neighbours on the MultiVI latent + diffusion pseudotime (PINNED) ──────
    sc.pp.neighbors(adata, use_rep=C.NEIGHBORS_USE_REP)
    sc.tl.diffmap(adata, n_comps=C.DIFFMAP_N_COMPS)
    # root = HSC cell that maximises diffusion component DIFFMAP_ROOT_COMP
    df = (
        pd.DataFrame({"diff_comp": adata.obsm["X_diffmap"][:, C.DIFFMAP_ROOT_COMP],
                      "cell_type": adata.obs[C.CELLTYPE_KEY].values})
        .reset_index().rename({"index": "obs_id"}, axis=1)
    )
    df = df.loc[df["cell_type"] == C.ROOT_CLUSTER, "diff_comp"]
    root_idx = df.index[df.argmax()]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata, n_dcs=C.DPT_N_DCS)
    print(f"  DPT root cell idx {root_idx}; pseudotime range "
          f"[{adata.obs[C.TIME_KEY].min():.3f}, {adata.obs[C.TIME_KEY].max():.3f}]", flush=True)

    # ── PseudotimeKernel (PINNED) ──────────────────────────────────────────────
    ptk = cr.kernels.PseudotimeKernel(adata, time_key=C.TIME_KEY).compute_transition_matrix(
        threshold_scheme=C.PTK_THRESHOLD_SCHEME)
    T = sp.csr_matrix(ptk.transition_matrix)
    rs = np.asarray(T.sum(1)).ravel()
    print(f"  T shape {T.shape}  nnz {T.nnz}  row-sum range [{rs.min():.3f}, {rs.max():.3f}]", flush=True)

    # ── GPCCA estimator (PINNED) ──────────────────────────────────────────────
    g = GPCCA(ptk)
    g.compute_schur(n_components=C.SCHUR_N_COMPONENTS)
    try:
        tsi_score = float(g.tsi(n_macrostates=C.TSI_N_MACROSTATES,
                                terminal_states=C.TSI_TERMINAL_STATES,
                                cluster_key=C.MACROSTATE_CLUSTER_KEY))
    except Exception as e:
        print(f"  [tsi] skipped: {e}"); tsi_score = float("nan")

    g.compute_macrostates(n_states=C.N_MACROSTATES, cluster_key=C.MACROSTATE_CLUSTER_KEY)
    macro_names = list(g.macrostates.cat.categories)
    print(f"  macrostates ({len(macro_names)}): {macro_names}", flush=True)
    macro_purity = get_state_purity(adata, g, "macrostates", C.MACROSTATE_CLUSTER_KEY)

    # set the 4 terminal states (fall back to base-name match if GPCCA suffixed them)
    def _resolve(term):
        if term in macro_names:
            return term
        cand = [m for m in macro_names if m.split("_")[0] == term]
        return cand[0] if cand else None
    terms_resolved = [t for t in (_resolve(x) for x in C.TERMINAL_STATES) if t is not None]
    print(f"  terminal states resolved: {terms_resolved}", flush=True)
    g.set_terminal_states(terms_resolved)
    term_purity = get_state_purity(adata, g, "terminal_states", C.MACROSTATE_CLUSTER_KEY)

    g.compute_fate_probabilities(tol=C.FATE_TOL, solver="direct", use_petsc=False)
    fate = g.fate_probabilities
    fate_names = list(fate.names)
    fate_probs = np.asarray(fate.X)
    print(f"  fate lineages: {fate_names}", flush=True)

    macro = g.macrostates.astype(str).values
    cats = set(adata.obs[C.CELLTYPE_KEY].cat.categories)
    boundaries = [(C.PROGENITOR_LABEL, t) for t in C.TERMINAL_STATES if t in cats]

    # ── ISOKANN features: PCA on the shipped HVG mask ─────────────────────────
    hvg_mask = (adata.var[C.HVG_VAR_KEY].astype(str) == C.HVG_VAR_TRUE).values
    n_hvg = int(hvg_mask.sum())
    print(f"  shipped HVGs (var['{C.HVG_VAR_KEY}']): {n_hvg}", flush=True)
    adata_hvg = adata[:, hvg_mask].copy()
    sc.pp.pca(adata_hvg, n_comps=C.PCA_N_COMPS)
    X_pca = np.asarray(adata_hvg.obsm["X_pca"][:, :C.ISOKANN_N_PCS], dtype=np.float32)
    PCs = np.asarray(adata_hvg.varm["PCs"][:, :C.ISOKANN_N_PCS], dtype=np.float32)  # (G,50)
    hvg_genes = adata_hvg.var_names.to_numpy()
    Xhvg = adata_hvg.X
    Xhvg = Xhvg.toarray() if sp.issparse(Xhvg) else np.asarray(Xhvg)

    # ── emit ISOKANN-arm artifacts (the notebook's data contract) ─────────────
    sp.save_npz(os.path.join(C.ARTIFACTS, "T.npz"), T)
    np.save(os.path.join(C.ARTIFACTS, "features.npy"), X_pca)
    np.save(os.path.join(C.ARTIFACTS, "pca_loadings.npy"), PCs.T.astype(np.float32))  # (50,G)
    np.save(os.path.join(C.ARTIFACTS, "pca_mean.npy"), Xhvg.mean(0).astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "hvg_expr.npy"), Xhvg.astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "hvg_genes.npy"), hvg_genes)
    np.save(os.path.join(C.ARTIFACTS, "cell_type.npy"),
            adata.obs[C.CELLTYPE_KEY].astype(str).to_numpy())
    np.save(os.path.join(C.ARTIFACTS, "cluster_fine.npy"),
            adata.obs[C.CELLTYPE_KEY].astype(str).to_numpy())
    np.save(os.path.join(C.ARTIFACTS, "day.npy"),
            adata.obs[C.TIME_KEY].to_numpy().astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "umap.npy"),
            np.asarray(adata.obsm["X_umap"], dtype=np.float32))
    np.save(os.path.join(C.ARTIFACTS, "cr2_fate_probs.npy"), fate_probs.astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "cr2_macrostates.npy"), macro)

    with open(os.path.join(C.ARTIFACTS, "cr2_tsi.json"), "w") as f:
        json.dump({"tsi": tsi_score, "tsi_n_macrostates": C.TSI_N_MACROSTATES}, f, indent=2)
    with open(os.path.join(C.ARTIFACTS, "cr2_purity.json"), "w") as f:
        json.dump({"macrostate": {str(k): float(v) for k, v in macro_purity.items()},
                   "terminal": {str(k): float(v) for k, v in term_purity.items()}}, f, indent=2)

    resolved = {
        "N": int(adata.n_obs), "n_hvg": int(n_hvg),
        "n_pcs_features": C.ISOKANN_N_PCS,
        "celltype_key": C.CELLTYPE_KEY,
        "cell_set_categories": sorted(cats),
        "terminal_states": C.TERMINAL_STATES,
        "fate_prob_columns": fate_names,
        "k_macrostates": C.N_MACROSTATES,
        "macrostates": macro_names,
        "progenitor_label": C.PROGENITOR_LABEL,
        "cbc_boundaries": [f"{s}->{t}" for s, t in boundaries],
        "tsi": tsi_score,
    }
    with open(C.RESOLVED_JSON, "w") as f:
        json.dump(resolved, f, indent=2)
    print("\nEssential artifacts written. Resolved config:")
    print(json.dumps(resolved, indent=2), flush=True)

    # ── optional extra: GPCCA lineage drivers (not required by the notebook) ──
    try:
        drivers = g.compute_lineage_drivers(
            return_drivers=True, cluster_key=C.DRIVER_CLUSTER_KEY,
            lineages=[C.DRIVER_LINEAGE], clusters=[C.PROGENITOR_LABEL, C.DRIVER_LINEAGE],
        )
        drivers.to_csv(os.path.join(C.ARTIFACTS, "cr2_drivers.csv"))
        print(f"  drivers computed for {C.DRIVER_LINEAGE}", flush=True)
    except Exception as e:
        print(f"  [drivers] skipped: {e}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
