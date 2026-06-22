"""
01 — Reproduce CellRank2 (RealTimeKernel "mef" analysis, Schiebinger reprogramming)
and emit the artifacts both arms consume.

Runs in the `cr2-py310` env (cellrank 2.0.7). Follows
cellrank2_reproducibility/scripts/realtime_kernel/mef/realtime_informed_pseudotime.py
verbatim for every PINNED step (see config.py), adding only: (a) artifact emission
for the ISOKANN arm, and (b) the CR2 head-to-head numbers (fate probs, drivers, CBC,
TSI, purity).

The serum subset ships the published CR2 RealTimeKernel transition matrix in
obsp["transition_matrix"] (row-stochastic, forward), so — unlike the reproducibility
script that reloads it from all_connectivities.npz — we read it straight off the
AnnData and wrap it in a PrecomputedKernel for GPCCA. This is the identical matrix
CR2 publishes; no WOT recomputation.

Emitted to artifacts/ (same contract as realtime_kernel/pharyngeal/):
    features.npy          (N,50)   X_pca on HVGs — the ISOKANN features
    T.npz                 (N,N)    RealTimeKernel transition matrix (row-stochastic)
    pca_loadings.npy      (50,G)   PCA components on the HVG gene universe
    pca_mean.npy          (G,)     per-HVG mean (for the gradient chain rule)
    hvg_genes.npy         (G,)     HVG gene names (the driver gene universe)
    hvg_expr.npy          (N,G)    HVG expression (z-scored downstream by the nets)
    cell_type.npy         (N,)     cell_sets
    cluster_fine.npy      (N,)     cell_sets (kept for artifact parity)
    day.npy               (N,)     experimental day
    umap.npy              (N,2)    CR2 force-directed embedding
    cr2_fate_probs.npy    (N,4)    GPCCA fate/absorption probs (col order in json)
    cr2_macrostates.npy   (N,)     macrostate assignment
    cr2_drivers.csv                IPS_corr / pval / rank for every gene (if available)
    cr2_cbc.npz                    per-boundary CBC arrays (RTK)
    cr2_tsi.json                   TSI score
    cr2_purity.json                macrostate + terminal-state purity
    cr2_config_resolved.json       N, n_hvg, terminal order, macro->cluster, boundaries
"""

from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import scipy.sparse as sp

import scanpy as sc
import cellrank as cr
from cellrank.kernels import PrecomputedKernel
from cellrank.estimators import GPCCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
# SLEPc/PETSc-free sparse Schur backend for GPCCA on the 165k-cell matrix (Windows).
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
    print("Loading reprogramming_schiebinger_serum.h5ad ...", flush=True)
    adata = sc.read(C.MEF_RAW)
    print(f"  shape: {adata.shape}", flush=True)
    print(f"  obsm: {list(adata.obsm.keys())}  obsp: {list(adata.obsp.keys())}")
    print(f"  cell_sets: {adata.obs[C.CELLTYPE_KEY].value_counts().to_dict()}")
    n_hvg = int(adata.var[C.HVG_VAR_KEY].sum())
    print(f"  shipped HVGs (var['{C.HVG_VAR_KEY}']): {n_hvg}", flush=True)

    # ── preprocessing (PINNED; sc.pp.pca auto-uses the shipped HVG mask) ──────
    sc.pp.pca(adata, n_comps=C.PCA_N_COMPS)
    sc.pp.neighbors(adata, random_state=C.NEIGHBORS_RANDOM_STATE)

    # ── RealTimeKernel: the published transition matrix shipped in obsp ────────
    T = sp.csr_matrix(adata.obsp[C.RTK_OBSP_KEY])
    rs = np.asarray(T.sum(1)).ravel()
    print(f"  T shape {T.shape}  nnz {T.nnz}  row-sum range [{rs.min():.3f}, {rs.max():.3f}]",
          flush=True)
    pk = PrecomputedKernel(T, adata=adata)

    # ── GPCCA estimator (PINNED) ──────────────────────────────────────────────
    g = GPCCA(pk)
    g.compute_schur(n_components=C.SCHUR_N_COMPONENTS)
    try:
        tsi_score = float(g.tsi(n_macrostates=C.TSI_N_MACROSTATES,
                                terminal_states=C.TSI_TERMINAL_STATES,
                                cluster_key=C.MACROSTATE_CLUSTER_KEY))
    except Exception as e:
        print(f"  [tsi] skipped: {e}"); tsi_score = float("nan")

    g.compute_macrostates(n_states=C.N_MACROSTATES, cluster_key=C.MACROSTATE_CLUSTER_KEY)
    print(f"  macrostates: {list(g.macrostates.cat.categories)}", flush=True)
    macro_purity = get_state_purity(adata, g, "macrostates", C.MACROSTATE_CLUSTER_KEY)
    g.set_terminal_states(C.TERMINAL_STATES)
    term_purity = get_state_purity(adata, g, "terminal_states", C.MACROSTATE_CLUSTER_KEY)

    # Direct sparse solve: factorize (I - Q) once and solve all lineages' RHS. On the
    # 165k-cell system per-RHS gmres fails to converge for the hardest lineage; the
    # direct solver is exact and reliable. (use_petsc=False -> scipy SuperLU.)
    try:
        g.compute_fate_probabilities(solver="direct", use_petsc=False)
    except Exception as e:
        print(f"  [fate] direct failed ({e}); retrying gmres", flush=True)
        g.compute_fate_probabilities(solver="gmres", use_petsc=False)

    fate = g.fate_probabilities                         # Lineage object
    fate_names = list(fate.names)
    fate_probs = np.asarray(fate.X)                     # (N, n_term)
    print(f"  fate lineages: {fate_names}", flush=True)

    macro = g.macrostates.astype(str).values
    cats = set(adata.obs[C.CELLTYPE_KEY].cat.categories)
    boundaries = [(C.PROGENITOR_LABEL, t) for t in C.TERMINAL_STATES if t in cats]

    # ── emit ISOKANN-arm artifacts FIRST (the notebook's data contract) ────────
    # Drivers and CBC come afterwards: they are optional extras (the consolidated
    # benchmark notebook recomputes the GPCCA driver ranking itself and shows no CBC
    # panel), and CBC over the 91k-cell MEF/other source is slow — so we never let
    # them gate the essential artifacts.
    hvg_mask = adata.var[C.HVG_VAR_KEY].values.astype(bool)
    hvg_genes = adata.var_names[hvg_mask].to_numpy()
    sp.save_npz(os.path.join(C.ARTIFACTS, "T.npz"), T)
    np.save(os.path.join(C.ARTIFACTS, "features.npy"),
            np.asarray(adata.obsm["X_pca"][:, :C.ISOKANN_N_PCS], dtype=np.float32))
    PCs = adata.varm["PCs"][:, :C.ISOKANN_N_PCS]        # (n_genes_all, 50)
    np.save(os.path.join(C.ARTIFACTS, "pca_loadings.npy"),
            PCs[hvg_mask].T.astype(np.float32))         # (50, G)
    Xhvg = adata[:, hvg_mask].X
    Xhvg = Xhvg.toarray() if sp.issparse(Xhvg) else np.asarray(Xhvg)
    np.save(os.path.join(C.ARTIFACTS, "pca_mean.npy"), Xhvg.mean(0).astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "hvg_expr.npy"), Xhvg.astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "hvg_genes.npy"), hvg_genes)
    np.save(os.path.join(C.ARTIFACTS, "cell_type.npy"),
            adata.obs[C.CELLTYPE_KEY].astype(str).to_numpy())
    np.save(os.path.join(C.ARTIFACTS, "cluster_fine.npy"),
            adata.obs[C.CELLTYPE_KEY].astype(str).to_numpy())
    np.save(os.path.join(C.ARTIFACTS, "day.npy"),
            adata.obs[C.TIME_KEY].astype(str).to_numpy())
    np.save(os.path.join(C.ARTIFACTS, "umap.npy"),
            np.asarray(adata.obsm["X_force_directed"], dtype=np.float32))
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
        "macrostates": list(map(str, np.unique(macro))),
        "progenitor_label": C.PROGENITOR_LABEL,
        "cbc_boundaries": [f"{s}->{t}" for s, t in boundaries],
        "tsi": tsi_score,
    }
    with open(C.RESOLVED_JSON, "w") as f:
        json.dump(resolved, f, indent=2)
    print("\nEssential artifacts written. Resolved config:")
    print(json.dumps(resolved, indent=2), flush=True)

    # ── optional extras: GPCCA lineage drivers + CBC (not used by the notebook) ──
    try:
        drivers = g.compute_lineage_drivers(
            return_drivers=True, cluster_key=C.DRIVER_CLUSTER_KEY,
            lineages=[C.DRIVER_LINEAGE],
        )
        drivers.to_csv(os.path.join(C.ARTIFACTS, "cr2_drivers.csv"))
        print(f"  drivers computed for {C.DRIVER_LINEAGE}", flush=True)
    except Exception as e:
        print(f"  [drivers] skipped: {e}", flush=True)

    # CBC source-capped: cellrank's cbc is per source-cell, and MEF/other has 91797
    # cells. Subsample the source to keep it tractable (CBC is an extra artifact only).
    CBC_SRC_CAP = 4000
    cbc_lab = adata.obs[C.CELLTYPE_KEY].astype(str).to_numpy().copy()
    src_idx = np.where(cbc_lab == C.PROGENITOR_LABEL)[0]
    if len(src_idx) > CBC_SRC_CAP:
        rng = np.random.default_rng(0)
        drop = rng.choice(src_idx, size=len(src_idx) - CBC_SRC_CAP, replace=False)
        cbc_lab[drop] = "MEF_bg"
    adata.obs["cell_sets_cbc"] = cbc_lab
    adata.obs["cell_sets_cbc"] = adata.obs["cell_sets_cbc"].astype("category")
    cbc = {}
    for src, tgt in boundaries:
        try:
            cbc[f"{src}->{tgt}"] = np.asarray(
                pk.cbc(source=src, target=tgt, cluster_key="cell_sets_cbc", rep=C.CBC_REP))
            print(f"  CBC {src}->{tgt}: mean={np.nanmean(cbc[f'{src}->{tgt}']):.3f}", flush=True)
        except Exception as e:
            print(f"  [cbc] {src}->{tgt} failed: {e}", flush=True)
    if cbc:
        np.savez(os.path.join(C.ARTIFACTS, "cr2_cbc.npz"), **cbc)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
