"""
01 — Reproduce CellRank2 (Fig 2, pharyngeal endoderm "subsetted data") and emit
the artifacts both arms consume.

Runs in the `cr2-py310` env (cellrank 2.0.7 + wot). Follows
cellrank2_reproducibility/scripts/realtime_kernel/pharyngeal_endoderm/
realtimekernel_subsetted_data.py verbatim for every PINNED step (see config.py),
adding only: (a) artifact emission for the ISOKANN arm, and (b) the CR2
head-to-head numbers (fate probs, drivers, CBC, TSI, purity).

Emitted to artifacts/:
    features.npy          (N,50)   X_pca on HVGs — the ISOKANN features
    T.npz                 (N,N)    RealTimeKernel transition matrix (row-stochastic)
    pca_loadings.npy      (50,G)   PCA components on the HVG gene universe
    pca_mean.npy          (G,)     per-HVG mean used by sc.tl.pca (for chain rule)
    hvg_genes.npy         (G,)     HVG gene names (the driver gene universe)
    cell_type.npy         (N,)     cluster_name
    cluster_fine.npy      (N,)     cluster_fine
    day.npy               (N,)     experimental day
    umap.npy              (N,2)    CR2 UMAP
    cr2_fate_probs.npy    (N,4)    GPCCA fate/absorption probs (col order in json)
    cr2_macrostates.npy   (N,)     macrostate assignment (argmax membership)
    cr2_drivers.csv                mTEC_corr / pval / rank for every gene
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
import wot  # noqa: F401  (required for RealTimeKernel.from_wot)
from cellrank.kernels import RealTimeKernel
from cellrank.estimators import GPCCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

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


def resolve_cluster_fine(adata) -> pd.Series:
    """
    cluster_fine = cluster_data.csv['res.1'] in the CR2 script. The figshare h5ad
    may already carry the fine clustering under a different obs key. Try, in order:
    an external cluster_data.csv, then common obs keys. Fail loudly with the list
    of available columns if none is found (so we never silently guess the subset).
    """
    csv = os.path.join(C.CR2_DATA, "pharyngeal_endoderm", "raw", "cluster_data.csv")
    if os.path.exists(csv):
        s = pd.read_csv(csv, index_col=0).loc[adata.obs_names, "res.1"]
        return s.astype(str).astype("category")
    for key in ["cluster_fine", "res.1", "res1", "cluster", "clusters", "seurat_clusters", "louvain"]:
        if key in adata.obs:
            print(f"  [cluster_fine] using adata.obs['{key}']")
            return adata.obs[key].astype(str).astype("category")
    raise RuntimeError(
        "Cannot resolve cluster_fine (CR2 'res.1' fine clustering). "
        f"No cluster_data.csv and none of the candidate obs keys present.\n"
        f"Available obs columns: {list(adata.obs.columns)}"
    )


def main():
    print("Loading adata_pharynx.h5ad ...")
    adata = sc.read(C.PHARYNX_RAW)
    print(f"  full shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  obsm: {list(adata.obsm.keys())}  layers: {list(adata.layers.keys())}")
    print(f"  X dtype {adata.X.dtype}  max {adata.X.max():.3f}  min {adata.X.min():.3f}")

    # ── obs setup exactly as the CR2 script ───────────────────────────────────
    if {"UMAP1", "UMAP2"}.issubset(adata.obs.columns):
        adata.obsm["X_umap"] = adata.obs[["UMAP1", "UMAP2"]].values
    if "day_str" in adata.obs:
        adata.obs["day"] = adata.obs["day_str"].astype(float)
    elif "day" in adata.obs:
        adata.obs["day"] = adata.obs["day"].astype(float)

    adata.obs["cluster_fine"] = resolve_cluster_fine(adata).values

    # ── subset (PINNED clusters) ──────────────────────────────────────────────
    adata = adata[adata.obs["cluster_fine"].isin(C.SUBSET_CLUSTERS), :].copy()
    print(f"  subset shape: {adata.shape}  "
          f"cluster_name: {adata.obs['cluster_name'].value_counts(dropna=False).to_dict()}")

    # ── preprocessing (PINNED scanpy defaults) ────────────────────────────────
    sc.pp.highly_variable_genes(adata, **C.HVG_KWARGS)
    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"  HVGs selected: {n_hvg}")
    sc.tl.pca(adata, n_comps=C.PCA_N_COMPS)
    sc.pp.neighbors(adata, n_pcs=C.NEIGHBORS_N_PCS, n_neighbors=C.NEIGHBORS_N_NEIGHBORS)

    # ── RealTimeKernel (PINNED) ───────────────────────────────────────────────
    # WOT OTModel needs a NUMERIC day; build/compute transport maps first, THEN
    # convert day to categorical for RealTimeKernel (matches the CR2 script order).
    os.makedirs(C.TMAPS_DIR, exist_ok=True)
    import glob
    if not glob.glob(os.path.join(C.TMAPS_DIR, "tmaps*")):
        print("  computing WOT transport maps ...")
        ot_model = wot.ot.OTModel(adata)
        ot_model.compute_all_transport_maps(tmap_out=os.path.join(C.TMAPS_DIR, "tmaps"))

    adata.obs["day"] = adata.obs["day"].astype("category")
    rtk = RealTimeKernel.from_wot(adata, path=C.TMAPS_DIR, time_key="day")
    # NOTE (version deviation, flagged in report): cellrank 2.0.7's
    # compute_transition_matrix takes only self_transitions + conn_weight. The
    # CR2 script's growth_iters=3 / growth_rate_key="growth_rate_init" belong to
    # an older API; growth is handled at the WOT transport-map stage here, and
    # CR2's precomputed growth_rate_init is not published (absent from figshare
    # and the repo). PINNED self_transitions + conn_weight are applied exactly.
    rtk.compute_transition_matrix(
        self_transitions=C.RTK_SELF_TRANSITIONS, conn_weight=C.RTK_CONN_WEIGHT,
    )
    T = sp.csr_matrix(rtk.transition_matrix)
    print(f"  T shape {T.shape}  nnz {T.nnz}  row-sum range "
          f"[{np.asarray(T.sum(1)).min():.3f}, {np.asarray(T.sum(1)).max():.3f}]")

    # ── GPCCA estimator (PINNED) ──────────────────────────────────────────────
    g = GPCCA(rtk)
    g.compute_schur(n_components=C.SCHUR_N_COMPONENTS)
    try:
        tsi_score = float(g.tsi(n_macrostates=C.TSI_N_MACROSTATES,
                                terminal_states=C.TERMINAL_STATES,
                                cluster_key=C.MACROSTATE_CLUSTER_KEY))
    except Exception as e:
        print(f"  [tsi] skipped: {e}")
        tsi_score = float("nan")

    g.compute_macrostates(n_states=C.N_MACROSTATES, cluster_key=C.MACROSTATE_CLUSTER_KEY)
    macro_purity = get_state_purity(adata, g, "macrostates", C.MACROSTATE_CLUSTER_KEY)
    g.set_terminal_states(C.TERMINAL_STATES)
    term_purity = get_state_purity(adata, g, "terminal_states", C.MACROSTATE_CLUSTER_KEY)

    try:
        g.compute_fate_probabilities(solver="gmres", use_petsc=False)
    except Exception as e:
        print(f"  [fate] gmres failed ({e}); retrying default solver")
        g.compute_fate_probabilities()

    fate = g.fate_probabilities                         # Lineage object
    fate_names = list(fate.names)
    fate_probs = np.asarray(fate.X)                     # (N, n_term)
    print(f"  fate lineages: {fate_names}")

    macro = g.macrostates.astype(str).values            # (N,) categorical -> str

    # ── lineage drivers (PINNED) ──────────────────────────────────────────────
    drivers = g.compute_lineage_drivers(
        return_drivers=True, cluster_key=C.DRIVER_CLUSTER_KEY,
        lineages=[C.DRIVER_LINEAGE], clusters=C.DRIVER_CLUSTERS,
    )
    drivers.to_csv(os.path.join(C.ARTIFACTS, "cr2_drivers.csv"))

    # ── CBC: progenitors -> each terminal (boundaries) ────────────────────────
    # Progenitor cells carry NaN cluster_name; CR2 relabels them "progenitors".
    # Use a relabeled key as a valid CBC source (NaN cannot be a source).
    adata.obs["cluster_cbc"] = (adata.obs[C.CBC_CLUSTER_KEY].astype(str)
                                .replace({"nan": "progenitors"}).astype("category"))
    cats = set(adata.obs["cluster_cbc"].cat.categories)
    boundaries = [("progenitors", t) for t in C.TERMINAL_STATES if t in cats]
    boundaries += [("early_thymus", t) for t in ["cTEC", "mTEC"] if "early_thymus" in cats and t in cats]
    cbc = {}
    for src, tgt in boundaries:
        try:
            cbc[f"{src}->{tgt}"] = np.asarray(
                rtk.cbc(source=src, target=tgt, cluster_key="cluster_cbc", rep=C.CBC_REP))
        except Exception as e:
            print(f"  [cbc] {src}->{tgt} failed: {e}")
    if cbc:
        np.savez(os.path.join(C.ARTIFACTS, "cr2_cbc.npz"), **cbc)

    # ── emit ISOKANN-arm artifacts ────────────────────────────────────────────
    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].to_numpy()
    sp.save_npz(os.path.join(C.ARTIFACTS, "T.npz"), T)
    np.save(os.path.join(C.ARTIFACTS, "features.npy"),
            np.asarray(adata.obsm["X_pca"][:, :C.ISOKANN_N_PCS], dtype=np.float32))
    # PCA loadings: scanpy stores varm["PCs"] over ALL genes; restrict to HVGs used.
    PCs = adata.varm["PCs"][:, :C.ISOKANN_N_PCS]        # (n_genes_all, 50)
    np.save(os.path.join(C.ARTIFACTS, "pca_loadings.npy"),
            PCs[hvg_mask].T.astype(np.float32))         # (50, G)
    Xhvg = adata[:, hvg_mask].X
    Xhvg = Xhvg.toarray() if sp.issparse(Xhvg) else np.asarray(Xhvg)
    np.save(os.path.join(C.ARTIFACTS, "pca_mean.npy"), Xhvg.mean(0).astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "hvg_expr.npy"), Xhvg.astype(np.float32))
    np.save(os.path.join(C.ARTIFACTS, "hvg_genes.npy"), hvg_genes)
    np.save(os.path.join(C.ARTIFACTS, "cell_type.npy"),
            adata.obs[C.MACROSTATE_CLUSTER_KEY].astype(str)
            .replace({"nan": "progenitors"}).to_numpy())
    np.save(os.path.join(C.ARTIFACTS, "cluster_fine.npy"),
            adata.obs["cluster_fine"].astype(str).to_numpy())
    np.save(os.path.join(C.ARTIFACTS, "day.npy"),
            adata.obs["day"].astype(str).to_numpy())
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
        "subset_clusters": C.SUBSET_CLUSTERS,
        "cluster_name_categories": cats,
        "terminal_states": C.TERMINAL_STATES,
        "fate_prob_columns": fate_names,
        "k_macrostates": C.N_MACROSTATES,
        "progenitor_label": prog,
        "cbc_boundaries": [f"{s}->{t}" for s, t in boundaries],
        "tsi": tsi_score,
        "x_max": float(adata.X.max()), "x_min": float(adata.X.min()),
    }
    with open(C.RESOLVED_JSON, "w") as f:
        json.dump(resolved, f, indent=2)

    print("\nDONE. Resolved config:")
    print(json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()
