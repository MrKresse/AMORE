"""
amore.scrna — single-cell RNA-seq extension of ISOKANN+AMORE.

Train k-D chi memberships on single-cell features using a CellRank-style transition
matrix as the Koopman conditional expectation (kchi = T @ chi), attribute genes to
each membership (gradient / Integrated Gradients / correlation), and plot
(chi-driven UMAP, attribution heatmaps, TF x cell-line drivers).

Quick start
-----------
>>> from amore.scrna import train_chi, binned_gradient_sensitivity, run_chi_umap
>>> res = train_chi(X_hvg, T, k=4)                     # T = CellRank kernel
>>> emb = run_chi_umap(res["chi"])                     # chi-driven UMAP
>>> sens = binned_gradient_sensitivity(res["net"], X_hvg, res["chi"][:, 1], mode=1)
"""
from .koopman import (
    train_chi, koopman_expectation, split_operator, to_sparse_torch, init_from_warmup,
)
from .attribution import (
    gene_gradient, binned_gradient_sensitivity, signed_corr, rank_genes, recovery_at_k,
)
from .plotting import (
    run_chi_umap, scatter_categorical, scatter_chi, plot_chi_umaps,
    driver_heatmap, tf_lineage_heatmap, expression_heatmap, expression_profiles,
    plot_loss,
)
from .mfep import (
    transition_state_medoid, medoid_path, gradient_path,
    project_to_chi_umap, project_to_embedding, draw_path, smooth_2d, ModeNet,
)

__all__ = [
    "train_chi", "koopman_expectation", "split_operator", "to_sparse_torch",
    "init_from_warmup",
    "gene_gradient", "binned_gradient_sensitivity", "signed_corr", "rank_genes",
    "recovery_at_k",
    "run_chi_umap", "scatter_categorical", "scatter_chi", "plot_chi_umaps",
    "driver_heatmap", "tf_lineage_heatmap", "expression_heatmap",
    "expression_profiles", "plot_loss",
    "transition_state_medoid", "medoid_path", "gradient_path",
    "project_to_chi_umap", "project_to_embedding", "draw_path", "smooth_2d",
]
