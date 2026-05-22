import numpy as np, anndata, scipy.sparse as sp
import os
DATA = os.path.join(os.path.dirname(__file__), "data")
adata = anndata.read_h5ad(os.path.join(DATA, "larry_processed.h5ad"))
print("X shape:", adata.X.shape, "type:", type(adata.X))
print("X dtype:", adata.X.dtype)
print("obs:", adata.obs.columns.tolist())
print("var:", adata.var.columns.tolist() if len(adata.var.columns) else "no var columns")
print("var_names[:10]:", adata.var_names[:10].tolist())
print("layers:", list(adata.layers.keys()) if adata.layers else "none")
# Check HVG flag
if "highly_variable" in adata.var.columns:
    hvg_mask = adata.var["highly_variable"].values
    print(f"HVGs flagged: {hvg_mask.sum()}")
else:
    print("No highly_variable column")
# Sample X values
X = adata.X
if sp.issparse(X):
    X_dense = X[:5, :5].toarray()
else:
    X_dense = X[:5, :5]
print("X sample (first 5x5):", X_dense)
print("X min/max:", X.min(), X.max())
# PCA
print("X_pca shape:", adata.obsm["X_pca"].shape)
