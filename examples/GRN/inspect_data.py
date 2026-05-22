import numpy as np, pandas as pd, os

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")

obs = pd.read_csv(os.path.join(DATA, "larry_obs.csv"), index_col=0)
print("obs cols:", obs.columns.tolist())
print("obs shape:", obs.shape)
print(obs.head(4).to_string())
print("\ntime values:", sorted(obs["time_info"].unique().tolist()))
print("\nstate_info value_counts:\n", obs["state_info"].value_counts().to_string())

# Clone column?
for col in obs.columns:
    print(f"\ncol '{col}' dtype={obs[col].dtype} nunique={obs[col].nunique()} sample={obs[col].iloc[:3].tolist()}")

chi = np.load(os.path.join(OUT, "multi_chi_all.npy"))
print("\nchi shape:", chi.shape, "dtype:", chi.dtype)
print("chi min/max:", chi.min(), chi.max())

ev = np.load(os.path.join(OUT, "multi_eigenvalues.npy"))
print("\neigenvalues:", ev)

src = np.load(os.path.join(DATA, "larry_src.npy"))
dst = np.load(os.path.join(DATA, "larry_dst.npy"))
print("\nsrc shape:", src.shape, "dst shape:", dst.shape)
print("src range:", src.min(), "-", src.max())
print("dst range:", dst.min(), "-", dst.max())

# Check h5ad for clone info
import anndata
adata = anndata.read_h5ad(os.path.join(DATA, "larry_processed.h5ad"))
print("\nadata obs cols:", adata.obs.columns.tolist())
print("adata obsm keys:", list(adata.obsm.keys()))
print("\nadata.obs head:\n", adata.obs.head(3).to_string())
# check X_clone
if "X_clone" in adata.obsm:
    Xc = adata.obsm["X_clone"]
    print("\nX_clone shape:", Xc.shape)
    print("X_clone type:", type(Xc))
