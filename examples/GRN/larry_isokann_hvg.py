"""
ISOKANN trained directly on the top-2000 highly-variable genes (no PCA).

Advantages over the PCA-based approach:
  - ∂chi/∂gene is exact (no Ridge regression approximation)
  - The chi gradient directly names genes, not PCA components
  - More general: works for any preprocessed feature set

Architecture: MLP with batch norm at input to handle high-dim expression
    2000 -> BN -> 512 -> Tanh -> 256 -> Tanh -> 128 -> Tanh -> 64 -> Sigmoid

The output is a single chi in (0,1).  This can later be extended to k>1
using ChiNetMulti from amore.isokann.

Run after larry_load.py.  Uses the multi-lag clone pairs from larry_load.py.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scanpy as sc
import pandas as pd
from sklearn.linear_model import Ridge    # only for PCA-baseline comparison

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_HVG          = 2000     # top highly variable genes used as input
N_POWER_ITER   = 60
EPOCHS_PER_ITER = 400
BATCH          = 2048
LR             = 1e-3
LR_DECAY       = 0.97
HIDDEN         = [512, 256, 128, 64]
DROPOUT        = 0.1

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Chi network with batch norm on input ───────────────────────────────────────

class ChiNetHVG(nn.Module):
    """
    Chi network for HVG expression space.
    Batch norm at input handles the high-dim, variable-magnitude expression values.
    """
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden) + [1]
        layers: list[nn.Module] = [nn.BatchNorm1d(in_dim)]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading ...")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
src   = np.load(os.path.join(DATA_DIR, "larry_src.npy"))
dst   = np.load(os.path.join(DATA_DIR, "larry_dst.npy"))

X_expr_full = adata.X
if hasattr(X_expr_full, "toarray"):
    X_expr_full = X_expr_full.toarray()
X_expr_full = np.array(X_expr_full, dtype=np.float32)

n_cells, n_genes_total = X_expr_full.shape
print(f"  {n_cells:,} cells x {n_genes_total:,} genes")
print(f"  {len(src):,} Koopman pairs")


# ── Select HVGs ───────────────────────────────────────────────────────────────

# Use scanpy's HVG selection on the loaded data
# cospar normalises the data; work with what we have
print(f"\nSelecting top {N_HVG} HVGs ...")
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="cell_ranger")
hvg_mask  = adata.var["highly_variable"].values
hvg_names = adata.var_names[hvg_mask].tolist()
X_hvg     = X_expr_full[:, hvg_mask]   # (n_cells, N_HVG)
print(f"  HVG matrix: {X_hvg.shape}")


# ── Standardise ───────────────────────────────────────────────────────────────

mu  = X_hvg[src].mean(0)
sig = X_hvg[src].std(0) + 1e-8

X_hvg_norm = (X_hvg - mu) / sig

x0n = X_hvg_norm[src]
x1n = X_hvg_norm[dst]
Xn  = X_hvg_norm

x0t = torch.tensor(x0n, dtype=torch.float32, device=DEVICE)
x1t = torch.tensor(x1n, dtype=torch.float32, device=DEVICE)
Xt  = torch.tensor(Xn,  dtype=torch.float32, device=DEVICE)

print(f"  Feature dim: {N_HVG}  |  Pairs: {len(x0t):,}  |  All cells: {len(Xt):,}")


# ── Chi network ───────────────────────────────────────────────────────────────

chi   = ChiNetHVG(N_HVG, HIDDEN, DROPOUT).to(DEVICE)
opt   = torch.optim.Adam(chi.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)

n_params = sum(p.numel() for p in chi.parameters())
print(f"\nChiNetHVG: {n_params:,} parameters")

# ── Power iteration ────────────────────────────────────────────────────────────

print(f"\nISokann power iteration ({N_POWER_ITER} x {EPOCHS_PER_ITER} steps)")
print(f"{'iter':>5}  {'loss':>10}  {'chi_min':>8}  {'chi_max':>8}  {'span':>8}")

losses = []
for it in range(N_POWER_ITER):
    chi.eval()
    with torch.no_grad():
        targets = chi(x1t)
        t_min, t_max = targets.min(), targets.max()
        span = (t_max - t_min).item()
        if span < 1e-4:
            targets = torch.rand_like(targets)
        else:
            targets = (targets - t_min) / (t_max - t_min + 1e-8)

    chi.train()
    iter_loss = 0.0
    for ep in range(EPOCHS_PER_ITER):
        idx  = torch.randperm(len(x0t), device=DEVICE)[:BATCH]
        pred = chi(x0t[idx])
        loss = nn.functional.mse_loss(pred, targets[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        iter_loss += loss.item()
    sched.step()
    losses.append(iter_loss / EPOCHS_PER_ITER)

    chi.eval()
    with torch.no_grad():
        chi_sample = chi(x0t[:4096])
    full_span = (chi_sample.max() - chi_sample.min()).item()

    if (it + 1) % 10 == 0 or it == 0:
        print(f"{it+1:>5}  {losses[-1]:>10.5f}  {chi_sample.min():.4f}  "
              f"{chi_sample.max():.4f}  {full_span:.4f}")


# ── Evaluate on all cells ──────────────────────────────────────────────────────

chi.eval()
with torch.no_grad():
    # Process in chunks to avoid OOM
    chi_vals = []
    for i in range(0, len(Xt), 4096):
        chi_vals.append(chi(Xt[i:i+4096]).cpu().numpy())
chi_vals = np.concatenate(chi_vals)
print(f"\nChi range: [{chi_vals.min():.3f}, {chi_vals.max():.3f}]  std={chi_vals.std():.3f}")


# ── Exact gene sensitivity (no regression needed) ─────────────────────────────

print("\nComputing exact dchi/dgene on a subset ...")
n_sens = min(2000, n_cells)
rng    = np.random.default_rng(SEED)
idx_s  = rng.choice(n_cells, n_sens, replace=False)
Xs     = Xt[idx_s].clone().requires_grad_(True)

chi_s  = chi(Xs)
chi_s.sum().backward()
dchi_dhvg = Xs.grad.detach().cpu().numpy()         # (n_sens, N_HVG)
mean_sens = np.abs(dchi_dhvg).mean(0)              # (N_HVG,)

top_idx   = np.argsort(mean_sens)[::-1]
top_genes = np.array(hvg_names)[top_idx[:30]]
top_vals  = mean_sens[top_idx[:30]]

print("\nTop-30 sensitive HVGs (exact dchi/dgene):")
for g, s in zip(top_genes[:30], top_vals[:30]):
    print(f"  {g:<20s}  {s:.5f}")


# ── Compare chi_HVG vs chi_PCA on cell states ─────────────────────────────────

chi_pca = np.load(os.path.join(OUT_DIR, "larry_chi_vals.npy"))
states  = adata.obs["state_info"].astype(str).values
cats    = sorted(set(states), key=lambda s: np.median(chi_vals[states==s]))

print("\nMedian chi (HVG-trained) per cell state:")
for s in cats:
    m = states == s
    print(f"  {s:<20s}  chi_HVG={np.median(chi_vals[m]):.3f}  "
          f"chi_PCA={np.median(chi_pca[m]):.3f}")


# ── Save ──────────────────────────────────────────────────────────────────────

np.save(os.path.join(OUT_DIR, "larry_chi_hvg.npy"), chi_vals)
np.save(os.path.join(OUT_DIR, "larry_dchi_dhvg.npy"), dchi_dhvg)
torch.save(chi.state_dict(), os.path.join(OUT_DIR, "larry_chi_hvg_net.pt"))

gene_df = pd.DataFrame({
    "gene":            np.array(hvg_names),
    "sensitivity":     mean_sens,
    "rank":            np.argsort(np.argsort(-mean_sens)),
})
gene_df.to_csv(os.path.join(OUT_DIR, "larry_hvg_sensitivity.csv"), index=False)


# ── Plots ─────────────────────────────────────────────────────────────────────

emb = adata.obsm["X_umap"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Chi-HVG on embedding
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
norm = Normalize(0, 1)
ax = axes[0]
ax.scatter(emb[:,0], emb[:,1], c=chi_vals, cmap="coolwarm", norm=norm,
           s=1, alpha=0.4, rasterized=True)
plt.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, label="chi (HVG)", shrink=0.8)
ax.set_title("Chi (HVG-trained)"); ax.set_xticks([]); ax.set_yticks([])

# Comparison: chi_HVG vs chi_PCA
ax = axes[1]
ax.scatter(chi_pca[::5], chi_vals[::5], s=1, alpha=0.3, rasterized=True)
r, _ = spearmanr(chi_pca, chi_vals)
ax.set_xlabel("chi (PCA-trained)"); ax.set_ylabel("chi (HVG-trained)")
ax.set_title(f"HVG vs PCA chi  Spearman={r:.3f}")

# Training convergence
ax = axes[2]
ax.plot(losses)
ax.set_xlabel("Power iteration"); ax.set_ylabel("MSE loss")
ax.set_title("HVG training convergence")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "larry_chi_hvg.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: larry_chi_hvg.png")

# Sensitivity bar chart
fig, ax = plt.subplots(figsize=(14, 4))
from larry_metrics import MOUSE_TFS_LOWER
is_tf = [g.lower() in MOUSE_TFS_LOWER for g in top_genes[:50]]
colors = ["crimson" if tf else "steelblue" for tf in is_tf]
top50_vals = mean_sens[top_idx[:50]]
ax.bar(range(50), top50_vals, color=colors)
ax.set_xticks(range(50))
ax.set_xticklabels(np.array(hvg_names)[top_idx[:50]], rotation=45, ha="right", fontsize=6)
ax.set_ylabel("|dchi/dgene| (exact)")
ax.set_title("Top-50 HVG sensitivity  (red = known TF)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "larry_hvg_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: larry_hvg_sensitivity.png")

print(f"\nAll outputs -> {OUT_DIR}/")
print("Next: run larry_metrics.py for TF enrichment on HVG sensitivity")
