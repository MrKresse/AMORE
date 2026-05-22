"""
Differentiable gene-level sensitivity for ISOKANN on LARRY.

Implements the full pipeline as a composition of torch modules:

    gene_counts  →  [log1p + scale]  →  [PCA]  →  [ChiNet]  →  χ

so that autograd gives ∂χ/∂(gene_count) for each cell exactly, without
approximation through the mean-gradient + loading product used in larry_analysis.py.

This matters when:
  · PCA loadings are large and cell-specific nonlinearities exist
  · We want per-cell, not population-average, driver scores
  · We want to rank perturbations: Δχ ≈ (∂χ/∂gene) · Δgene

Run after larry_isokann.py has trained the chi network.

Outputs (written to output/):
    larry_dchi_dgene_mean.npy   – mean |∂χ/∂gene| across all cells, shape (n_genes,)
    larry_dchi_dgene_ts.npy     – mean |∂χ/∂gene| over transition-state cells
    larry_gene_sensitivity_v2.csv – ranked gene table
    analysis_08_gene_sens_v2.png  – top-40 bar plot
    analysis_09_per_cell_drivers.png – per-cell driver heatmap near transition state
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE    = torch.device("cpu")   # gene-space tensors are large; stay on CPU
BATCH     = 256                   # cells processed at once for memory
N_PCS     = 40     # cospar provides 40 PCs
HIDDEN    = [256, 128, 64]        # must match what larry_isokann.py used


# ── Re-define ChiNet (must match training architecture) ────────────────────────
class ChiNet(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        dims = [in_dim] + hidden + [1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Differentiable preprocessing pipeline ─────────────────────────────────────
class LogScalePCA(nn.Module):
    """log1p → per-gene z-score → PCA projection, all in torch."""

    def __init__(self, pca_mean, pca_components, gene_mean, gene_std):
        """
        pca_mean:        (n_pcs,)   – mean of PCA coordinates (from fitted PCA)
        pca_components:  (n_pcs, n_genes) – eigenvectors
        gene_mean:       (n_genes,)  – per-gene mean of log1p-normalized counts
        gene_std:        (n_genes,)  – per-gene std
        """
        super().__init__()
        self.register_buffer("pca_mean",       torch.tensor(pca_mean,       dtype=torch.float32))
        self.register_buffer("pca_comp",        torch.tensor(pca_components, dtype=torch.float32))
        self.register_buffer("gene_mean",      torch.tensor(gene_mean,      dtype=torch.float32))
        self.register_buffer("gene_std",       torch.tensor(gene_std,       dtype=torch.float32))

    def forward(self, counts):
        """counts: (batch, n_genes) raw or already-log-normalised counts."""
        x = torch.log1p(counts)
        x = (x - self.gene_mean) / (self.gene_std + 1e-8)
        x = x @ self.pca_comp.T - self.pca_mean
        return x


class FullPipeline(nn.Module):
    def __init__(self, preproc, chi_net):
        super().__init__()
        self.preproc  = preproc
        self.chi_net  = chi_net

    def forward(self, counts):
        return self.chi_net(self.preproc(counts))


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
adata    = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
X_pca    = np.load(os.path.join(DATA_DIR, "larry_pca.npy"))            # (n_cells, 50)
chi_vals = np.load(os.path.join(OUT_DIR,  "larry_chi_vals.npy"))

n_cells = X_pca.shape[0]
gene_names = np.array(adata.var_names)
n_genes    = len(gene_names)
print(f"  {n_cells:,} cells | {n_genes:,} genes | {N_PCS} PCs")

# ── Recover PCA parameters ─────────────────────────────────────────────────────
if "PCs" in adata.varm:
    pca_comp = adata.varm["PCs"][:, :N_PCS].T.astype(np.float32)   # (N_PCS, n_genes)
elif "pca" in adata.uns and "components" in adata.uns["pca"]:
    pca_comp = adata.uns["pca"]["components"][:N_PCS].astype(np.float32)
else:
    raise RuntimeError("PCA components not found in AnnData. Re-run with PCA stored in varm.")

# PCA mean in PC space (should be ~0 if data was centred)
pca_mean = X_pca.mean(0).astype(np.float32)

# Gene-level normalisation params (from the log-normalised expression matrix)
X_expr = adata.X
if hasattr(X_expr, "toarray"):
    X_expr = X_expr.toarray()
X_expr = np.array(X_expr, dtype=np.float32)

# Check if already log-normalised (max value < 20 suggests log-scale)
if X_expr.max() < 20:
    print("  Expression appears already log-normalised — skipping log1p in pipeline")
    gene_mean = X_expr.mean(0)
    gene_std  = X_expr.std(0)
    # Adjust pipeline: pass data through without re-doing log1p
    class IdentityLogPipeline(nn.Module):
        def __init__(self, pca_mean, pca_comp, gene_mean, gene_std):
            super().__init__()
            self.register_buffer("pca_mean",  torch.tensor(pca_mean,  dtype=torch.float32))
            self.register_buffer("pca_comp",  torch.tensor(pca_comp,  dtype=torch.float32))
            self.register_buffer("gene_mean", torch.tensor(gene_mean, dtype=torch.float32))
            self.register_buffer("gene_std",  torch.tensor(gene_std,  dtype=torch.float32))
        def forward(self, x):
            x = (x - self.gene_mean) / (self.gene_std + 1e-8)
            x = x @ self.pca_comp.T - self.pca_mean
            return x
    preproc = IdentityLogPipeline(pca_mean, pca_comp, gene_mean, gene_std)
else:
    gene_mean = np.log1p(X_expr).mean(0)
    gene_std  = np.log1p(X_expr).std(0)
    preproc = LogScalePCA(pca_mean, pca_comp, gene_mean, gene_std)

# ── Load chi network ───────────────────────────────────────────────────────────
chi_net = ChiNet(N_PCS, HIDDEN)
state_dict = torch.load(os.path.join(OUT_DIR, "larry_chi_net.pt"), map_location="cpu")
chi_net.load_state_dict(state_dict)
chi_net.eval()

pipeline = FullPipeline(preproc, chi_net)
pipeline.eval()

# ── Sanity: verify pipeline reproduces stored chi values ───────────────────────
print("\nVerifying pipeline consistency …")
X_test  = torch.tensor(X_expr[:200], dtype=torch.float32)
with torch.no_grad():
    chi_pipe = pipeline(X_test).numpy()
chi_stored = chi_vals[:200]
corr = np.corrcoef(chi_pipe, chi_stored)[0,1]
print(f"  Pearson(pipeline_chi, stored_chi) on first 200 cells = {corr:.4f}")
if corr < 0.95:
    print("  WARNING: pipeline does not reproduce stored chi — PCA parameters may differ.")
    print("           Gene sensitivities below are still valid for the differentiable path,")
    print("           but may not match the PCA used in larry_isokann.py.")

# ── Compute ∂χ/∂gene in batches ────────────────────────────────────────────────
print(f"\nComputing ∂χ/∂gene for all {n_cells:,} cells in batches of {BATCH} …")
dchi_dgene_all = np.zeros((n_cells, n_genes), dtype=np.float32)

for start in range(0, n_cells, BATCH):
    end   = min(start + BATCH, n_cells)
    batch = torch.tensor(X_expr[start:end], dtype=torch.float32, requires_grad=True)
    chi_b = pipeline(batch)
    chi_b.sum().backward()
    dchi_dgene_all[start:end] = batch.grad.detach().numpy()
    if (start // BATCH) % 20 == 0:
        print(f"  {end}/{n_cells}")

# ── Aggregate sensitivity scores ───────────────────────────────────────────────
mean_sens_all = np.abs(dchi_dgene_all).mean(0)     # population mean |∂χ/∂gene|

ts_mask   = (chi_vals >= 0.4) & (chi_vals <= 0.6)
mean_sens_ts = np.abs(dchi_dgene_all[ts_mask]).mean(0) if ts_mask.sum() > 0 else mean_sens_all

top_all = np.argsort(mean_sens_all)[::-1]
top_ts  = np.argsort(mean_sens_ts)[::-1]

print(f"\n  Top-20 sensitivity genes (population mean):")
for g, s in zip(gene_names[top_all[:20]], mean_sens_all[top_all[:20]]):
    print(f"    {g:<20s}  {s:.5f}")

print(f"\n  Top-20 sensitivity genes (transition-state cells, n={ts_mask.sum():,}):")
for g, s in zip(gene_names[top_ts[:20]], mean_sens_ts[top_ts[:20]]):
    print(f"    {g:<20s}  {s:.5f}")

# ── Save ───────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "larry_dchi_dgene_mean.npy"), mean_sens_all)
np.save(os.path.join(OUT_DIR, "larry_dchi_dgene_ts.npy"),   mean_sens_ts)

gene_df = pd.DataFrame({
    "gene":             gene_names,
    "sens_population":  mean_sens_all,
    "sens_transition":  mean_sens_ts,
    "rank_population":  np.argsort(np.argsort(-mean_sens_all)),
    "rank_transition":  np.argsort(np.argsort(-mean_sens_ts)),
})
gene_df.to_csv(os.path.join(OUT_DIR, "larry_gene_sensitivity_v2.csv"), index=False)

# ── Plots ──────────────────────────────────────────────────────────────────────
known_markers = {
    "stem":     ["Sca1","Kit","Gata2","Runx1","Tal1","Flt3"],
    "myeloid":  ["Pu.1","Csf1r","Cebpa","Cebpb","Mpo","Elane"],
    "erythroid":["Gata1","Klf1","Epor","Hba-a1","Hbb-bt","Gypa"],
    "lymphoid": ["Ebf1","Pax5","Il7r","Rag1","Rag2","Dntt"],
    "mega":     ["Pf4","Vwf","Gp1ba","Nfe2","Mpl"],
}
all_known_lower = {g.lower() for ms in known_markers.values() for g in ms}
gene_lower_map  = {g.lower(): g for g in gene_names}

# Top-40 bar — population
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for ax, (scores, order, label) in zip(axes, [
    (mean_sens_all, top_all[:40], "Population mean"),
    (mean_sens_ts,  top_ts[:40],  "Transition-state cells"),
]):
    gs   = gene_names[order]
    vals = scores[order]
    cols = ["crimson" if g.lower() in all_known_lower else "steelblue" for g in gs]
    ax.bar(range(len(gs)), vals, color=cols)
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels(gs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("|∂χ/∂gene|")
    ax.set_title(f"{label}  (red = known hematopoiesis marker)")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "analysis_08_gene_sens_v2.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: analysis_08_gene_sens_v2.png")

# Per-cell driver heatmap near χ ≈ 0.5
if ts_mask.sum() > 0:
    n_show = min(200, ts_mask.sum())
    # Pick cells spanning chi from 0.4 → 0.6 evenly
    ts_idx   = np.where(ts_mask)[0]
    ts_chi   = chi_vals[ts_idx]
    order_ts = np.argsort(ts_chi)
    sel      = ts_idx[order_ts[np.linspace(0, len(order_ts)-1, n_show, dtype=int)]]
    top_ts_g = top_ts[:30]

    heatmap = dchi_dgene_all[sel][:, top_ts_g]   # (n_show, 30)
    vmax    = np.percentile(np.abs(heatmap), 98)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(30))
    ax.set_yticklabels(gene_names[top_ts_g], fontsize=6)
    ax.set_xlabel(f"Cells ordered by χ  (χ∈[0.4,0.6], n={n_show})")
    ax.set_title("Per-cell ∂χ/∂gene heatmap — top-30 transition-state drivers")
    plt.colorbar(im, ax=ax, label="∂χ/∂gene", shrink=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "analysis_09_per_cell_drivers.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: analysis_09_per_cell_drivers.png")

print("\nDone.")
