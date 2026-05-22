"""
ISOKANN on LARRY hematopoiesis data.

The Koopman power iteration learns a continuous committor-like function
χ: R^{50} → [0,1] on PCA space.  Clone-matched pairs (ancestor at day T_EARLY,
descendant at day T_LATE) play the role of short-time Koopman pairs from MD.

Connection to AMORE/MD examples
---------------------------------
    MD:           x_0 ──[Langevin, τ]──→ x_τ        (one trajectory burst)
    scRNA:        x_0 ──[differentiation]──→ x_τ     (cells in same clone, ΔT days apart)

Everything downstream (power iteration, chi network, MEP on level sets) is identical.
The learned χ:
  · generalises to unseen cells (inductive, unlike graph-based methods)
  · has analytic gradient ∂χ/∂x — identifies fate-driving genes in PCA-loadings space
  · its level sets are input to amore.mep for minimum-χ-energy paths between phenotypes

Run larry_load.py first to generate ./data/larry_*.npy and larry_processed.h5ad.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_POWER_ITER   = 80     # outer Koopman power iterations
EPOCHS_PER_ITER = 400   # gradient steps per iteration
BATCH          = 4096
LR             = 2e-3
LR_DECAY       = 0.97   # multiplicative LR decay per power iteration
HIDDEN         = [256, 128, 64]  # chi network hidden layer widths
N_PCA          = 40     # input dimensionality (cospar provides 40 PCs)

SEED           = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Load data ──────────────────────────────────────────────────────────────────

X_all = np.load(os.path.join(DATA_DIR, "larry_pca.npy")).astype(np.float32)
x0    = np.load(os.path.join(DATA_DIR, "larry_x0.npy")).astype(np.float32)
x1    = np.load(os.path.join(DATA_DIR, "larry_x1.npy")).astype(np.float32)

print(f"All cells: {len(X_all):,} | Koopman pairs: {len(x0):,} | Feature dim: {N_PCA}")

# Standardise using pair statistics so the network sees O(1) inputs
mu  = x0.mean(0, keepdims=True)
sig = x0.std(0,  keepdims=True) + 1e-8

x0n = (x0    - mu) / sig
x1n = (x1    - mu) / sig
Xn  = (X_all - mu) / sig

x0t = torch.tensor(x0n, device=DEVICE)
x1t = torch.tensor(x1n, device=DEVICE)
Xt  = torch.tensor(Xn,  device=DEVICE)


# ── Chi network ────────────────────────────────────────────────────────────────
# Architecture mirrors amore/sims/openmm_sim.py pairnet_nodes but takes PCA input.
# Output is Sigmoid → χ ∈ (0,1).

class ChiNet(nn.Module):
    """Fully-connected network mapping PCA coordinates to χ ∈ (0,1)."""

    def __init__(self, in_dim: int, hidden: list[int]) -> None:
        super().__init__()
        dims = [in_dim] + hidden + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


chi = ChiNet(N_PCA, HIDDEN).to(DEVICE)
opt = torch.optim.Adam(chi.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)

n_params = sum(p.numel() for p in chi.parameters())
print(f"Chi network: {n_params:,} parameters")


# ── ISOKANN power iteration ────────────────────────────────────────────────────
#
# Each outer iteration is one step of the Koopman power method:
#
#   χ_{n+1}(x_0) ≈ argmin_f  Σ_i  [ f(x0_i) - χ_n(x1_i) ]²
#
# Targets χ_n(x1_i) are frozen (no_grad) before each inner training loop.
# This is identical to the 'power_method' in MoKiTo but implemented explicitly
# for the single-cell feature space.
#
# Collapse guard: if the learned function saturates to near-constant (span < ε),
# the targets are re-seeded with small noise to escape the fixed point χ ≡ const.

losses_per_iter: list[float] = []
span_per_iter:   list[float] = []

print(f"\nISokann power iteration  ({N_POWER_ITER} × {EPOCHS_PER_ITER} steps)\n"
      f"{'iter':>5}  {'loss':>10}  {'chi_min':>8}  {'chi_max':>8}  {'span':>8}")

for it in range(N_POWER_ITER):

    # Freeze current network → compute regression targets
    with torch.no_grad():
        targets = chi(x1t)

    t_min, t_max = targets.min().item(), targets.max().item()
    span = t_max - t_min

    if span < 1e-4:
        # Trivial fixed point χ ≡ const — restart targets with slight noise
        targets = targets + 0.05 * torch.randn_like(targets)
        t_min, t_max = targets.min().item(), targets.max().item()
        span = t_max - t_min

    # Normalise targets to [0,1] to keep loss scale stable across iterations
    targets = (targets - t_min) / (span + 1e-8)

    # Inner optimisation: fit χ_new(x0) → targets
    chi.train()
    iter_loss = 0.0
    for ep in range(EPOCHS_PER_ITER):
        idx  = torch.randperm(len(x0t), device=DEVICE)[:BATCH]
        pred = chi(x0t[idx])
        loss = nn.functional.mse_loss(pred, targets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        iter_loss += loss.item()

    scheduler.step()

    # Evaluate span on the full cell set
    chi.eval()
    with torch.no_grad():
        chi_all = chi(Xt).cpu().numpy()

    avg_loss = iter_loss / EPOCHS_PER_ITER
    full_span = chi_all.max() - chi_all.min()
    losses_per_iter.append(avg_loss)
    span_per_iter.append(full_span)

    if (it + 1) % 5 == 0 or it == 0:
        print(f"{it+1:>5}  {avg_loss:>10.5f}  {chi_all.min():>8.4f}  "
              f"{chi_all.max():>8.4f}  {full_span:>8.4f}")


# ── Diagnostics: chi gradient w.r.t. PCA coordinates ────────────────────────
# ∂χ/∂x_j evaluated at each cell and projected back to gene space via PCA loadings
# gives a first-pass driver-gene score (cf. chi_sensitivity in amore/chi.py).

chi.eval()
Xt_grad = Xt.clone().requires_grad_(True)
chi_vals_grad = chi(Xt_grad)
chi_vals_grad.sum().backward()
dchi_dx = Xt_grad.grad.detach().cpu().numpy()          # (n_cells, 50)
sensitivity = (dchi_dx ** 2).mean(0)                   # mean-squared gradient per PC

top_pcs = np.argsort(sensitivity)[::-1][:10]
print(f"\nTop-10 PCs by ∂χ/∂PC sensitivity (index, score):")
for pc in top_pcs:
    print(f"  PC{pc+1:02d}  {sensitivity[pc]:.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────

chi.eval()
with torch.no_grad():
    chi_vals = chi(Xt).cpu().numpy()

np.save(os.path.join(OUT_DIR, "larry_chi_vals.npy"), chi_vals)
np.save(os.path.join(OUT_DIR, "larry_dchi_dx.npy"),  dchi_dx)
torch.save(chi.state_dict(), os.path.join(OUT_DIR, "larry_chi_net.pt"))
print(f"\nSaved chi values, gradients and network weights to {OUT_DIR}/")


# ── Plots ─────────────────────────────────────────────────────────────────────

norm = Normalize(vmin=0, vmax=1)
cmap = "coolwarm"

# 1. Convergence curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses_per_iter)
axes[0].set_xlabel("Power iteration")
axes[0].set_ylabel("Average MSE loss")
axes[0].set_title("ISOKANN convergence")

axes[1].plot(span_per_iter)
axes[1].set_xlabel("Power iteration")
axes[1].set_ylabel("χ range (max − min)")
axes[1].set_title("Chi function spread")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "larry_convergence.png"), dpi=150)
plt.close("all")

# 2. χ on PCA embedding
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sc_ = ax.scatter(X_all[:, 0], X_all[:, 1], c=chi_vals, cmap=cmap, norm=norm,
                 s=1, alpha=0.4, rasterized=True)
plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="χ")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("ISOKANN χ — PCA projection")

# 3. χ on UMAP (requires AnnData)
ax = axes[1]
try:
    import scanpy as sc
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
    umap = adata.obsm["X_umap"]
    sc_ = ax.scatter(umap[:, 0], umap[:, 1], c=chi_vals, cmap=cmap, norm=norm,
                     s=1, alpha=0.4, rasterized=True)
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="χ")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("ISOKANN χ — UMAP")
except Exception as e:
    ax.text(0.5, 0.5, f"UMAP unavailable\n({e})", transform=ax.transAxes,
            ha="center", va="center")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "larry_chi.png"), dpi=150)
plt.close("all")

# 4. Chi distribution histogram
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(chi_vals, bins=60, color="steelblue", edgecolor="none")
ax.set_xlabel("χ")
ax.set_ylabel("Cell count")
ax.set_title("Chi distribution over all cells")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "larry_chi_hist.png"), dpi=150)
plt.close("all")

print(f"\nAll plots saved to {OUT_DIR}/")
print("\nNext steps:")
print("  · Compare χ values to clonal fate-bias labels (adata.obs) to validate polarity")
print("  · Project ∂χ/∂x through PCA loadings to rank fate-driving genes")
print("  · Use amore.mep on transition-state cells (χ ≈ 0.5) for minimum-χ-energy paths")
