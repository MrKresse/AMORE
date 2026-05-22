"""
Multi-dimensional ISOKANN on LARRY hematopoiesis.

Strategy
--------
Train with k_max chi functions (overparameterised), then determine the
correct number of metastable states from the spectral gap in the Koopman
eigenvalue spectrum — the same criterion PCCA+ uses for Markov state model
selection.

For a k-state system the Koopman operator has exactly k eigenvalues close to 1
(slow processes) and all others near 0.  The spectral gap

    gap_i = |lambda_i| - |lambda_{i+1}|

is maximised at i = k_correct.

Workflow
--------
1. Load PCA features + multi-lag clone pairs (from larry_load.py)
2. Train ChiNetMultiRaw (k_max sigmoid outputs) via SVD power iteration
3. Evaluate chi on all cells, compute K = A^{-1}C, extract eigenvalues
4. Plot timescale spectrum, identify spectral gap -> k_correct
5. Assign each cell to its dominant chi function, compare to cell-state labels
6. Compare chi simplex to cospar's X_emb (the 2-D embedding)

Architecture
------------
ChiNetMultiRaw: 40 -> 512 -> 256 -> 128 -> k_max  (sigmoid, no softmax)
SVD power iteration (as in amore.isokann.power_method_multi)
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
import scanpy as sc

# amore imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from amore.isokann import ChiNetMultiRaw, power_method_multi, implied_timescales, koopman_matrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
K_MAX          = 15      # 15 gives clean convergence; all |lambda| < 1
K_OVERRIDE     = 13      # user-identified spectral gap; None = auto-detect
N_PCS          = 40      # cospar PCA dimensionality
HIDDEN         = [512, 256, 128]

N_POWER_ITER   = 80
EPOCHS_PER_ITER = 400
BATCH          = 4096
LR             = 2e-3
LR_DECAY       = 0.97

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data ...")
adata    = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
X_pca    = np.load(os.path.join(DATA_DIR, "larry_pca.npy")).astype(np.float32)
x0_raw   = np.load(os.path.join(DATA_DIR, "larry_x0.npy")).astype(np.float32)
x1_raw   = np.load(os.path.join(DATA_DIR, "larry_x1.npy")).astype(np.float32)

states   = adata.obs["state_info"].astype(str).values
emb      = adata.obsm["X_umap"]
n_cells  = len(X_pca)

print(f"  {n_cells:,} cells | {N_PCS} PCs | {len(x0_raw):,} pairs | k_max={K_MAX}")

# Standardise
mu  = x0_raw.mean(0, keepdims=True)
sig = x0_raw.std(0,  keepdims=True) + 1e-8
x0n = (x0_raw  - mu) / sig
x1n = (x1_raw  - mu) / sig
Xn  = (X_pca   - mu) / sig

x0t = torch.tensor(x0n, dtype=torch.float32, device=DEVICE)
x1t = torch.tensor(x1n, dtype=torch.float32, device=DEVICE)
Xt  = torch.tensor(Xn,  dtype=torch.float32, device=DEVICE)


# ── Network ────────────────────────────────────────────────────────────────────
chi = ChiNetMultiRaw(in_dim=N_PCS, k=K_MAX, hidden=HIDDEN).to(DEVICE)
n_params = sum(p.numel() for p in chi.parameters())
print(f"\nChiNetMultiRaw: {n_params:,} params  k_max={K_MAX}")


# ── Multi-D power iteration ────────────────────────────────────────────────────
result = power_method_multi(
    chi, x0t, x1t,
    n_iter          = N_POWER_ITER,
    epochs_per_iter = EPOCHS_PER_ITER,
    lr              = LR,
    lr_decay        = LR_DECAY,
    batch           = BATCH,
    verbose         = True,
)


# ── Koopman eigenvalue spectrum ────────────────────────────────────────────────
print("\n-- Eigenvalue spectrum --")
chi.eval()
with torch.no_grad():
    chi_x0 = chi(x0t)   # (n_pairs, k_max)
    chi_x1 = chi(x1t)

evals_raw, timescales = implied_timescales(chi_x0, chi_x1, lagtime=1.0)
abs_evals = np.sort(np.abs(evals_raw))[::-1]     # sorted descending

# Spectral gap detection
gaps  = abs_evals[:-1] - abs_evals[1:]
k_auto = int(np.argmax(gaps)) + 1   # largest single drop

if K_OVERRIDE is not None:
    k_gap = K_OVERRIDE
    print(f"  k_correct = {k_gap}  (manually set; auto-detect would give k={k_auto})")
else:
    k_gap = k_auto

print(f"  Eigenvalues (sorted): {abs_evals.round(4)}")
print(f"  Gaps:                 {gaps.round(4)}")
print(f"  Spectral gap at k={k_gap}  (gap size={gaps[k_gap-1]:.4f})")
print(f"  Implied timescales:   {timescales.round(2)}")

np.save(os.path.join(OUT_DIR, "multi_eigenvalues.npy"), abs_evals)
np.save(os.path.join(OUT_DIR, "multi_timescales.npy"),  timescales)


# ── Evaluate chi on all cells ──────────────────────────────────────────────────
chi.eval()
with torch.no_grad():
    chi_all_list = []
    for i in range(0, len(Xt), 4096):
        chi_all_list.append(chi(Xt[i:i+4096]).cpu().numpy())
chi_all = np.concatenate(chi_all_list)           # (n_cells, k_max)

np.save(os.path.join(OUT_DIR, "multi_chi_all.npy"), chi_all)
torch.save(chi.state_dict(), os.path.join(OUT_DIR, "multi_chi_net.pt"))


# ── Plots ──────────────────────────────────────────────────────────────────────

# 1. Eigenvalue spectrum with gap highlighted
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.bar(range(1, K_MAX+1), abs_evals, color="steelblue", alpha=0.8)
ax.axvline(k_gap + 0.5, color="crimson", lw=2, ls="--",
           label=f"Spectral gap at k={k_gap}")
ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("|lambda|")
ax.set_title("Koopman eigenvalue spectrum")
ax.legend(); ax.set_ylim(0, 1.05)

ax = axes[1]
ax.bar(range(1, len(timescales)+1), np.clip(timescales, 0, 50), color="steelblue", alpha=0.8)
ax.axvline(k_gap - 0.5, color="crimson", lw=2, ls="--",
           label=f"k_correct={k_gap}")
ax.set_xlabel("Mode index"); ax.set_ylabel("Implied timescale (x lag)")
ax.set_title("Implied timescales")
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_spectrum.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: multi_spectrum.png")


# 2. Each chi function on the embedding (use the k_gap most important ones)
k_show = min(k_gap, 12)
ncols  = 4
nrows  = (k_show + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
axes = np.array(axes).flatten()

# Sort chi functions by eigenvalue magnitude
order = np.argsort(-abs_evals[:K_MAX])   # highest eigenvalue first

for plot_i, chi_i in enumerate(order[:k_show]):
    ax = axes[plot_i]
    vals = chi_all[:, chi_i]
    vmin, vmax = np.percentile(vals, [2, 98])
    sc_ = ax.scatter(emb[:,0], emb[:,1], c=vals, cmap="coolwarm",
                     vmin=vmin, vmax=vmax, s=1, alpha=0.4, rasterized=True)
    plt.colorbar(sc_, ax=ax, shrink=0.7)
    lam = abs_evals[chi_i] if chi_i < len(abs_evals) else 0
    ax.set_title(f"chi_{chi_i+1}  |lam|={lam:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

for j in range(k_show, len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f"Top-{k_show} chi functions on UMAP (k_max={K_MAX}, spectral gap at k={k_gap})",
             fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_chi_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_chi_umap.png")


# 3. Cell-state assignment from argmax chi vs known state labels
argmax_chi = np.argmax(chi_all[:, order[:k_gap]], axis=1)   # (n_cells,)

# What cell state is most common in each chi cluster?
print(f"\n-- Chi cluster -> cell state mapping (k={k_gap}) --")
state_cats = sorted(set(states))
assignment  = {}
for k in range(k_gap):
    mask = argmax_chi == k
    if mask.sum() == 0:
        continue
    counts   = {s: (states[mask] == s).sum() for s in state_cats}
    top3     = sorted(counts.items(), key=lambda x: -x[1])[:3]
    dominant = top3[0][0]
    assignment[k] = dominant
    top3_str = ", ".join(f"{s}({c})" for s, c in top3 if c > 0)
    print(f"  chi cluster {k:>2d}  n={mask.sum():>5,}  top states: {top3_str}")

# Adjusted Rand Index: how well does chi argmax recover cell-state labels?
# Map state labels to integers
state_to_int = {s: i for i, s in enumerate(state_cats)}
states_int   = np.array([state_to_int[s] for s in states])
ari = adjusted_rand_score(states_int, argmax_chi)
print(f"\n  Adjusted Rand Index (chi argmax vs cell state): {ari:.4f}")
print(f"  (0=random, 1=perfect; ARI>0.3 is considered good)")


# 4. UMAP coloured by argmax chi cluster
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
cmap_cl = plt.get_cmap("tab20", k_gap)
for k in range(k_gap):
    mask = argmax_chi == k
    label = f"chi_{order[k]+1} ({assignment.get(k,'?')})"
    ax.scatter(emb[mask,0], emb[mask,1], color=cmap_cl(k), s=1,
               alpha=0.5, label=label, rasterized=True)
ax.legend(fontsize=5, markerscale=5, ncol=2, loc="upper right")
ax.set_title(f"Chi argmax clusters (k={k_gap})"); ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
state_cmap = plt.get_cmap("tab20", len(state_cats))
for i, s in enumerate(state_cats):
    mask = states == s
    ax.scatter(emb[mask,0], emb[mask,1], color=state_cmap(i), s=1,
               alpha=0.5, label=s, rasterized=True)
ax.legend(fontsize=5, markerscale=5, ncol=2, loc="upper right")
ax.set_title("Known cell states"); ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(f"Multi-D ISOKANN k={k_gap} vs known cell states  ARI={ari:.3f}", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_state_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_state_comparison.png")


# 5. Convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(result["losses"])
axes[0].set_xlabel("Power iteration"); axes[0].set_ylabel("Avg MSE loss")
axes[0].set_title("Training convergence")

spans = result["spans"]   # (n_iter, k_max)
for i in range(K_MAX):
    axes[1].plot(spans[:, i], alpha=0.5, lw=1)
axes[1].set_xlabel("Power iteration"); axes[1].set_ylabel("chi span")
axes[1].set_title(f"Chi span evolution (all {K_MAX} functions)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_convergence.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_convergence.png")

print(f"\n-- Summary --")
print(f"  k_max={K_MAX}  spectral gap at k={k_gap}")
print(f"  Eigenvalues (top-{k_gap}): {abs_evals[:k_gap].round(4)}")
print(f"  ARI vs cell states: {ari:.4f}")
print(f"\nAll outputs -> {OUT_DIR}/")
