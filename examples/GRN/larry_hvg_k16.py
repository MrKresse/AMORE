"""
LARRY: SVD-Power with k=16, HVG features, adaptive stopping.

Design choices
--------------
  Features : top-2000 HVGs, z-score standardised on training pairs
  Network  : BN_input → [1024, 512, 256, 128] → ReLU hidden → sigmoid output
             (ReLU avoids gradient vanishing in deep layers;
              sigmoid only at output for (0,1) constraint)
  k        : 16 (overparameterised — effective k read from SD elbow)
  Stopping : run until convergence (loss plateau < rel_tol over a window),
             max 200 outer iterations
  Koopman  : SVD deflation power iteration (same as power_method_multi)

Diagnostics
-----------
  - Per-iteration loss curve
  - Per-mode SD profile (collapse detector)
  - Eigenvalue spectrum (k_eff from SD elbow)
  - Spearman r vs NeuMon fate bias (Tier-2 hard split)
  - AUC-ROC per fate at k_eff
  - UMAP of chi[:, :k_eff] coloured by cell state

Outputs
-------
  output/larry_hvg_chi_k16.npy    — (N_cells, 16)
  output/larry_hvg_losses.npy     — (n_iters,)
  output/larry_hvg_spans.npy      — (n_iters, 16)
  output/larry_hvg_eigenvalues.npy
  output/larry_hvg_chi_umap.npy
  output/larry_hvg_loss_curve.png
  output/larry_hvg_sd_profile.png
  output/larry_hvg_umap.png
  output/larry_hvg_neumon_scatter.png
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_HVG          = 2000
K              = 16
HIDDEN         = [1024, 512, 256, 128]
MAX_ITERS      = 200
EPOCHS_PER_ITER = 400
BATCH          = 2048
LR             = 2e-3
LR_DECAY       = 0.97
COLLAPSE_EPS   = 1e-3

# Convergence: stop when median loss change over WINDOW iters < REL_TOL
WINDOW  = 10
REL_TOL = 5e-4

SEED  = 42
DEVICE = torch.device("cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

FATES     = ["Mast","Baso","Meg","Erythroid","Lymphoid",
             "Neutrophil","Monocyte","Eos","pDC","Ccr7_DC"]
FATE_COLS = [f"progenitor_{f}" for f in FATES]


# ══════════════════════════════════════════════════════════════════════════════
# Network: BN input, ReLU hidden, sigmoid output
# ══════════════════════════════════════════════════════════════════════════════

class ChiNetHVGMulti(nn.Module):
    """
    Deep chi network for HVG expression space.
    BatchNorm at input: normalises across the high-magnitude, variable-scale
    expression values without losing gene-specific signal.
    Tanh hidden: matches ChiNetMultiRaw (the working PCA architecture);
    smooth activations suit smooth Koopman eigenfunctions better than ReLU.
    Sigmoid output: enforces chi ∈ (0,1)^k for SVD-Power compatibility.
    """
    def __init__(self, in_dim: int, k: int, hidden: list[int]) -> None:
        super().__init__()
        dims   = [in_dim] + list(hidden) + [k]
        layers: list[nn.Module] = [nn.BatchNorm1d(in_dim)]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading …")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
src   = np.load(os.path.join(DATA_DIR, "larry_src.npy"))
dst   = np.load(os.path.join(DATA_DIR, "larry_dst.npy"))
obs   = adata.obs.copy()

X_expr = adata.X
if hasattr(X_expr, "toarray"):
    X_expr = X_expr.toarray()
X_expr = X_expr.astype(np.float32)
print(f"  {X_expr.shape[0]:,} cells × {X_expr.shape[1]:,} genes")
print(f"  {len(src):,} Koopman pairs")

# ── HVG selection ──────────────────────────────────────────────────────────────
print(f"\nSelecting top {N_HVG} HVGs …")
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="cell_ranger")
hvg_mask  = adata.var["highly_variable"].values
X_hvg     = X_expr[:, hvg_mask].astype(np.float32)   # (N_cells, N_HVG)
print(f"  HVG matrix: {X_hvg.shape}")

# Standardise on training-pair distribution
mu  = X_hvg[src].mean(0)
sig = X_hvg[src].std(0) + 1e-8
X_hvg_norm = (X_hvg - mu) / sig

x0_t = torch.tensor(X_hvg_norm[src], dtype=torch.float32, device=DEVICE)
x1_t = torch.tensor(X_hvg_norm[dst], dtype=torch.float32, device=DEVICE)
x_all_t = torch.tensor(X_hvg_norm,   dtype=torch.float32, device=DEVICE)

# Fate / NeuMon labels
time_info  = obs["time_info"].astype(str).values
day2_mask  = time_info == "2"
nm_bias    = obs["NeuMon_fate_bias"].values.astype(float)
nm_mask    = obs["NeuMon_mask"].values.astype(bool) & day2_mask
fate_labels = {f: obs[c].values.astype(float)
               for f, c in zip(FATES, FATE_COLS) if c in obs.columns}
state_info = obs["state_info"].values.astype(str)
X_umap_orig = adata.obsm["X_umap"].astype(np.float32)

print(f"  Day-2 cells: {day2_mask.sum()},  NeuMon subset: {nm_mask.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# Training with adaptive stopping
# ══════════════════════════════════════════════════════════════════════════════
def scale_to_unit(Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (Y - Y.min(0).values) / (Y.max(0).values - Y.min(0).values + eps)


net = ChiNetHVGMulti(in_dim=N_HVG, k=K, hidden=HIDDEN).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)

print(f"\nTraining SVD-Power  k={K}  max_iters={MAX_ITERS}  "
      f"epochs/iter={EPOCHS_PER_ITER}")
print(f"Architecture: BN → {HIDDEN} → k={K}  (ReLU hidden, sigmoid output)")
print(f"Convergence: stop if loss plateau < {REL_TOL} over {WINDOW} iters\n")
print(f"{'iter':>5}  {'loss':>8}  {'Δloss':>8}  {'k_live':>6}  "
      f"{'SD_max':>7}  {'SD_min':>7}")

losses_all : list[float] = []
spans_all  : list[np.ndarray] = []
converged  = False

for it in range(MAX_ITERS):

    # ── 1. Koopman action ─────────────────────────────────────────────────────
    net.eval()
    with torch.no_grad():
        Y = net(x1_t)                      # (n, K)

    # ── 2. SVD orthogonalisation ──────────────────────────────────────────────
    Yc = Y - Y.mean(0)
    try:
        U, S, _ = torch.linalg.svd(Yc, full_matrices=False)   # U: (n, K)
    except torch.linalg.LinAlgError:
        U = Yc

    # Collapse guard
    col_std = U.std(0)
    for j in range(K):
        if col_std[j].item() < COLLAPSE_EPS:
            U = U.clone()
            U[:, j] += COLLAPSE_EPS * torch.randn(len(U), device=DEVICE)

    # ── 3. Scale to [0,1] ─────────────────────────────────────────────────────
    targets = scale_to_unit(U)

    # ── 4. Inner SGD ──────────────────────────────────────────────────────────
    net.train()
    n, total = len(x0_t), 0.0
    for _ in range(EPOCHS_PER_ITER):
        idx  = torch.randperm(n, device=DEVICE)[:BATCH]
        loss = F.mse_loss(net(x0_t[idx]), targets[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    avg_loss = total / EPOCHS_PER_ITER
    sch.step()

    # ── Diagnostics ───────────────────────────────────────────────────────────
    net.eval()
    with torch.no_grad():
        chi_cur = net(x0_t)
    spans = (chi_cur.max(0).values - chi_cur.min(0).values).cpu().numpy()
    losses_all.append(avg_loss)
    spans_all.append(spans)

    k_live = int((spans > 0.05).sum())
    dloss  = (losses_all[-1] - losses_all[-2]) / (losses_all[-2] + 1e-9) if it > 0 else 0.0

    if (it + 1) % 5 == 0 or it == 0:
        print(f"{it+1:>5}  {avg_loss:>8.5f}  {dloss:>+8.5f}  "
              f"{k_live:>6}  {spans.max():>7.4f}  {spans.min():>7.4f}")

    # ── Convergence check ─────────────────────────────────────────────────────
    if it >= WINDOW:
        recent = losses_all[-WINDOW:]
        rel_change = abs(recent[0] - recent[-1]) / (recent[0] + 1e-9)
        if rel_change < REL_TOL:
            print(f"\n  Converged at iter {it+1}  "
                  f"(loss change {rel_change:.2e} < {REL_TOL})")
            converged = True
            break

if not converged:
    print(f"\n  Reached max_iters={MAX_ITERS} without full convergence")

n_iters = len(losses_all)
losses_arr = np.array(losses_all)
spans_arr  = np.array(spans_all)   # (n_iters, K)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate chi on all cells
# ══════════════════════════════════════════════════════════════════════════════
print("\nEvaluating chi on all cells …")
net.eval()
with torch.no_grad():
    chi_all = net(x_all_t).cpu().numpy()   # (N_cells, K)

chi_sd = chi_all.std(0)
print(f"  SD per mode: {chi_sd.round(3)}")

# Determine k_eff from SD elbow (first mode below 0.05 threshold)
k_eff = max(1, int(np.sum(chi_sd > 0.05)))
print(f"  SD threshold=0.05 → k_eff={k_eff}")

# Eigenvalues
from amore.isokann.power import koopman_matrix, implied_timescales
with torch.no_grad():
    ev, its = implied_timescales(
        torch.tensor(chi_all[src], dtype=torch.float32),
        torch.tensor(chi_all[dst], dtype=torch.float32),
        lagtime=1.0)
print(f"  Eigenvalues: {np.abs(ev[:8]).round(3)}")
print(f"  Implied timescales: {its[:7].round(2)}")

# ── Save arrays ───────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "larry_hvg_chi_k16.npy"),     chi_all)
np.save(os.path.join(OUT_DIR, "larry_hvg_losses.npy"),      losses_arr)
np.save(os.path.join(OUT_DIR, "larry_hvg_spans.npy"),       spans_arr)
np.save(os.path.join(OUT_DIR, "larry_hvg_eigenvalues.npy"), np.abs(ev))


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

# ── Spearman + Pearson r on NeuMon hard split ─────────────────────────────────
bias_valid = nm_bias[nm_mask]
chi_nm     = chi_all[nm_mask]

best_sp, best_pe, best_col = 0.0, 0.0, 0
for i in range(K):
    r_sp, _ = spearmanr(chi_nm[:, i], bias_valid)
    r_pe, _ = pearsonr(chi_nm[:, i],  bias_valid)
    if abs(r_sp) > abs(best_sp):
        best_sp, best_pe, best_col = r_sp, r_pe, i

print(f"\nNeu-vs-Mono hard split (n={nm_mask.sum()}):")
print(f"  Best col={best_col}  Spearman r={best_sp:.3f}  Pearson r={best_pe:.3f}")
print(f"  Literature state-only ceiling: Spearman r ≈ 0.3–0.5")

# ── AUC-ROC per fate at k_eff ─────────────────────────────────────────────────
print(f"\nAU-ROC at k_eff={k_eff} (best chi column per fate):")
chi_d2 = chi_all[day2_mask]
auc_results = {}
for fate in FATES:
    if fate not in fate_labels: continue
    labs = fate_labels[fate][day2_mask].astype(int)
    if labs.sum() < 5: continue
    best_auc = 0.0
    for i in range(k_eff):
        for scores in [chi_d2[:, i], -chi_d2[:, i]]:
            try: best_auc = max(best_auc, roc_auc_score(labs, scores))
            except: pass
    auc_results[fate] = best_auc
    print(f"  {fate:15s}  AUC={best_auc:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

STATE_COLORS = {
    "undiff":"#aaaaaa","Neutrophil":"#4daf4a","Monocyte":"#f781bf",
    "Baso":"#ff7f00","Mast":"#e41a1c","Meg":"#984ea3",
    "Erythroid":"#a65628","Lymphoid":"#377eb8","Eos":"#999999",
    "Neu_Mon":"#8dd3c7","Ccr7_DC":"#ffff33","pDC":"#e0e0e0",
}

def savefig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

# ── Loss curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(range(1, n_iters+1), losses_arr, lw=1.5, color="steelblue")
ax.axvline(n_iters, ls="--", c="gray", lw=1,
           label=f"{'converged' if converged else 'max iters'} @ {n_iters}")
ax.set_xlabel("Outer iteration"); ax.set_ylabel("MSE loss")
ax.set_title(f"SVD-Power k={K} HVG (BN+ReLU+sigmoid) — convergence curve")
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "larry_hvg_loss_curve.png")

# ── SD profile ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Final SD bar
ax = axes[0]
colors = ["tomato" if s > 0.05 else "#cccccc" for s in chi_sd]
ax.bar(range(1, K+1), chi_sd, color=colors, alpha=0.85)
ax.axhline(0.05, ls="--", c="gray", lw=1, label="collapse threshold (0.05)")
ax.set_xlabel("Mode"); ax.set_ylabel("SD of chi")
ax.set_title(f"SD profile — k_eff={k_eff} live modes")
ax.legend(fontsize=8)

# SD trajectory across iterations
ax = axes[1]
for i in range(K):
    alpha = 0.8 if chi_sd[i] > 0.05 else 0.25
    lw    = 1.5 if chi_sd[i] > 0.05 else 0.6
    color = f"C{i % 10}"
    ax.plot(range(1, n_iters+1), spans_arr[:, i],
            lw=lw, alpha=alpha, color=color, label=f"χ{i+1}" if i < 8 else "")
ax.axhline(0.05, ls="--", c="gray", lw=1)
ax.set_xlabel("Iteration"); ax.set_ylabel("Chi span (max−min)")
ax.set_title("Mode activation over training")
ax.legend(fontsize=7, ncol=2, loc="upper left")
plt.suptitle(f"SVD-Power k={K}, HVG={N_HVG}, ReLU+sigmoid, {n_iters} iters", fontsize=10)
plt.tight_layout()
savefig(fig, "larry_hvg_sd_profile.png")

# ── Eigenvalue spectrum ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.5))
ev_abs = np.abs(ev[:K])
ax.bar(range(1, K+1), ev_abs, color="steelblue", alpha=0.8)
ax.axhline(1, ls="--", c="gray", lw=1)
ax.axvline(k_eff + 0.5, ls="--", c="tomato", lw=1.5, label=f"k_eff={k_eff}")
ax.set_xlabel("Mode"); ax.set_ylabel("|λ|")
ax.set_title(f"Koopman eigenvalues — HVG k={K}")
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "larry_hvg_eigenvalues.png")

# ── NeuMon scatter ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(chi_all[nm_mask, best_col], bias_valid,
           s=3, alpha=0.3, c="steelblue")
ax.set_xlabel(f"χ_{best_col+1}  (HVG, k={K})")
ax.set_ylabel("Neu / (Neu+Mono) fate bias")
ax.set_title(f"Neu-vs-Mono hard split\n"
             f"Spearman r={best_sp:.3f}  Pearson r={best_pe:.3f}\n"
             f"Literature state-only: 0.26–0.50")
plt.tight_layout()
savefig(fig, "larry_hvg_neumon_scatter.png")

# ── UMAP in chi-space (k_eff modes) ──────────────────────────────────────────
print(f"\nComputing UMAP on chi[:, :{k_eff}] …")
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        metric="euclidean", random_state=42, verbose=False)
    chi_umap = reducer.fit_transform(chi_all[:, :k_eff])
    np.save(os.path.join(OUT_DIR, "larry_hvg_chi_umap.npy"), chi_umap)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in [
        (axes[0], X_umap_orig,  "Original UMAP (PCA-40)"),
        (axes[1], chi_umap,     f"Chi-space UMAP (HVG k_eff={k_eff})"),
    ]:
        for state, c in STATE_COLORS.items():
            m = state_info == state
            ax.scatter(coords[m, 0], coords[m, 1],
                       s=1, c=c, alpha=0.5, label=state, rasterized=True)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=4, fontsize=7, ncol=2,
                   loc="upper left", framealpha=0.7)
    plt.suptitle(
        f"LARRY — original UMAP vs chi-space UMAP\n"
        f"SVD-Power HVG-{N_HVG}, k={K}, k_eff={k_eff}, {n_iters} iters",
        fontsize=11)
    plt.tight_layout()
    savefig(fig, "larry_hvg_umap.png")
except ImportError:
    print("  umap-learn not available — skipping UMAP")

# ── Pairwise chi scatter (first 4 modes) ─────────────────────────────────────
pairs = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
n_pair = len([p for p in pairs if p[1] < k_eff])
if n_pair > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (i, j) in zip(axes.ravel(), pairs):
        if j >= k_eff:
            ax.set_visible(False); continue
        for state, c in STATE_COLORS.items():
            m = state_info == state
            ax.scatter(chi_all[m, i], chi_all[m, j],
                       s=1, c=c, alpha=0.4, rasterized=True)
        ax.set_xlabel(f"χ_{i+1}  SD={chi_sd[i]:.3f}")
        ax.set_ylabel(f"χ_{j+1}  SD={chi_sd[j]:.3f}")
    axes[0, 0].legend(
        [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=c,ms=5)
         for c in STATE_COLORS.values()],
        list(STATE_COLORS.keys()), fontsize=6, ncol=2,
        bbox_to_anchor=(0,1), loc="lower left")
    plt.suptitle(f"Pairwise chi scatter — SVD-Power HVG k_eff={k_eff}", fontsize=11)
    plt.tight_layout()
    savefig(fig, "larry_hvg_chi_scatter.png")

print(f"\n=== SUMMARY ===")
print(f"  k={K} trained,  k_eff={k_eff} live modes (SD > 0.05)")
print(f"  Converged at iter {n_iters}")
print(f"  Eigenvalues: {np.abs(ev[:k_eff+1]).round(3)}")
print(f"  NeuMon Spearman r = {best_sp:.3f}  Pearson r = {best_pe:.3f}")
print(f"  AUC-ROC: {auc_results}")
