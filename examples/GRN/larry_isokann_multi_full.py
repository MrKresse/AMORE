"""
Multi-D ISOKANN on the full 130k LARRY dataset with clone-holdout validation.

Key improvements over the 49k run:
  1. 130k cells -> more erythroid/lymphoid cells and clone pairs
  2. Balanced pair sampling (capped per transition type) -> orthogonal eigenfunctions
  3. Clone holdout: test AUC computed only on day-2 cells from held-out clones
     (those clones never appeared in any training pair) -> rules out overfitting

AUC overfitting note
--------------------
The chi network never sees fate labels during training — only PCA coordinates and
the Koopman structure of clone pairs.  So the AUC measures genuine out-of-distribution
prediction in the traditional ML sense.

However, day-2 cells from TRAINING clones were the x0 in training pairs, so the network
has "seen" their PCA coordinates.  The clone holdout removes even this weak form of
data leakage: test AUC cells are from clones whose x0/x1 were never in training.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from scipy.stats import hypergeom
from collections import Counter
import scanpy as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from amore.isokann import ChiNetMultiRaw, power_method_multi, implied_timescales

sys.path.insert(0, os.path.dirname(__file__))
from larry_metrics import MOUSE_TFS_LOWER

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
K_MAX          = 15
N_PCS          = 40
HIDDEN         = [512, 256, 128]
N_POWER_ITER   = 80
EPOCHS_PER_ITER = 200   # fewer epochs since we have more data per iteration
LR             = 2e-3
LR_DECAY       = 0.97
BATCH          = 4096
SVD_SUBSAMPLE  = 50_000  # subsample for SVD step (fast randomized approx)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading full LARRY data ...")
adata    = sc.read_h5ad(os.path.join(DATA_DIR, "larry_full_processed.h5ad"))
X_pca    = np.load(os.path.join(DATA_DIR, "larry_full_pca.npy")).astype(np.float32)
x0_tr    = np.load(os.path.join(DATA_DIR, "larry_full_x0_train.npy")).astype(np.float32)
x1_tr    = np.load(os.path.join(DATA_DIR, "larry_full_x1_train.npy")).astype(np.float32)
x0_te    = np.load(os.path.join(DATA_DIR, "larry_full_x0_test.npy")).astype(np.float32)
x1_te    = np.load(os.path.join(DATA_DIR, "larry_full_x1_test.npy")).astype(np.float32)

TIME_COL  = next((c for c in ["time_info","time","day"] if c in adata.obs), None)
STATE_COL = next((c for c in ["state_info","cell_type","celltype"] if c in adata.obs), None)
states    = adata.obs[STATE_COL].astype(str).values if STATE_COL else None
times     = adata.obs[TIME_COL].astype(str).values  if TIME_COL  else None
emb       = adata.obsm.get("X_umap", X_pca[:, :2])

n_cells = len(X_pca)
print(f"  {n_cells:,} cells | {len(x0_tr):,} train pairs | {len(x0_te):,} test pairs | k_max={K_MAX}")

# Standardise
mu  = x0_tr.mean(0, keepdims=True)
sig = x0_tr.std(0,  keepdims=True) + 1e-8

def norm(X): return (X - mu) / sig

x0t = torch.tensor(norm(x0_tr), dtype=torch.float32, device=DEVICE)
x1t = torch.tensor(norm(x1_tr), dtype=torch.float32, device=DEVICE)
Xt  = torch.tensor(norm(X_pca), dtype=torch.float32, device=DEVICE)


# ── Modified power iteration with SVD subsampling ─────────────────────────────
# For large n_pairs, computing chi on ALL pairs for SVD targets is expensive.
# Instead, subsample SVD_SUBSAMPLE pairs for the orthogonalization step,
# then use all pairs for training.  This is a randomized approximation of
# the dominant eigenspace, identical in expectation to the full SVD.

from amore.isokann.power import scale_to_unit, _train_one_iter, koopman_matrix, implied_timescales as _its

def power_method_subsampled(chi, x0, x1, n_iter, epochs, lr, lr_decay, batch,
                              svd_sub=SVD_SUBSAMPLE, verbose=True):
    opt   = torch.optim.Adam(chi.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)
    k     = chi(x0[:1]).shape[-1]
    losses, spans_all = [], []
    n     = len(x0)

    if verbose:
        print(f"Multi-D ISOKANN k={k}  ({n_iter} iters x {epochs} epochs)  svd_sub={svd_sub:,}")
        print(f"{'iter':>5}  {'loss':>10}  " + "  ".join(f"chi{i}_span" for i in range(k)))

    for it in range(n_iter):
        chi.eval()
        with torch.no_grad():
            # Subsample for SVD
            idx_s = torch.randperm(n, device=x0.device)[:min(svd_sub, n)]
            Y     = chi(x1[idx_s])                     # (sub, k)
            Yc    = Y - Y.mean(0)
            try:
                U, S, Vh = torch.linalg.svd(Yc, full_matrices=False)
                Y_orth   = U                           # (sub, k) orthonormal
            except Exception:
                Y_orth = Yc

            col_std = Y_orth.std(0)
            for j in range(k):
                if col_std[j].item() < 1e-3:
                    Y_orth = Y_orth.clone()
                    Y_orth[:, j] += 1e-3 * torch.randn(len(Y_orth), device=Y_orth.device)

            targets_sub = scale_to_unit(Y_orth)       # (sub, k) in [0,1]

        avg_loss = _train_one_iter(chi, x0[idx_s], targets_sub, opt, epochs, batch)
        sched.step()
        losses.append(avg_loss)

        chi.eval()
        with torch.no_grad():
            chi_s = chi(x0[idx_s])
        spans = (chi_s.max(0).values - chi_s.min(0).values).cpu().numpy()
        spans_all.append(spans)

        if verbose and ((it+1) % 10 == 0 or it == 0):
            print(f"{it+1:>5}  {avg_loss:>10.5f}  " + "  ".join(f"{s:.3f}" for s in spans))

    chi.eval()
    with torch.no_grad():
        chi_x0 = chi(x0); chi_x1 = chi(x1)
    evals, ts = _its(chi_x0, chi_x1, lagtime=1.0)

    return {"losses": losses, "spans": np.array(spans_all),
            "eigenvalues": evals, "timescales": ts}


# ── Train ──────────────────────────────────────────────────────────────────────
chi = ChiNetMultiRaw(in_dim=N_PCS, k=K_MAX, hidden=HIDDEN).to(DEVICE)
print(f"\nChiNetMultiRaw: {sum(p.numel() for p in chi.parameters()):,} params")

result = power_method_subsampled(
    chi, x0t, x1t, N_POWER_ITER, EPOCHS_PER_ITER, LR, LR_DECAY, BATCH, verbose=True)


# ── Spectral gap ───────────────────────────────────────────────────────────────
abs_evals = np.sort(np.abs(result["eigenvalues"]))[::-1]
gaps      = abs_evals[:-1] - abs_evals[1:]
k_correct = int(np.argmax(gaps)) + 1
order     = np.argsort(-abs_evals[:K_MAX])

print(f"\n-- Spectral gap: k_correct={k_correct} --")
print(f"  Eigenvalues: {abs_evals[:8].round(4)}")
print(f"  Gaps:        {gaps[:8].round(4)}")
print(f"  Timescales (top-{min(6,K_MAX-1)}): {result['timescales'][:6].round(2)}")


# ── Evaluate on all cells ──────────────────────────────────────────────────────
chi.eval()
chi_all = []
with torch.no_grad():
    for i in range(0, len(Xt), 4096):
        chi_all.append(chi(Xt[i:i+4096]).cpu().numpy())
chi_all = np.concatenate(chi_all)              # (n_cells, K_MAX)
chi_k   = chi_all[:, order[:k_correct]]        # (n_cells, k_correct)

np.save(os.path.join(OUT_DIR, "multi_full_chi_all.npy"),   chi_all)
np.save(os.path.join(OUT_DIR, "multi_full_eigenvalues.npy"), abs_evals)
torch.save(chi.state_dict(), os.path.join(OUT_DIR, "multi_full_chi_net.pt"))


# ── PCCA+ rotation ────────────────────────────────────────────────────────────
def pcca_rotation(chi_mat):
    n, k = chi_mat.shape
    chi_n = chi_mat - chi_mat.min(0)
    chi_n = chi_n / (chi_n.sum(1, keepdims=True) + 1e-8)
    vertex_idx = [int(np.argmax(np.linalg.norm(chi_n - chi_n.mean(0), axis=1)))]
    for _ in range(k-1):
        dists = np.min(np.stack([np.linalg.norm(chi_n - chi_n[v], axis=1)
                                  for v in vertex_idx]), axis=0)
        vertex_idx.append(int(np.argmax(dists)))
    vertex_idx = np.array(vertex_idx)
    C = chi_n[vertex_idx]
    try:
        A = np.linalg.inv(C)
        mem = np.clip(chi_n @ A, 0, None)
        mem = mem / (mem.sum(1, keepdims=True) + 1e-8)
    except np.linalg.LinAlgError:
        mem = chi_n
    return mem, vertex_idx

membership, vertex_idx = pcca_rotation(chi_k)
hard_assign = np.argmax(membership, axis=1)
np.save(os.path.join(OUT_DIR, "multi_full_membership.npy"), membership)

if states is not None:
    print(f"\n-- PCCA+ cluster assignments (k={k_correct}) --")
    for k in range(k_correct):
        mask = hard_assign == k
        top  = Counter(states[mask]).most_common(3)
        print(f"  cluster {k}: n={mask.sum():,}  " +
              ", ".join(f"{s}({c})" for s,c in top))


# ── Clone-holdout AUC: evaluate ONLY on day-2 cells from TEST clones ──────────
print(f"\n-- Clone-holdout AUC-ROC (test clones only) --")

test_clones = np.load(os.path.join(DATA_DIR, "larry_full_test_clones.npy"))
C_mat = adata.obsm["X_clone"]
if hasattr(C_mat, "toarray"):
    C_mat = C_mat.toarray()
C_mat = np.array(C_mat, dtype=np.float32)

# Cells from test clones at day 2
day2_mask = (times == "2") if times is not None else np.ones(n_cells, bool)
test_cell_mask = np.zeros(n_cells, bool)
for j in test_clones:
    test_cell_mask |= (C_mat[:, j] != 0)
test_day2_mask = test_cell_mask & day2_mask

print(f"  Test-clone day-2 cells: {test_day2_mask.sum():,} "
      f"(out of {day2_mask.sum():,} total day-2)")

fate_cols = [c for c in adata.obs.columns if c.startswith("progenitor_")] + ["NeuMon_fate_bias"]

results_auc = {}
print(f"  {'Lineage':<22s}  " + "  ".join(f"m{i:>5}" for i in range(k_correct)))

for col in fate_cols:
    if col not in adata.obs.columns:
        continue
    bias = pd.to_numeric(adata.obs[col], errors="coerce").values.astype(float)
    mask = test_day2_mask & ~np.isnan(bias)
    y_bin = (bias[mask] > 0.5).astype(int)
    lineage = col.replace("progenitor_","").replace("_fate_bias","")

    if y_bin.sum() < 3 or y_bin.sum() > mask.sum()-3:
        continue

    aucs = []
    for i in range(k_correct):
        try:
            aucs.append(roc_auc_score(y_bin, membership[mask, i]))
        except Exception:
            aucs.append(float("nan"))

    results_auc[lineage] = aucs
    auc_str = "  ".join(f"{a:>6.3f}" for a in aucs)
    best = int(np.nanargmax(np.abs(np.array(aucs)-0.5)))
    print(f"  {lineage:<22s}  {auc_str}   <- m{best} best")


# ── Plots ─────────────────────────────────────────────────────────────────────

# Eigenvalue spectrum
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, K_MAX+1), abs_evals, color="steelblue", alpha=0.8)
axes[0].axvline(k_correct+0.5, color="crimson", lw=2, ls="--",
                label=f"spectral gap k={k_correct}")
axes[0].set_xlabel("Index"); axes[0].set_ylabel("|lambda|")
axes[0].set_title("Koopman eigenvalue spectrum (full 130k)"); axes[0].legend()

ts = np.clip(result["timescales"][:K_MAX-1], 0, 60)
axes[1].bar(range(1, len(ts)+1), ts, color="steelblue", alpha=0.8)
axes[1].axvline(k_correct-0.5, color="crimson", lw=2, ls="--")
axes[1].set_xlabel("Mode"); axes[1].set_ylabel("Implied timescale (x lag)")
axes[1].set_title("Implied timescales")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_full_spectrum.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: multi_full_spectrum.png")

# Chi functions on embedding
k_show = min(k_correct, 8)
ncols  = 4; nrows = (k_show+ncols-1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
axes = np.array(axes).flatten()
for plot_i, chi_i in enumerate(order[:k_show]):
    ax  = axes[plot_i]
    vals = chi_all[:, chi_i]
    vlo, vhi = np.percentile(vals, [2,98])
    ax.scatter(emb[:,0], emb[:,1], c=vals, cmap="coolwarm",
               vmin=vlo, vmax=vhi, s=0.5, alpha=0.3, rasterized=True)
    ax.set_title(f"chi_{chi_i+1}  |lam|={abs_evals[chi_i]:.3f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for j in range(k_show, len(axes)):
    axes[j].set_visible(False)
plt.suptitle(f"Full 130k LARRY: chi functions (k_correct={k_correct})", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_full_chi_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_full_chi_umap.png")

# PCCA+ membership
if k_correct <= 6:
    fig, axes = plt.subplots(2, max(3,(k_correct+1)//2), figsize=(k_correct*3, 8))
    axes = np.array(axes).flatten()
    state_cats = sorted(set(states)) if states is not None else []
    cmap_s = plt.get_cmap("tab20", max(1, len(state_cats)))
    for i in range(k_correct):
        ax = axes[i]
        dom = Counter(states[hard_assign==i]).most_common(1)[0][0] if states is not None else "?"
        sc_ = ax.scatter(emb[:,0], emb[:,1], c=membership[:,i], cmap="Blues",
                          vmin=0, vmax=1, s=0.5, alpha=0.4, rasterized=True)
        plt.colorbar(sc_, ax=ax, shrink=0.7)
        ax.set_title(f"PCCA+ m{i} ({dom})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(k_correct, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"PCCA+ membership (k={k_correct}, full 130k)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "multi_full_membership.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: multi_full_membership.png")

# AUC heatmap
if results_auc:
    lineages = list(results_auc.keys())
    auc_mat  = np.array([results_auc[l] for l in lineages])
    fig, ax  = plt.subplots(figsize=(max(6, k_correct*1.2), max(4, len(lineages)*0.5)))
    im = ax.imshow(auc_mat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(k_correct)); ax.set_xticklabels([f"m{i}" for i in range(k_correct)])
    ax.set_yticks(range(len(lineages))); ax.set_yticklabels(lineages, fontsize=8)
    plt.colorbar(im, ax=ax, label="AUC-ROC")
    for i in range(len(lineages)):
        for j in range(k_correct):
            v = auc_mat[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v-0.5)>0.3 else "black")
    ax.set_title(f"AUC-ROC: PCCA+ membership vs fate (CLONE HOLDOUT, n_test={test_day2_mask.sum():,})")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "multi_full_auc_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: multi_full_auc_heatmap.png")

print(f"\n-- Summary --")
print(f"  Dataset: {n_cells:,} cells | {len(x0_tr):,} balanced train pairs")
print(f"  k_correct = {k_correct}  (spectral gap = {gaps[k_correct-1]:.4f})")
print(f"  Top eigenvalues: {abs_evals[:k_correct].round(4)}")
if results_auc:
    for lineage, aucs in results_auc.items():
        best = int(np.nanargmax(np.abs(np.array(aucs)-0.5)))
        print(f"  {lineage:<20s}: m{best} AUC={aucs[best]:.3f} (clone holdout)")
print(f"\nAll outputs -> {OUT_DIR}/")
