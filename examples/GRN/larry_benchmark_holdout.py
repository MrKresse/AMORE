"""
LARRY clone holdout: retrain ISOKANN on 80% of clones, evaluate on 20%.

This is the proper generalization test — day-2 progenitors from held-out
clones were never seen during chi training, yet their fate is predicted
from chi alone.

Outputs
-------
  output/benchmark/holdout_auroc.png
  output/benchmark/holdout_auroc.csv
  output/benchmark/holdout_chi_all.npy     — chi for held-out cells
  output/benchmark/holdout_membership.npy  — PCCA+(k=4) membership
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

BASE  = os.path.dirname(__file__)
DATA  = os.path.join(BASE, "data")
OUT   = os.path.join(BASE, "output")
BENCH = os.path.join(OUT, "benchmark")
os.makedirs(BENCH, exist_ok=True)

FATES    = ["Mast", "Baso", "Meg", "Erythroid", "Lymphoid",
            "Neutrophil", "Monocyte", "Eos", "pDC", "Ccr7_DC"]
FATE_COLS = [f"progenitor_{f}" for f in FATES]
HOLDOUT_FRAC = 0.20
SEED = 42
K_CHI = 13
K_PCCA = 4   # spectral gap k for PCCA+

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
import anndata
import scipy.sparse as sp

adata  = anndata.read_h5ad(os.path.join(DATA, "larry_processed.h5ad"))
obs    = adata.obs.copy()
X_pca  = adata.obsm["X_pca"].astype(np.float32)
X_clone = adata.obsm["X_clone"]   # (49116, 5864) sparse
src    = np.load(os.path.join(DATA, "larry_src.npy"))
dst    = np.load(os.path.join(DATA, "larry_dst.npy"))

N_CELLS  = X_pca.shape[0]
N_CLONES = X_clone.shape[1]
print(f"  {N_CELLS} cells, {N_CLONES} clones, {len(src)} Koopman pairs")

# day-2 mask
day2_mask = obs["time_info"].astype(str).values == "2"
print(f"  Day-2 cells: {day2_mask.sum()}")

# fate labels
fate_labels = {f: obs[c].values.astype(float)
               for f, c in zip(FATES, FATE_COLS) if c in obs.columns}

# ══════════════════════════════════════════════════════════════════════════════
# 2. Clone split — 80 / 20 by clone barcode
# ══════════════════════════════════════════════════════════════════════════════
print("Splitting clones …")
rng = np.random.default_rng(SEED)

# For each day-2 cell with a fate label: find which clone(s) it belongs to
# Use the clone with the most cells (each day-2 cell may belong to one clone)
X_csr = X_clone.tocsr()
# clone membership per cell: argmax row of X_clone (one clone per cell)
cell_clone = np.array(X_csr.argmax(axis=1)).ravel()  # (N_CELLS,)

all_clones = np.arange(N_CLONES)
rng.shuffle(all_clones)
n_hold = int(N_CLONES * HOLDOUT_FRAC)
hold_clones  = set(all_clones[:n_hold].tolist())
train_clones = set(all_clones[n_hold:].tolist())
print(f"  Train clones: {len(train_clones)}, Hold-out clones: {len(hold_clones)}")

# Cell membership
cell_in_train = np.array([cell_clone[i] in train_clones for i in range(N_CELLS)])
cell_in_hold  = np.array([cell_clone[i] in hold_clones  for i in range(N_CELLS)])
print(f"  Train cells: {cell_in_train.sum()}, Hold-out cells: {cell_in_hold.sum()}")

# Koopman pairs where BOTH src and dst are in train
pair_in_train = cell_in_train[src] & cell_in_train[dst]
src_tr = src[pair_in_train]
dst_tr = dst[pair_in_train]
print(f"  Train pairs: {pair_in_train.sum()} / {len(src)}")

# Hold-out day-2 progenitors
hold_day2 = cell_in_hold & day2_mask
print(f"  Hold-out day-2 progenitors: {hold_day2.sum()}")
for fate in FATES:
    if fate in fate_labels:
        n = (fate_labels[fate][hold_day2] > 0).sum()
        if n > 0: print(f"    {fate}: {n}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Build feature tensors for training
# ══════════════════════════════════════════════════════════════════════════════
import torch

DEVICE = torch.device("cpu")
N_PCS  = X_pca.shape[1]   # 40

x0_tr = torch.tensor(X_pca[src_tr], dtype=torch.float32, device=DEVICE)
x1_tr = torch.tensor(X_pca[dst_tr], dtype=torch.float32, device=DEVICE)
x_all = torch.tensor(X_pca,         dtype=torch.float32, device=DEVICE)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Import ISOKANN module and train
# ══════════════════════════════════════════════════════════════════════════════
from amore.isokann import ChiNetMultiRaw, power_method_multi, koopman_matrix, whiten

N_POWER_ITER   = 80
EPOCHS_PER_ITER = 400
LR             = 2e-3
LR_DECAY       = 0.97

net = ChiNetMultiRaw(in_dim=N_PCS, k=K_CHI, hidden=[512, 256, 128]).to(DEVICE)

print(f"\nTraining chi (k={K_CHI}, {N_POWER_ITER} power iter × {EPOCHS_PER_ITER} epochs) …")
print(f"  Training on {len(src_tr)} pairs (80% of clones)")

result = power_method_multi(
    net, x0_tr, x1_tr,
    n_iter=N_POWER_ITER,
    epochs_per_iter=EPOCHS_PER_ITER,
    lr=LR,
    lr_decay=LR_DECAY,
    verbose=True,
)
print(f"  Final loss: {result['losses'][-1]:.5f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluate chi on all cells
# ══════════════════════════════════════════════════════════════════════════════
print("\nEvaluating chi on all cells …")
net.eval()
with torch.no_grad():
    chi_hold = net(x_all).cpu().numpy()   # (N_CELLS, K_CHI)

np.save(os.path.join(BENCH, "holdout_chi_all.npy"), chi_hold)

# ══════════════════════════════════════════════════════════════════════════════
# 6. PCCA+ at k=K_PCCA and compute AUC-ROC on hold-out cells
# ══════════════════════════════════════════════════════════════════════════════
def pcca_rotation(chi_mat):
    n, k = chi_mat.shape
    chi_n = chi_mat - chi_mat.min(0)
    chi_n = chi_n / (chi_n.sum(1, keepdims=True) + 1e-8)
    vertex_idx = [int(np.argmax(np.linalg.norm(chi_n - chi_n.mean(0), axis=1)))]
    for _ in range(k - 1):
        dists = np.min(np.stack([np.linalg.norm(chi_n - chi_n[v], axis=1)
                                  for v in vertex_idx]), axis=0)
        vertex_idx.append(int(np.argmax(dists)))
    vertex_idx = np.array(vertex_idx)
    C = chi_n[vertex_idx]
    try:
        A = np.linalg.inv(C)
        membership = chi_n @ A
        membership = np.clip(membership, 0, None)
        membership = membership / (membership.sum(1, keepdims=True) + 1e-8)
    except np.linalg.LinAlgError:
        membership = chi_n
    return membership, vertex_idx


def best_auc(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return max(roc_auc_score(labels, scores), roc_auc_score(labels, -scores))


print(f"PCCA+ rotation at k={K_PCCA} …")
mem_pcca, _ = pcca_rotation(chi_hold[:, :K_PCCA])
np.save(os.path.join(BENCH, "holdout_membership.npy"), mem_pcca)

# AUC-ROC on hold-out day-2 progenitors
print("\nHold-out AUC-ROC (day-2 progenitors from 20% held-out clones):")
hold_results = {"k_pcca": K_PCCA}
mem_ho  = mem_pcca[hold_day2]
chi_ho  = chi_hold[hold_day2]

rows = []
for fate in FATES:
    if fate not in fate_labels: continue
    labels = fate_labels[fate][hold_day2].astype(int)
    if labels.sum() < 3: continue
    auc_mem = max(best_auc(mem_ho[:, i], labels) for i in range(K_PCCA))
    auc_chi = max(best_auc(chi_ho[:, i], labels) for i in range(K_CHI))
    print(f"  {fate:15s}  membership={auc_mem:.3f}  chi={auc_chi:.3f}  (n={labels.sum()})")
    rows.append({"fate": fate, "auc_membership": auc_mem, "auc_chi": auc_chi, "n_pos": labels.sum()})

df_hold = pd.DataFrame(rows).set_index("fate")
df_hold.to_csv(os.path.join(BENCH, "holdout_auroc.csv"))

# Compare with full-model at same fates
print("\nComparison: full-model (all clones) vs hold-out model at k=13 best chi:")
chi_full = np.load(os.path.join(OUT, "multi_chi_all.npy"))
for fate in FATES:
    if fate not in fate_labels: continue
    labels = fate_labels[fate][hold_day2].astype(int)
    if labels.sum() < 3: continue
    chi_full_ho = chi_full[hold_day2]
    auc_full = max(best_auc(chi_full_ho[:, i], labels) for i in range(K_CHI))
    auc_new  = df_hold.loc[fate, "auc_chi"] if fate in df_hold.index else float("nan")
    print(f"  {fate:15s}  full-model={auc_full:.3f}  hold-out-model={auc_new:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
if len(rows) > 0:
    FATE_COLORS = {
        "Mast": "#e41a1c", "Baso": "#ff7f00", "Meg": "#984ea3",
        "Erythroid": "#a65628", "Lymphoid": "#377eb8", "Neutrophil": "#4daf4a",
        "Monocyte": "#f781bf", "Eos": "#999999", "pDC": "#ffff33", "Ccr7_DC": "#8dd3c7",
    }
    fates_plot = [r["fate"] for r in rows]
    mem_vals   = [r["auc_membership"] for r in rows]
    chi_vals   = [r["auc_chi"]        for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(fates_plot))
    w = 0.35
    ax.bar(x - w/2, mem_vals, w, label=f"PCCA+ k={K_PCCA}", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, chi_vals, w, label=f"Best chi k={K_CHI}", color="tomato", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(fates_plot, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("AU-ROC"); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_title(f"Clone holdout ({int(HOLDOUT_FRAC*100)}% hold-out): AU-ROC on unseen clones")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(BENCH, "holdout_auroc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: holdout_auroc.png")

print("\nClone holdout complete.")
