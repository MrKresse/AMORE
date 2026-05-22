"""
Multi-D ISOKANN benchmark: AUC-ROC per PCCA+ membership + differentiable PCCA+ gradients.

Three analyses:
1. AUC-ROC for each membership function as a fate predictor (like 1D chi benchmark)
2. Differentiable PCCA+ gradients: ∂membership_i/∂gene = Σ_k A_ki * ∂chi_k/∂gene
   These are lineage-specific sensitivity scores (not just general axis sensitivity)
3. TF enrichment for each lineage-specific gradient — does each cluster pick up
   biologically relevant TFs for that specific lineage?

Differentiable PCCA+
--------------------
Given:
  chi(x)  : R^n -> R^k  (k Koopman eigenfunctions)
  A        : R^(k,k)     (rotation matrix, A = chi[vertices]^{-1})
  membership_i(x) = Σ_j chi_j(x) * A_{j,i}

The gradient is:
  ∂membership_i/∂gene_l = Σ_j A_{j,i} * ∂chi_j/∂gene_l

This is just a linear combination of the eigenfunction gradients — differentiable,
no need for a differentiable vertex selection step.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import hypergeom
from collections import Counter
import scanpy as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from amore.isokann import ChiNetMultiRaw

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")

N_PCS  = 40
HIDDEN = [512, 256, 128]
# Read K_MAX from checkpoint (last linear layer output dim) to avoid hardcoding
_sd   = torch.load(os.path.join(OUT_DIR, "multi_chi_net.pt"), map_location="cpu")
K_MAX = _sd[sorted(k for k in _sd if k.endswith(".weight"))[-1]].shape[0]
del _sd
print(f"  K_MAX={K_MAX} (from checkpoint)")

# Mouse TF list (from larry_metrics.py)
sys.path.insert(0, os.path.dirname(__file__))
from larry_metrics import MOUSE_TFS_LOWER

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading ...")
adata      = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
X_pca      = np.load(os.path.join(DATA_DIR, "larry_pca.npy")).astype(np.float32)
chi_all    = np.load(os.path.join(OUT_DIR,  "multi_chi_all.npy"))   # (49116, 15)
membership = np.load(os.path.join(OUT_DIR,  "multi_membership.npy"))# (49116, k_correct)
abs_evals  = np.load(os.path.join(OUT_DIR,  "multi_eigenvalues.npy"))

# k_correct comes from the membership matrix shape (set by larry_multi_analysis.py)
# Do NOT re-derive from eigenvalue gap here — it may differ from the override used in analysis
k_correct = membership.shape[1]
order     = np.argsort(-abs_evals)
chi_k     = chi_all[:, order[:k_correct]]   # (49116, k_correct)

obs    = adata.obs
states = obs["state_info"].astype(str).values
times  = obs["time_info"].astype(str).values
emb    = adata.obsm["X_umap"]
day2   = times == "2"
n_cells = len(chi_all)

print(f"  {n_cells:,} cells | k_correct={k_correct} | membership shape {membership.shape}")

# Cluster labels from membership argmax
hard_assign = np.argmax(membership, axis=1)
state_cats  = sorted(set(states))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  AUC-ROC per PCCA+ membership as fate predictor
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- AUC-ROC: PCCA+ membership vs fate-bias labels --")
fate_cols = [c for c in obs.columns if c.startswith("progenitor_")] + ["NeuMon_fate_bias"]

results_auc = {}
header = f"{'Lineage':<22s}  " + "  ".join(f"m{i:>5}" for i in range(k_correct))
print(f"  {header}")
print(f"  {'-'*70}")

for col in fate_cols:
    if col not in obs.columns:
        continue
    bias  = pd.to_numeric(obs[col], errors="coerce").values.astype(float)
    mask  = day2 & ~np.isnan(bias)
    y_bin = (bias[mask] > 0.5).astype(int)
    lineage = col.replace("progenitor_","").replace("_fate_bias","")

    if y_bin.sum() < 5 or y_bin.sum() > mask.sum()-5:
        continue

    aucs = []
    for i in range(k_correct):
        try:
            aucs.append(roc_auc_score(y_bin, membership[mask, i]))
        except Exception:
            aucs.append(float("nan"))

    results_auc[lineage] = aucs
    auc_str = "  ".join(f"{a:>6.3f}" for a in aucs)
    # Mark best AUC with *
    best = np.nanargmax(np.abs(np.array(aucs) - 0.5))
    print(f"  {lineage:<22s}  {auc_str}   <- m{best} best")

# Which cluster is the best predictor for which lineage?
print(f"\n  Lineage -> best membership:")
for lineage, aucs in results_auc.items():
    best_i = int(np.nanargmax(np.abs(np.array(aucs) - 0.5)))
    best_auc = aucs[best_i]
    direction = "pos" if best_auc > 0.5 else "neg"
    print(f"    {lineage:<20s}  m{best_i}  AUC={best_auc:.3f} ({direction})")

# Plot AUC heatmap
lineages = list(results_auc.keys())
auc_mat  = np.array([results_auc[l] for l in lineages])  # (n_lineages, k)

fig, ax = plt.subplots(figsize=(max(6, k_correct*1.2), max(4, len(lineages)*0.5)))
im = ax.imshow(auc_mat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(k_correct)); ax.set_xticklabels([f"m{i}" for i in range(k_correct)])
ax.set_yticks(range(len(lineages))); ax.set_yticklabels(lineages, fontsize=8)
plt.colorbar(im, ax=ax, label="AUC-ROC")
ax.axhline(0.5-0.5, color="gray", lw=0.5)
for i in range(len(lineages)):
    for j in range(k_correct):
        v = auc_mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v-0.5) > 0.3 else "black")
ax.set_title(f"AUC-ROC: PCCA+ membership (columns) vs fate (rows)\n"
             f"blue=membership predicts fate, red=anti-predicts")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_auc_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: multi_auc_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Differentiable PCCA+: ∂membership_i/∂gene
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- Differentiable PCCA+ gradients --")

# Reconstruct the rotation matrix A from vertices
# membership = chi_k @ A  =>  A = lstsq(chi_k, membership)
A_rot = np.linalg.lstsq(chi_k, membership, rcond=None)[0]  # (k_correct, k_correct)
print(f"  Rotation matrix A: {A_rot.shape}  (reconstruction error: "
      f"{np.mean((chi_k @ A_rot - membership)**2):.5f})")

# Load chi network and PCA loadings for gene-level sensitivity
chi_net = ChiNetMultiRaw(in_dim=N_PCS, k=K_MAX, hidden=HIDDEN)
chi_net.load_state_dict(torch.load(os.path.join(OUT_DIR, "multi_chi_net.pt"),
                                    map_location="cpu"))
chi_net.eval()

# Standardise PCA features (same as during training)
x0_raw = np.load(os.path.join(DATA_DIR, "larry_x0.npy")).astype(np.float32)
mu  = x0_raw.mean(0, keepdims=True)
sig = x0_raw.std(0,  keepdims=True) + 1e-8

Xn = torch.tensor((X_pca - mu) / sig, dtype=torch.float32)

# Compute ∂chi_j/∂PC for each cell (subset for speed)
n_grad = min(3000, n_cells)
rng    = np.random.default_rng(42)
idx_g  = rng.choice(n_cells, n_grad, replace=False)
Xg     = Xn[idx_g].clone().requires_grad_(True)

chi_g  = chi_net(Xg)[:, order[:k_correct]]  # (n_grad, k_correct)

# ∂chi/∂PC: (n_grad, k_correct, N_PCS) — compute column by column
dchi_dpc = []
for j in range(k_correct):
    chi_g.sum(0)[j].backward(retain_graph=(j < k_correct-1))
    dchi_dpc.append(Xg.grad.detach().clone().numpy())
    Xg.grad.zero_()

dchi_dpc = np.stack(dchi_dpc, axis=1)   # (n_grad, k_correct, N_PCS)

# Rotation: ∂membership_i/∂PC = Σ_j A_{j,i} * ∂chi_j/∂PC
# dmem_dpc[cell, i, pc] = Σ_j A_rot[j,i] * dchi_dpc[cell, j, pc]
dmem_dpc = np.einsum("nkp,ki->nip", dchi_dpc, A_rot)  # (n_grad, k_correct, N_PCS)

# Gene-level via Ridge regression loadings (dual form)
X_expr = adata.X
if hasattr(X_expr, "toarray"):
    X_expr = X_expr.toarray()
X_expr = np.array(X_expr, dtype=np.float32)

print("  Fitting gene->PC mapping via dual Ridge ...")
n_sub  = 6000
idx_r  = rng.choice(n_cells, n_sub, replace=False)
X_c    = (X_expr[idx_r] - X_expr[idx_r].mean(0)).astype(np.float64)
P_c    = (X_pca[idx_r]  - X_pca[idx_r].mean(0)).astype(np.float64)
alpha  = 1.0
K_gram = X_c @ X_c.T + alpha * np.eye(n_sub)
dual   = np.linalg.solve(K_gram, P_c)
W      = (X_c.T @ dual).astype(np.float32)  # (n_genes, N_PCS)

gene_names = np.array(adata.var_names)

# Sensitivity per membership function: |∂membership_i/∂gene|
mem_sens = {}
for i in range(k_correct):
    mean_dpc_i  = np.abs(dmem_dpc[:, i, :]).mean(0)   # (N_PCS,)
    gene_sens_i = np.abs(W) @ mean_dpc_i               # (n_genes,)
    mem_sens[i] = gene_sens_i

print(f"\n  Top-15 genes per membership function:")
for i in range(k_correct):
    top_idx  = np.argsort(mem_sens[i])[::-1][:15]
    top_g    = gene_names[top_idx]
    top_s    = mem_sens[i][top_idx]
    tfs_in   = [g for g in top_g if g.lower() in MOUSE_TFS_LOWER]
    print(f"\n  Membership {i} (cluster: {sorted(Counter(states[hard_assign==i]).items(), key=lambda x:-x[1])[:2]}):")
    for g, s in zip(top_g, top_s):
        tag = " **TF**" if g.lower() in MOUSE_TFS_LOWER else ""
        print(f"    {g:<20s} {s:.5f}{tag}")
    print(f"  TFs in top-15: {tfs_in}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TF enrichment per membership gradient
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- TF enrichment per membership --")
N_universe = len(gene_names)
K_tfs      = sum(1 for g in gene_names if g.lower() in MOUSE_TFS_LOWER)
print(f"  Universe: {N_universe:,} genes | TFs: {K_tfs} ({100*K_tfs/N_universe:.1f}%)")
print(f"  {'Membership':<14}  {'top-100 TFs':>11}  {'expected':>9}  {'p-value':>12}  {'fold':>6}")
print(f"  {'-'*58}")

for i in range(k_correct):
    top100 = gene_names[np.argsort(mem_sens[i])[::-1][:100]]
    k_found = sum(1 for g in top100 if g.lower() in MOUSE_TFS_LOWER)
    expected = 100 * K_tfs / N_universe
    pval = hypergeom.sf(k_found - 1, N_universe, K_tfs, 100)
    fold = k_found / expected
    dominant_state = Counter(states[hard_assign==i]).most_common(1)[0][0]
    print(f"  m{i} ({dominant_state:<10s})  {k_found:>11}  {expected:>9.1f}  "
          f"{pval:>12.2e}  {fold:>6.1f}x")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Bar plots: top-20 genes per membership
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, k_correct, figsize=(k_correct*4, 5))
if k_correct == 1:
    axes = [axes]

dominant_labels = {i: Counter(states[hard_assign==i]).most_common(1)[0][0]
                   for i in range(k_correct)}

for i, ax in enumerate(axes):
    top_idx  = np.argsort(mem_sens[i])[::-1][:20]
    top_g    = gene_names[top_idx]
    top_s    = mem_sens[i][top_idx]
    colors   = ["crimson" if g.lower() in MOUSE_TFS_LOWER else "steelblue" for g in top_g]
    ax.barh(range(20), top_s[::-1], color=colors[::-1])
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_g[::-1], fontsize=7)
    ax.set_xlabel("|∂membership/∂gene|")
    ax.set_title(f"m{i}: {dominant_labels[i]}", fontsize=9)

plt.suptitle("Top-20 genes per PCCA+ membership gradient  (red=TF)", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_membership_sensitivity.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: multi_membership_sensitivity.png")

# Save full tables
for i in range(k_correct):
    pd.DataFrame({
        "gene":       gene_names,
        "sensitivity": mem_sens[i],
        "rank":        np.argsort(np.argsort(-mem_sens[i])),
        "is_tf":       [g.lower() in MOUSE_TFS_LOWER for g in gene_names],
    }).sort_values("sensitivity", ascending=False).to_csv(
        os.path.join(OUT_DIR, f"multi_membership_{i}_sensitivity.csv"), index=False)

print(f"\nAll outputs -> {OUT_DIR}/")
