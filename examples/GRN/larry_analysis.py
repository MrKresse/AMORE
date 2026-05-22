"""
Post-ISOKANN analysis for LARRY hematopoiesis - v3.

Key fixes vs v2:
  - Fate-bias validation restricted to day-2 progenitor cells (where bias is defined)
  - Gene loadings recovered by linear regression: solve X_pca = X_expr @ W
  - Chi bimodality diagnosis: histogram + KDE coloured by lineage
  - Pair quality check: what fraction of pairs bridge different cell states?
"""

import os, sys
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr, spearmanr
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from collections import Counter
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

N_PCS  = 40
HIDDEN = [256, 128, 64]

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

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading ...")
adata    = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
X_pca    = np.load(os.path.join(DATA_DIR, "larry_pca.npy"))
chi_vals = np.load(os.path.join(OUT_DIR,  "larry_chi_vals.npy"))
dchi_dpc = np.load(os.path.join(OUT_DIR,  "larry_dchi_dx.npy"))
src      = np.load(os.path.join(DATA_DIR, "larry_src.npy"))
dst      = np.load(os.path.join(DATA_DIR, "larry_dst.npy"))

obs        = adata.obs.copy()
emb        = adata.obsm["X_umap"]
gene_names = np.array(adata.var_names)
states     = obs["state_info"].astype(str).values
times      = obs["time_info"].astype(str).values
n_cells    = len(chi_vals)

X_expr = adata.X
if hasattr(X_expr, "toarray"):
    X_expr = X_expr.toarray()
X_expr = np.array(X_expr, dtype=np.float32)

norm = Normalize(0, 1)
print(f"  {n_cells:,} cells | chi=[{chi_vals.min():.3f},{chi_vals.max():.3f}]  std={chi_vals.std():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Chi bimodality diagnosis
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- 1. Chi bimodality --")

cats_ordered = sorted(set(states), key=lambda s: np.median(chi_vals[states==s]))
cmap_s = plt.get_cmap("tab20", len(cats_ordered))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Overall histogram
ax = axes[0]
ax.hist(chi_vals, bins=60, color="steelblue", edgecolor="none", density=True)
kde = gaussian_kde(chi_vals, bw_method=0.1)
xs  = np.linspace(0, 1, 300)
ax.plot(xs, kde(xs), "r-", lw=2)
ax.set_xlabel("chi"); ax.set_ylabel("density")
ax.set_title(f"Chi distribution  std={chi_vals.std():.3f}")

# Per-lineage KDE
ax = axes[1]
for i, s in enumerate(cats_ordered):
    vals = chi_vals[states == s]
    if len(vals) < 20: continue
    kde_s = gaussian_kde(vals, bw_method=0.15)
    ax.plot(xs, kde_s(xs), color=cmap_s(i), lw=1.5, label=f"{s} (n={len(vals):,})")
ax.set_xlabel("chi"); ax.set_ylabel("density")
ax.set_title("Chi KDE by cell state")
ax.legend(fontsize=5, ncol=2)

# Embedding coloured by chi
ax = axes[2]
sc_ = ax.scatter(emb[:,0], emb[:,1], c=chi_vals, cmap="coolwarm", norm=norm,
                 s=1, alpha=0.4, rasterized=True)
plt.colorbar(sc_, ax=ax, label="chi", shrink=0.8)
ax.set_title("Chi on embedding"); ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
savefig(fig, "analysis_01_chi_bimodality.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Pair quality: do pairs bridge different cell states?
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- 2. Pair quality check --")

src_states = states[src]
dst_states = states[dst]
same_state = src_states == dst_states
cross_state = ~same_state

print(f"  Total pairs: {len(src):,}")
print(f"  Same-state pairs:  {same_state.sum():,} ({100*same_state.mean():.1f}%)")
print(f"  Cross-state pairs: {cross_state.sum():,} ({100*cross_state.mean():.1f}%)")

# Chi change per pair
delta_chi = chi_vals[dst] - chi_vals[src]
print(f"  Mean |delta_chi| per pair: {np.abs(delta_chi).mean():.4f}")
print(f"  Pairs where chi increases: {(delta_chi>0).mean()*100:.1f}%")

# Transition matrix: what states pair to what states?
pair_df = pd.DataFrame({"src_state": src_states, "dst_state": dst_states})
transition = pd.crosstab(pair_df["src_state"], pair_df["dst_state"])
print("\n  Pair transition counts (src day2 -> dst day4):")
print(transition.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(delta_chi, bins=50, color="steelblue", edgecolor="none")
ax.axvline(0, color="red", lw=1.5, ls="--")
ax.set_xlabel("chi(dst) - chi(src)")
ax.set_ylabel("pair count")
ax.set_title(f"Chi change per clone pair  mean|delta|={np.abs(delta_chi).mean():.4f}")

ax = axes[1]
im = ax.imshow(transition.values, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(transition.columns)))
ax.set_yticks(range(len(transition.index)))
ax.set_xticklabels(transition.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(transition.index, fontsize=7)
ax.set_xlabel("day-4 state"); ax.set_ylabel("day-2 state")
ax.set_title("Clone pair transition matrix")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
savefig(fig, "analysis_02_pair_quality.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Fate-bias validation — restricted to day-2 undiff progenitors
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- 3. Fate-bias validation (day-2 progenitors only) --")

day2_mask = times == "2"
print(f"  Day-2 cells: {day2_mask.sum():,}  (undiff: {((times=='2')&(states=='undiff')).sum():,})")

fate_cols = [c for c in obs.columns if c.startswith("progenitor_")]

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()
ax_idx = 0

results = {}
for col in fate_cols + ["NeuMon_fate_bias"]:
    if col not in obs.columns: continue
    bias = pd.to_numeric(obs[col], errors="coerce").values.astype(float)

    # Use all day-2 cells
    mask = day2_mask & ~np.isnan(bias)
    if mask.sum() < 20:
        continue

    r_s, p_s = spearmanr(chi_vals[mask], bias[mask])
    results[col] = r_s
    lineage = col.replace("progenitor_", "").replace("_fate_bias", "")
    print(f"  {lineage:<20s}  Spearman(chi, fate_bias) = {r_s:+.3f}  (n={mask.sum():,})")

    if ax_idx < len(axes):
        ax = axes[ax_idx]; ax_idx += 1
        sc_ = ax.scatter(chi_vals[mask], bias[mask], s=3, alpha=0.4,
                         c=chi_vals[mask], cmap="coolwarm", rasterized=True)
        ax.set_xlabel("chi", fontsize=7); ax.set_ylabel("fate bias", fontsize=7)
        ax.set_title(f"{lineage}  rho={r_s:+.2f}", fontsize=8)

for j in range(ax_idx, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Chi vs lineage fate bias (day-2 cells only)", fontsize=11)
plt.tight_layout()
savefig(fig, "analysis_03_chi_vs_fatebias_day2.png")

# Summary
best_pos = max(results.items(), key=lambda x: x[1]) if results else ("?", 0)
best_neg = min(results.items(), key=lambda x: x[1]) if results else ("?", 0)
print(f"\n  chi=1 corresponds to: {best_pos[0]}  (rho={best_pos[1]:+.3f})")
print(f"  chi=0 corresponds to: {best_neg[0]}  (rho={best_neg[1]:+.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Gene sensitivity via linear regression on PCA
#     Solve: X_pca ~ X_expr @ W  (recovers effective gene->PC mapping)
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- 4. Gene sensitivity via regression loadings --")

# Recover effective loadings W: X_pca ~ X_centered @ W
# Use Ridge regression on a random subset for speed
print(f"  Fitting gene->PC mapping via Ridge regression ...")
n_sub = min(8000, n_cells)
rng   = np.random.default_rng(42)
idx_sub = rng.choice(n_cells, n_sub, replace=False)
X_sub   = X_expr[idx_sub].astype(np.float64)
P_sub   = X_pca[idx_sub].astype(np.float64)

# Center
X_mean  = X_sub.mean(0)
X_c     = X_sub - X_mean

# Ridge: P ~ X_c @ W,  solve for W (n_genes, N_PCS)
alpha   = 1.0
W       = np.linalg.lstsq(X_c.T @ X_c + alpha*np.eye(X_c.shape[1]),
                           X_c.T @ P_sub, rcond=None)[0]  # (n_genes, N_PCS)

# Verify alignment
P_pred  = X_c @ W
corrs   = [np.corrcoef(P_pred[:,k], P_sub[:,k])[0,1] for k in range(N_PCS)]
print(f"  Alignment (mean |r| over {N_PCS} PCs): {np.mean(np.abs(corrs)):.3f}")
print(f"  PC1 r={corrs[0]:.3f}  PC2 r={corrs[1]:.3f}  PC3 r={corrs[2]:.3f}")

# Gene sensitivity = |dchi/dPC| @ |W|^T  (summed over PCs)
mean_dpc    = np.abs(dchi_dpc).mean(0)            # (N_PCS,)
gene_sens   = np.abs(W) @ mean_dpc                 # (n_genes,)

# Transition-state sensitivity
ts_mask     = (chi_vals >= 0.4) & (chi_vals <= 0.6)
mean_dpc_ts = np.abs(dchi_dpc[ts_mask]).mean(0)
gene_sens_ts = np.abs(W) @ mean_dpc_ts

top_idx     = np.argsort(gene_sens)[::-1]
top_genes   = gene_names[top_idx[:50]]
top_scores  = gene_sens[top_idx[:50]]

top_ts_idx    = np.argsort(gene_sens_ts)[::-1]
top_ts_genes  = gene_names[top_ts_idx[:50]]

print(f"\n  Top-20 sensitivity genes (all cells):")
for g, s in zip(top_genes[:20], top_scores[:20]):
    print(f"    {g:<20s}  {s:.5f}")

# Known markers
known_markers = {
    "erythroid": ["Gata1","Klf1","Epor","Hba-a1","Hbb-bt","Gypa","Zfpm1","Tal1","Nfe2"],
    "myeloid":   ["Spi1","Csf1r","Cebpa","Cebpb","Mpo","Elane","Gfi1","Irf8","Irf4"],
    "stem":      ["Sca1","Kit","Gata2","Runx1","Flt3","Cd34","Mpl","Erg"],
    "baso/mast": ["Gata2","Mcpt8","Fcer1a","Prss34","Cpa3","Hdc"],
    "meg":       ["Pf4","Vwf","Gp1ba","Nfe2","Itga2b","Mpl"],
}
all_known = {g.lower() for ms in known_markers.values() for g in ms}
gene_lower = {g.lower(): g for g in gene_names}

print("\n  Known-marker recovery (top-100):")
for lineage, markers in known_markers.items():
    found = []
    for m in markers:
        actual = gene_lower.get(m.lower())
        if actual:
            rank = np.where(gene_names[top_idx] == actual)[0]
            if len(rank) and rank[0] < 100:
                found.append(f"{actual}(#{rank[0]+1})")
    tag = found if found else f"(none of {markers[:3]})"
    print(f"    {lineage:<12s}  {tag}")

# Save table
pd.DataFrame({
    "gene": gene_names,
    "sens_all": gene_sens,
    "rank_all": np.argsort(np.argsort(-gene_sens)),
    "sens_transition": gene_sens_ts,
    "rank_transition": np.argsort(np.argsort(-gene_sens_ts)),
}).sort_values("sens_all", ascending=False).to_csv(
    os.path.join(OUT_DIR, "larry_gene_sensitivity.csv"), index=False)

# Plots
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
for ax, (top_g, scores, label) in zip(axes, [
    (top_genes[:40],    top_scores[:40],               "All cells"),
    (top_ts_genes[:40], gene_sens_ts[top_ts_idx[:40]], "Transition zone (chi in [0.4,0.6])"),
]):
    cols = ["crimson" if g.lower() in all_known else "steelblue" for g in top_g]
    ax.bar(range(40), scores, color=cols)
    ax.set_xticks(range(40))
    ax.set_xticklabels(top_g, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("|dchi/dgene|")
    ax.set_title(f"Top-40 genes — {label}  (red=known TF/marker)")

plt.tight_layout()
savefig(fig, "analysis_04_gene_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Top genes vs chi axis
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- 5. Top genes along chi axis --")

top10 = gene_names[top_ts_idx[:10]]
chi_bins = np.linspace(chi_vals.min(), chi_vals.max(), 25)
chi_mid  = 0.5*(chi_bins[:-1]+chi_bins[1:])

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for ax, g in zip(axes.flatten(), top10):
    if g not in adata.var_names:
        ax.set_visible(False); continue
    gi   = list(adata.var_names).index(g)
    expr = X_expr[:, gi]
    bins = []
    for lo, hi in zip(chi_bins[:-1], chi_bins[1:]):
        sel = expr[(chi_vals>=lo) & (chi_vals<hi)]
        bins.append(sel.mean() if len(sel) > 0 else np.nan)
    ax.plot(chi_mid, bins, "o-", ms=3, lw=1.5)
    r_s, _ = spearmanr(chi_vals, expr)
    ax.set_title(f"{g}  rho={r_s:.2f}", fontsize=8)
    ax.set_xlabel("chi", fontsize=7); ax.set_ylabel("mean expr", fontsize=7)

plt.suptitle("Top-10 transition drivers vs chi axis", fontsize=10)
plt.tight_layout()
savefig(fig, "analysis_05_genes_vs_chi.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Lineage-specific chi on embedding
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- 6. Lineage maps --")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
key_states = ["Erythroid","Baso","Mast","Meg","Neutrophil","Monocyte","Lymphoid","undiff"]

for ax, s in zip(axes, key_states):
    m = states == s
    ax.scatter(emb[~m,0], emb[~m,1], c="lightgray", s=0.5, alpha=0.15, rasterized=True)
    if m.sum() > 0:
        sc_ = ax.scatter(emb[m,0], emb[m,1], c=chi_vals[m], cmap="coolwarm", norm=norm,
                         s=2, alpha=0.7, rasterized=True)
        plt.colorbar(sc_, ax=ax, shrink=0.7)
    ax.set_title(f"{s}  n={m.sum():,}  chi_med={np.median(chi_vals[m]):.3f}")
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Chi within each lineage", fontsize=11)
plt.tight_layout()
savefig(fig, "analysis_06_lineage_chi.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- Summary --")
print(f"  chi range: [{chi_vals.min():.3f}, {chi_vals.max():.3f}]  std={chi_vals.std():.3f}")
low  = (chi_vals < 0.2).sum(); high = (chi_vals > 0.8).sum()
print(f"  chi<0.2: {low:,}  chi>0.8: {high:,}  chi in [0.4,0.6]: {ts_mask.sum():,}")
print(f"  Mean |delta_chi| per pair: {np.abs(delta_chi).mean():.4f}")
if results:
    print(f"  Strongest chi=1 lineage: {best_pos[0]}  rho={best_pos[1]:+.3f}")
    print(f"  Strongest chi=0 lineage: {best_neg[0]}  rho={best_neg[1]:+.3f}")
print(f"  Top transition driver: {top_ts_genes[0]}")
print(f"\nAll plots -> {OUT_DIR}/")
