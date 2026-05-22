"""
Post-processing for multi-D ISOKANN on LARRY.

The argmax of raw sigmoid chi functions is NOT the correct state assignment —
it just picks whichever chi happens to be highest at a given cell, which is
dominated by the function with the highest baseline.

The correct interpretation:
  1. Chi-space geometry: cells near a simplex vertex are in that state
  2. PCCA+ rotation: find the optimal rotation A that maps chi to a proper
     membership matrix (rows sum ~1, each column peaks at one metastable state)
  3. Chi-UMAP: embed the chi vectors in 2D and color by known states

PCCA+ rotation (simplified vertex-based):
  - Find k "extreme" cells (approximate simplex vertices) via successive
    max-distance selection (Schur vector method from MSM literature)
  - Define A = chi[vertices]^{-1}  (rotation to vertex coordinates)
  - Membership = chi @ A  (softmax-normalised)

References
----------
Roblitz & Weber (2013) Adv. Data Anal. Classif. — PCCA+
Deuflhard & Weber (2005) Linear Algebra Appl. — original PCCA
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading ...")
adata    = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
chi_all  = np.load(os.path.join(OUT_DIR, "multi_chi_all.npy"))     # (49116, 15)
abs_evals = np.load(os.path.join(OUT_DIR, "multi_eigenvalues.npy"))

states   = adata.obs["state_info"].astype(str).values
times    = adata.obs["time_info"].astype(str).values
emb      = adata.obsm["X_umap"]
n_cells  = len(chi_all)
k_max    = chi_all.shape[1]

# Spectral gap selection — override manually to use user-identified gap
K_OVERRIDE = 13   # set to None for automatic largest-gap detection
gaps       = abs_evals[:-1] - abs_evals[1:]
k_auto     = int(np.argmax(gaps)) + 1
k_correct  = K_OVERRIDE if K_OVERRIDE is not None else k_auto
print(f"  k_max={k_max}  k_correct={k_correct}  "
      f"(override={K_OVERRIDE}  auto-detect={k_auto}  gap@k={k_correct}: {gaps[k_correct-1]:.4f})")

# Sort chi functions by eigenvalue magnitude (highest first)
order       = np.argsort(-abs_evals)
chi_ordered = chi_all[:, order[:k_correct]]   # (n_cells, k_correct)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PCCA+ vertex selection and rotation
# ══════════════════════════════════════════════════════════════════════════════

def pcca_rotation(chi_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simplified PCCA+ rotation via successive max-distance vertex selection.

    Algorithm (Schur-vector based simplex extremes):
      1. Start from the cell furthest from the mean in chi-space
      2. Iteratively pick the cell maximally distant from all current vertices
      3. k vertices define the simplex corners
      4. Rotation A = chi[vertices]^{-1}  (maps simplex to standard basis)
      5. Membership = chi @ A, clipped to [0,1] and row-normalised

    Parameters
    ----------
    chi_mat : (n, k)

    Returns
    -------
    membership : (n, k)  — rows in [0,1], approximately sum to 1
    vertex_idx : (k,)    — cell indices of simplex corners
    """
    n, k = chi_mat.shape

    # Normalise each row to the unit simplex approximation
    chi_n = chi_mat - chi_mat.min(0)
    chi_n = chi_n / (chi_n.sum(1, keepdims=True) + 1e-8)

    # Successive max-distance selection
    vertex_idx = [int(np.argmax(np.linalg.norm(chi_n - chi_n.mean(0), axis=1)))]
    for _ in range(k - 1):
        dists = np.array([
            min(np.linalg.norm(chi_n - chi_n[v], axis=1).min()
                for v in vertex_idx)
        ] if False else
            np.min(np.stack([np.linalg.norm(chi_n - chi_n[v], axis=1)
                              for v in vertex_idx]), axis=0)
        )
        vertex_idx.append(int(np.argmax(dists)))

    vertex_idx = np.array(vertex_idx)
    C = chi_n[vertex_idx]               # (k, k) — chi at vertices

    # Rotation: membership = chi_n @ C^{-1}
    try:
        A          = np.linalg.inv(C)
        membership = chi_n @ A
        # Clip and renormalise rows (small negative values arise from noise)
        membership = np.clip(membership, 0, None)
        membership = membership / (membership.sum(1, keepdims=True) + 1e-8)
    except np.linalg.LinAlgError:
        membership = chi_n  # fallback

    return membership, vertex_idx


print("\nApplying PCCA+ rotation ...")
membership, vertex_idx = pcca_rotation(chi_ordered)
print(f"  Vertex cells: {vertex_idx}")
for i, vi in enumerate(vertex_idx):
    print(f"    vertex {i}: cell={vi}  state={states[vi]}  "
          f"chi=[{', '.join(f'{v:.2f}' for v in chi_ordered[vi])}]")

# Hard assignment from membership
hard_assign = np.argmax(membership, axis=1)

state_cats = sorted(set(states))
print(f"\n-- PCCA+ cluster -> cell state mapping --")
assignment = {}
for k in range(k_correct):
    mask   = hard_assign == k
    counts = Counter(states[mask])
    top3   = counts.most_common(3)
    dominant = top3[0][0] if top3 else "?"
    assignment[k] = dominant
    top3_str = ", ".join(f"{s}({c})" for s, c in top3)
    print(f"  cluster {k}  n={mask.sum():>6,}  top: {top3_str}")

state_to_int = {s: i for i, s in enumerate(state_cats)}
states_int   = np.array([state_to_int[s] for s in states])
ari_pcca     = adjusted_rand_score(states_int, hard_assign)
print(f"\n  ARI (PCCA+ membership argmax vs cell state): {ari_pcca:.4f}")

np.save(os.path.join(OUT_DIR, "multi_membership.npy"), membership)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Chi-space UMAP (embed the k_correct chi vectors in 2D)
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing chi-space UMAP ...")
try:
    import umap
    reducer   = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    chi_umap  = reducer.fit_transform(chi_ordered)
    has_umap  = True
    print(f"  Chi-UMAP shape: {chi_umap.shape}")
except ImportError:
    print("  umap-learn not installed, using PCA of chi as fallback")
    from sklearn.decomposition import PCA
    chi_umap  = PCA(n_components=2, random_state=42).fit_transform(chi_ordered)
    has_umap  = False


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Plots
# ══════════════════════════════════════════════════════════════════════════════

state_cmap = plt.get_cmap("tab20", len(state_cats))
cluster_cmap = plt.get_cmap("tab10", k_correct)

# A. Chi-space projection coloured by cell state
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i, s in enumerate(state_cats):
    mask = states == s
    ax.scatter(chi_umap[mask,0], chi_umap[mask,1], c=[state_cmap(i)],
               s=1, alpha=0.5, label=s, rasterized=True)
ax.legend(fontsize=5, markerscale=6, ncol=2)
title = "chi-space UMAP" if has_umap else "chi-space PCA"
ax.set_title(f"{title} coloured by cell state"); ax.set_xticks([]); ax.set_yticks([])

# B. Chi-space projection coloured by PCCA+ cluster
ax = axes[1]
for k in range(k_correct):
    mask = hard_assign == k
    dom  = assignment.get(k, "?")
    ax.scatter(chi_umap[mask,0], chi_umap[mask,1], c=[cluster_cmap(k)],
               s=1, alpha=0.5, label=f"k{k}:{dom}", rasterized=True)
ax.legend(fontsize=6, markerscale=5, ncol=2)
ax.set_title(f"PCCA+ clusters (k={k_correct})  ARI={ari_pcca:.3f}")
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(f"Multi-D ISOKANN chi-space geometry  (k_correct={k_correct})", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_chispace.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_chispace.png")


# B. Membership per function on the UMAP embedding
fig, axes = plt.subplots(2, (k_correct+1)//2, figsize=(k_correct*2, 8))
axes = np.array(axes).flatten()
for i in range(k_correct):
    ax   = axes[i]
    vals = membership[:, i]
    sc_  = ax.scatter(emb[:,0], emb[:,1], c=vals, cmap="Blues",
                      vmin=0, vmax=1, s=1, alpha=0.5, rasterized=True)
    plt.colorbar(sc_, ax=ax, shrink=0.7)
    dom  = assignment.get(i, "?")
    ax.set_title(f"Membership {i} ({dom})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

for j in range(k_correct, len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f"PCCA+ membership functions on UMAP (k={k_correct})", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_membership_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_membership_umap.png")


# C. Membership distribution per cell state (are states enriched in clusters?)
fig, ax = plt.subplots(figsize=(12, 4))
for i in range(k_correct):
    means = [membership[states==s, i].mean() for s in state_cats]
    ax.plot(range(len(state_cats)), means, "o-", lw=1.5, ms=4,
            label=f"membership_{i} ({assignment.get(i,'?')})")
ax.set_xticks(range(len(state_cats)))
ax.set_xticklabels(state_cats, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean membership")
ax.set_title(f"PCCA+ membership per cell state (k={k_correct})")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "multi_membership_profile.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: multi_membership_profile.png")


# D. Chi pair-scatter for the first 3 components (simplex triangle)
if k_correct >= 3:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pairs = [(0,1),(0,2),(1,2)]
    for ax, (i,j) in zip(axes, pairs):
        for si, s in enumerate(state_cats):
            mask = states == s
            ax.scatter(membership[mask,i], membership[mask,j],
                       c=[state_cmap(si)], s=1, alpha=0.3, rasterized=True, label=s)
        ax.set_xlabel(f"membership_{i}"); ax.set_ylabel(f"membership_{j}")
        ax.set_title(f"m{i} vs m{j}")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
    axes[0].legend(fontsize=5, markerscale=5, ncol=2)
    plt.suptitle("Simplex face scatter (membership components)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "multi_simplex_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: multi_simplex_scatter.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n-- Summary --")
print(f"  Spectral gap selects k={k_correct}")
print(f"  Eigenvalues (top-{k_correct}): {abs_evals[:k_correct].round(4)}")
print(f"  ARI (raw argmax): {adjusted_rand_score(states_int, np.argmax(chi_ordered,1)):.4f}")
print(f"  ARI (PCCA+ ):     {ari_pcca:.4f}")
print(f"\n  Cell-state enrichment per PCCA+ cluster:")
for k in range(k_correct):
    mask   = hard_assign == k
    counts = Counter(states[mask])
    top    = counts.most_common(4)
    print(f"    cluster {k}: " + ", ".join(f"{s}({c})" for s,c in top))
