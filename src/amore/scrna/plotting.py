"""
amore.scrna.plotting — reusable plots for single-cell ISOKANN chi analyses.

  run_chi_umap          : run UMAP's layout directly on the chi memberships
  scatter_categorical   : 2-D embedding coloured by a discrete label (cell type)
  scatter_chi           : 2-D embedding coloured by a chi membership (inferno, 0..1)
  plot_chi_umaps        : one scatter_chi panel per membership
  driver_heatmap        : genes x chi-grid attribution heatmap (clustered, signed)
  tf_lineage_heatmap    : TF x cell-line signed driver heatmap
  plot_loss             : invariance loss (train + monitoring) curve

All functions take/return matplotlib Axes/Figures so a notebook can compose them.
Default DPI for saved figures is 600.
"""
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from umap.umap_ import simplicial_set_embedding, find_ab_params
CURATED_COLOR = "#00AB8E"
TF_COLOR = "#E8820C"
DPI = 600


def bhattacharyya(chi_a, chi_b):
    """BC(i,j) = sum_k sqrt(chi_ik chi_jk). Shape (B, B)."""
    return (chi_a.sqrt().unsqueeze(1) * chi_b.sqrt().unsqueeze(0)).sum(-1)

def imap_sgd(chi, *, min_dist=0.3, n_epochs=500,
              batch_size=1024, lr=1.0, random_state=0):
    """IMAP via direct SGD on cross-entropy between BC(chi) and low-dim kernel.
    No kNN graph, no bandwidth calibration."""
    torch.manual_seed(random_state)
    chi_t = torch.tensor(chi, dtype=torch.float32)
    chi_t = (chi_t.clamp(min=0) / chi_t.sum(1, keepdim=True))
    n = chi_t.shape[0]

    a, b = find_ab_params(1.0, min_dist)
    a, b = float(a), float(b)

    # spectral init: project first 2 PCs of chi
    _, _, V = torch.pca_lowrank(chi_t, q=2)
    z = (chi_t @ V).detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(n_epochs):
        idx = torch.randperm(n)[:batch_size]
        zi = z[idx]
        chi_i = chi_t[idx]

        w = bhattacharyya(chi_i, chi_i).clamp(0, 1)           # (B,B) high-dim
        d2 = ((zi.unsqueeze(1) - zi.unsqueeze(0))**2).sum(-1) # (B,B) low-dim
        v = 1.0 / (1.0 + a * d2**b)                           # UMAP kernel

        eps = 1e-6
        loss = -(w * (v + eps).log() + (1 - w) * (1 - v + eps).log()).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    return z.detach().numpy()



def chi_affinity_graph(chi, n_neighbors=30):
    """Sparse symmetric affinity on the membership simplex.

    Edge weight = Bhattacharyya coefficient BC(i,j) = sum_k sqrt(chi_ik chi_jk),
    in [0,1], = 1 iff identical. No per-point bandwidth (no rho/sigma).

    With R = sqrt(chi), every row has ||R_i||^2 = sum_k chi_ik = 1, so
    ||R_i - R_j||^2 = 2 - 2 BC(i,j). Euclidean kNN on sqrt(chi) is therefore
    exactly Hellinger kNN on chi, and BC = 1 - d^2/2 recovers the affinity.
    """
    chi = np.clip(np.asarray(chi, dtype=np.float64), 0.0, None)
    chi = chi / chi.sum(1, keepdims=True)
    R = np.sqrt(chi)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(R)
    d, idx = nn.kneighbors(R)
    bc = np.clip(1.0 - 0.5 * d**2, 0.0, 1.0)

    n = chi.shape[0]
    rows = np.repeat(np.arange(n), n_neighbors)
    A = sp.csr_matrix((bc.ravel(), (rows, idx.ravel())), shape=(n, n))
    G = A + A.T - A.multiply(A.T)          # UMAP's probabilistic t-conorm
    G.eliminate_zeros()
    return G


def run_chi_umap(chi, *, n_neighbors=30, min_dist=0.3, n_epochs=500,
                 random_state=0):
    """CUMAP: UMAP's cross-entropy layout driven directly by the chi simplex
    affinity, bypassing the kNN fuzzy-simplicial-set / bandwidth heuristic.
    Returns (N, 2). Drop-in replacement for the old PCA-graph version."""
    chi = np.asarray(chi, dtype=np.float64)
    G = chi_affinity_graph(chi, n_neighbors=n_neighbors)
    a, b = find_ab_params(1.0, min_dist)
    emb, _ = simplicial_set_embedding(
        data=chi, graph=G, n_components=2,
        initial_alpha=1.0, a=a, b=b, gamma=1.0,
        negative_sample_rate=5, n_epochs=n_epochs,
        init="spectral", random_state=check_random_state(random_state),
        metric="euclidean", metric_kwds={},
        densmap=False, densmap_kwds={}, output_dens=False, verbose=False,
    )
    return np.asarray(emb)


#def run_chi_umap(chi, *, n_neighbors=30, min_dist=0.3, random_state=0):
    """Run UMAP's cross-entropy layout on the chi memberships themselves
    (not the PCA graph). Returns (N, 2) embedding."""
 #   import umap
 #   return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
 #                    random_state=random_state).fit_transform(np.asarray(chi))


def scatter_categorical(ax, emb, labels, *, title="", s=4, cmap="tab10", legend=True):
    labels = np.asarray(labels)
    cats = np.unique(labels)
    cm = plt.get_cmap(cmap, len(cats))
    for j, lab in enumerate(cats):
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=s, color=cm(j), label=str(lab), linewidths=0)
    if legend:
        ax.legend(markerscale=3, fontsize=7, loc="center left",
                  bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return ax


def scatter_chi(ax, emb, values, *, title="", s=5, cmap="inferno", vmin=0.0, vmax=1.0):
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=s, c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("χ  (0 = progenitor → 1 = differentiated)", fontsize=7)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_chi_umaps(emb, chi, names=None, *, cmap="inferno", figsize_per=4.3):
    k = chi.shape[1]
    names = names or [f"χ_{i}" for i in range(k)]
    fig, axes = plt.subplots(1, k, figsize=(figsize_per * k, 4))
    axes = np.atleast_1d(axes)
    for i in range(k):
        scatter_chi(axes[i], emb, chi[:, i], title=names[i], cmap=cmap)
    fig.tight_layout()
    return fig


def driver_heatmap(ax, chi_col, per_cell_signed, genes_top, *,
                   n_grid=100, bw=0.03, tfs=None, curated=None, title=""):
    """
    Attribution along the differentiation axis. x = chi on a 100-pt grid (0->1,
    locally Gaussian-averaged, no binning blocks); y = the supplied top genes,
    hierarchically clustered (y only); colour = per-gene row-normalised signed
    attribution (red=+ pushes toward the lineage). `per_cell_signed` is
    (N, len(genes_top)) signed attribution (e.g. d chi/d gene per cell).
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    chi_col = np.asarray(chi_col); A = np.asarray(per_cell_signed)
    grid = np.linspace(0, 1, n_grid)
    prof = np.full((A.shape[1], n_grid), np.nan)
    for c, gp in enumerate(grid):
        w = np.exp(-0.5 * ((chi_col - gp) / bw) ** 2); sw = w.sum()
        if sw > 1e-3:
            prof[:, c] = (A * w[:, None]).sum(0) / sw
    profn = prof / (np.nanmax(np.abs(prof), 1, keepdims=True) + 1e-12)
    leaf = (leaves_list(linkage(np.nan_to_num(profn), method="ward"))
            if A.shape[1] > 2 else np.arange(A.shape[1]))
    profn = profn[leaf]; genes_top = np.asarray(genes_top)[leaf]
    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("lightgray")
    im = ax.imshow(np.ma.masked_invalid(profn), aspect="auto", cmap=cmap,
                   vmin=-1, vmax=1, extent=[0, 1, len(genes_top), 0], interpolation="nearest")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xlabel("χ  →  differentiation")
    ax.set_yticks(np.arange(len(genes_top)) + 0.5); ax.set_yticklabels(genes_top, fontsize=6)
    for tick, g in zip(ax.get_yticklabels(), genes_top):
        if curated and g in curated:
            tick.set_color(CURATED_COLOR); tick.set_fontweight("bold")
        elif tfs and g in tfs:
            tick.set_color(TF_COLOR)
    ax.set_title(title, fontsize=9)
    return im


def tf_lineage_heatmap(score_df, *, tfs=None, curated=None, top_per_col=8,
                       title="TF driver per cell line (signed)"):
    """
    score_df: DataFrame (genes x lineages) of signed driver scores. Restricts to
    TFs (if `tfs` given), takes the top per lineage, column-normalises, clusters
    rows, and draws a signed heatmap (red=+driver, blue=-). Returns (fig, ax).
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    df = score_df
    if tfs is not None:
        df = df.loc[[g for g in df.index if g in tfs]]
    top = set()
    for col in df.columns:
        top.update(df[col].abs().sort_values(ascending=False).index[:top_per_col])
    top = sorted(top)
    M = df.loc[top]
    Mn = M / (M.abs().max(0) + 1e-12)
    leaf = leaves_list(linkage(Mn.values, method="ward")) if len(top) > 2 else np.arange(len(top))
    Mn = Mn.iloc[leaf]
    fig, ax = plt.subplots(figsize=(0.9 * len(df.columns) + 3, max(5, 0.32 * len(Mn))))
    im = ax.imshow(Mn.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(Mn))); ax.set_yticklabels(Mn.index, fontsize=7)
    for tick, g in zip(ax.get_yticklabels(), Mn.index):
        if curated and g in curated:
            tick.set_color(CURATED_COLOR); tick.set_fontweight("bold")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.6, label="signed driver (col-normalised)")
    fig.tight_layout()
    return fig, ax


def _chi_bin_profile(chi_col, expr_top, nbins=25):
    """Mean expression of each gene per χ-bin over [0,1]. -> (genes, nbins), bin centres."""
    chi_col = np.asarray(chi_col); A = np.asarray(expr_top)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    binid = np.clip(np.digitize(chi_col, edges) - 1, 0, nbins - 1)
    prof = np.full((A.shape[1], nbins), np.nan)
    for b in range(nbins):
        m = binid == b
        if m.sum() >= 3:
            prof[:, b] = A[m].mean(0)
    return prof, 0.5 * (edges[:-1] + edges[1:])


def expression_heatmap(ax, chi_col, expr_top, genes_top, *, nbins=25, tfs=None,
                       curated=None, cluster=True, title="", cmap="viridis"):
    """Positive-only χ-resolved expression heatmap: genes × χ-bins, per-gene min-max
    normalised to [0,1] (viridis), genes hierarchically clustered (y only) — for
    positive lineage drivers (à la CR2). Green=curated marker, orange=TF on the axis."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    prof, _ = _chi_bin_profile(chi_col, expr_top, nbins)
    mn = np.nanmin(prof, 1, keepdims=True); mx = np.nanmax(prof, 1, keepdims=True)
    profn = (prof - mn) / (mx - mn + 1e-9)
    genes_top = np.asarray(genes_top)
    if cluster and len(genes_top) > 2:
        leaf = leaves_list(linkage(np.nan_to_num(profn), method="ward"))
        profn = profn[leaf]; genes_top = genes_top[leaf]
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("lightgray")
    im = ax.imshow(np.ma.masked_invalid(profn), aspect="auto", cmap=cm, vmin=0, vmax=1,
                   extent=[0, 1, len(genes_top), 0], interpolation="nearest")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xlabel("χ  →  differentiation")
    ax.set_yticks(np.arange(len(genes_top)) + 0.5); ax.set_yticklabels(genes_top, fontsize=6)
    for tick, g in zip(ax.get_yticklabels(), genes_top):
        if curated and g in curated:
            tick.set_color(CURATED_COLOR); tick.set_fontweight("bold")
        elif tfs and g in tfs:
            tick.set_color(TF_COLOR)
    ax.set_title(title, fontsize=9)
    return im


def expression_profiles(ax, chi_col, expr_top, genes_top, *, nbins=25, title=""):
    """Line profiles of top genes' per-gene min-max-normalised mean expression along χ
    (the commitment axis the MEP traces)."""
    prof, centres = _chi_bin_profile(chi_col, expr_top, nbins)
    mn = np.nanmin(prof, 1, keepdims=True); mx = np.nanmax(prof, 1, keepdims=True)
    profn = (prof - mn) / (mx - mn + 1e-9)
    cmap = plt.get_cmap("tab10", max(len(genes_top), 1))
    for j, g in enumerate(genes_top):
        ax.plot(centres, profn[j], lw=1.4, color=cmap(j % 10), label=g)
    ax.set_xlabel("χ  →  differentiation"); ax.set_ylabel("norm. expression")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    return ax


def plot_loss(ax, loss_train, loss_monitor=None, *, title="ISOKANN invariance loss"):
    ax.plot(loss_train, lw=1.5, label="train operator (ISA-target)")
    if loss_monitor is not None:
        ax.plot(loss_monitor, lw=1.5, alpha=0.8, label="held-out transitions (monitor)")
    ax.set_yscale("log"); ax.set_xlabel("ISA outer iteration")
    ax.set_ylabel("invariance loss (MSE)"); ax.set_title(title); ax.legend()
    return ax
