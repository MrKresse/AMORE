# -*- coding: utf-8 -*-
"""
analysis.py — simplex-edge analysis, rare-edge detection, and zeroth-order pathway export
for the 2cm2 multi-D ISOKANN example.

Given the k membership functions chi (N, k) on the simplex Delta^{k-1}, the pairwise
interconversions are the simplex **edges** (i,j).  For each edge we use the edge coordinate

    s_ij = 1/2 (chi_i - chi_j + 1)   in [0,1]      (s=0 -> vertex j, s=1 -> vertex i)

A frame lies "on" edge (i,j) when chi_i + chi_j is close to 1 (the other memberships vanish);
it is a genuine transition-state frame when in addition s_ij is intermediate (neither vertex).
Edges with few such transition frames are the **rare** simplex edges.

The zeroth-order transition pathway for an edge is simply the on-edge frames **ordered by
s_ij** — a reordering of existing trajectory frames from vertex j to vertex i, written out as a
DCD (+ PDB topology) for visualisation.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src")))


# ── edge coordinate & populations ────────────────────────────────────────────
def edge_coord(chi, i, j):
    """s_ij = 1/2 (chi_i - chi_j + 1)  in [0,1]."""
    chi = np.asarray(chi)
    return 0.5 * (chi[:, i] - chi[:, j] + 1.0)


def state_populations(chi, tau_vertex=0.8):
    """Per-vertex population = # frames with chi_i >= tau_vertex (state i actually visited)."""
    chi = np.asarray(chi)
    return {i: int((chi[:, i] >= tau_vertex).sum()) for i in range(chi.shape[1])}


def edge_table(chi, tau_edge=0.8, s_lo=0.15, s_hi=0.85, tau_vertex=0.8):
    """Per-edge occupancy statistics.

    For every pair (i,j):
      on_edge     : # frames with chi_i + chi_j >= tau_edge  (near the i-j edge)
      transition  : # on-edge frames with s_lo <= s_ij <= s_hi  (genuine transition states)
      pop_i, pop_j: vertex populations of the two endpoints
      relevant    : both endpoints are visited (pop_i>0 and pop_j>0)

    Returns a list of dicts, one per edge, sorted by ascending `transition` count.
    """
    chi = np.asarray(chi)
    k = chi.shape[1]
    vpop = state_populations(chi, tau_vertex)
    rows = []
    for i, j in combinations(range(k), 2):
        onedge = chi[:, i] + chi[:, j] >= tau_edge
        s = edge_coord(chi, i, j)
        trans = onedge & (s >= s_lo) & (s <= s_hi)
        rows.append(dict(edge=(i, j), on_edge=int(onedge.sum()),
                         transition=int(trans.sum()),
                         pop_i=vpop[i], pop_j=vpop[j],
                         relevant=bool(vpop[i] > 0 and vpop[j] > 0)))
    rows.sort(key=lambda r: r["transition"])
    return rows


def rare_edges(rows, rel_frac=0.2, abs_max=None):
    """Flag the rare simplex edges among the *relevant* ones.

    "Rare" = rarely traversed relative to the busiest interconversion: an edge is rare if its
    transition count is below `rel_frac` * (MAX transition count over the relevant edges), or
    below `abs_max` if that is given.  The max (not the median) is the reference because with
    only a few edges the median is unstable — when one edge dominates, the ~20x-rarer edges
    are exactly the ones we want flagged.  Returns the list of rare edge tuples.
    """
    rel = [r for r in rows if r["relevant"]]
    if not rel:
        return []
    ref = max(r["transition"] for r in rel)
    if ref == 0:
        return []
    thr = abs_max if abs_max is not None else max(1.0, rel_frac * ref)
    return [r["edge"] for r in rel if r["transition"] < thr]


def format_edge_table(rows, rare):
    lines = [f"{'edge':>7} {'on_edge':>8} {'transition':>11} {'pop_i':>6} {'pop_j':>6} "
             f"{'relevant':>9} {'rare':>5}"]
    for r in rows:
        e = r["edge"]
        lines.append(f"{str(e[0])+'-'+str(e[1]):>7} {r['on_edge']:>8} {r['transition']:>11} "
                     f"{r['pop_i']:>6} {r['pop_j']:>6} {str(r['relevant']):>9} "
                     f"{'YES' if e in rare else '':>5}")
    return "\n".join(lines)


# ── zeroth-order pathway (frames ordered along the edge coordinate) ───────────
def pathway_frames(chi, i, j, tau_edge=0.8):
    """On-edge frame indices ordered by s_ij (vertex j -> vertex i).  Returns (order, s_sorted)
    where `order` indexes into the anchor/chi array."""
    chi = np.asarray(chi)
    onedge = np.where(chi[:, i] + chi[:, j] >= tau_edge)[0]
    if len(onedge) == 0:
        return np.array([], int), np.array([])
    s = edge_coord(chi, i, j)[onedge]
    o = np.argsort(s)
    return onedge[o], s[o]


def windowed_edges(chi, i, j, n_windows=20, tau_edge=0.8):
    """Bin the real on-edge frames for edge (i,j) into `n_windows` contiguous, equal-
    POPULATION windows along s_ij (not equal-width in s -- real dwell time isn't uniform in
    s, so equal-width bins can end up empty near the edges while equal-population bins are
    always usable). This is the partitioning step for an a-posteriori "string": pick one
    representative real frame per window (e.g. its aligned-RMSD medoid) instead of running
    any simulation.

    Returns
    -------
    windows : list of length n_windows, each an np.ndarray of anchor indices (same indexing
        as `pathway_frames`'s `order` -- NOT yet offset by NSTART)
    window_s : np.ndarray, shape (n_windows,) -- mean s_ij of each window's frames
    """
    order, s = pathway_frames(chi, i, j, tau_edge=tau_edge)
    n = len(order)
    if n < n_windows:
        raise ValueError(f"edge {i}-{j}: only {n} on-edge frames, need >= {n_windows}")
    bounds = np.linspace(0, n, n_windows + 1).round().astype(int)
    windows = [order[bounds[w]:bounds[w + 1]] for w in range(n_windows)]
    window_s = np.array([s[bounds[w]:bounds[w + 1]].mean() for w in range(n_windows)])
    return windows, window_s


def export_pathway(order, nstart, pdb_file, dcd_file, out_dcd, out_pdb,
                   selection="protein"):
    """Write the ordered trajectory frames to `out_dcd` (+ topology `out_pdb`).

    `order` are anchor indices; trajectory frame = nstart + anchor index.  Only `selection`
    atoms are written so the file is small and self-contained.  Returns the number of frames.
    """
    import MDAnalysis as mda
    if len(order) == 0:
        return 0
    u = mda.Universe(pdb_file, dcd_file)
    sel = u.select_atoms(selection)
    os.makedirs(os.path.dirname(out_dcd), exist_ok=True)
    u.trajectory[int(nstart + order[0])]
    sel.write(out_pdb)                                   # topology snapshot
    with mda.Writer(out_dcd, sel.n_atoms) as W:
        for a in order:
            u.trajectory[int(nstart + a)]
            W.write(sel)
    return len(order)


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_loss(res, ax=None):
    ax = ax or plt.gca()
    lt = np.asarray(res["loss_train"], float)
    lv = np.asarray(res["loss_val"], float)
    ax.plot(lt, label="train (MSE to ISA target)")
    ax.plot(lv, label="val (Gram-Schmidt residual)")
    ax.set_yscale("log"); ax.set_xlabel("ISA power iteration"); ax.set_ylabel("loss")
    ax.set_title(f"k={res['k']} training"); ax.legend(); ax.grid(alpha=0.3)
    return ax


def plot_chi_trajectory(chi, nstart, ax=None):
    chi = np.asarray(chi)
    ax = ax or plt.gca()
    frames = np.arange(chi.shape[0]) + nstart
    for i in range(chi.shape[1]):
        ax.plot(frames, chi[:, i], lw=0.8, label=f"$\\chi_{{{i}}}$")
    ax.set_xlabel("trajectory frame"); ax.set_ylabel("membership $\\chi_i$")
    ax.set_ylim(-0.02, 1.02); ax.set_title("memberships along the trajectory")
    ax.legend(ncol=min(chi.shape[1], 6), fontsize=8); ax.grid(alpha=0.3)
    return ax


def plot_edge_populations(rows, rare, ax=None):
    ax = ax or plt.gca()
    labels = [f"{r['edge'][0]}-{r['edge'][1]}" for r in rows]
    vals = [r["transition"] for r in rows]
    colors = ["crimson" if r["edge"] in rare else ("steelblue" if r["relevant"] else "0.7")
              for r in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("# transition frames"); ax.set_title("simplex-edge transition population\n"
                  "(red = rare relevant edge, grey = endpoint state unvisited)")
    ax.grid(axis="y", alpha=0.3)
    return ax


def chi_umap(chi, min_dist=0.3, n_epochs=500, seed=0):
    """IMAP (Invariant Manifold Approximation and Projection) layout of the memberships:
    `amore.scrna.plotting.imap_sgd` — direct SGD on the cross-entropy between the
    Bhattacharyya chi-affinity and the low-dim UMAP kernel, with NO kNN graph and NO bandwidth
    calibration (fewer hyperparameters than UMAP/CUMAP).  Returns (N, 2)."""
    from amore.scrna.plotting import imap_sgd
    return imap_sgd(np.asarray(chi), min_dist=min_dist, n_epochs=n_epochs, random_state=seed)


def plot_chi_umap(chi, emb=None, seed=0):
    """UMAP layout of the memberships (as in src/amore/scrna/plotting.py): the embedding
    coloured by the dominant state, plus one panel per membership χ_i (inferno, 0..1).
    Returns (fig, emb)."""
    from amore.scrna.plotting import scatter_categorical, scatter_chi
    chi = np.asarray(chi)
    k = chi.shape[1]
    if emb is None:
        emb = chi_umap(chi, seed=seed)
    fig, axes = plt.subplots(1, k + 1, figsize=(4.3 * (k + 1), 4))
    scatter_categorical(axes[0], emb, chi.argmax(1), title="dominant state",
                        s=6, cmap="tab10")
    for i in range(k):
        scatter_chi(axes[i + 1], emb, chi[:, i], title=f"$\\chi_{{{i}}}$")
    fig.suptitle("chi IMAP layout (Invariant Manifold Approximation and Projection)", y=1.02)
    fig.tight_layout()
    return fig, emb


def plot_simplex_triangle(chi, ax=None):
    """k=3 membership triangle (barycentric)."""
    chi = np.asarray(chi)
    assert chi.shape[1] == 3, "triangle plot only for k=3"
    ax = ax or plt.gca()
    V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    xy = chi @ V
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=chi.argmax(1), cmap="viridis", alpha=0.5)
    tri = np.vstack([V, V[:1]])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
    for m, v in enumerate(V):
        ax.annotate(f"state {m}", v, ha="center", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("membership simplex $\\Delta^2$")
    return ax
