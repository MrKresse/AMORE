"""
amore.scrna.mfep — minimum-"free-energy"-path (χ-MFEP) extraction for single-cell
ISOKANN χ. There is no potential energy here, so we offer three definitions of the
progenitor→terminal path for a membership χ_i, all anchored at the χ_i = 0.5
transition state:

  Option 1  medoid spline   : per χ-isosurface, take the medoid cell in feature
                              space; spline through their embedding coordinates.
                              Pure data-driven, always defined (no network gradient).
  Option 2  ∇χ path         : from the transition state, descend along −∇χ_i to
                              χ_i→0 and ascend along +∇χ_i to χ_i→1, retracting to
                              each level set (the MEP integrator with no energy).
(A third, ENSEMBLE-AVERAGED-gradient variant — the χ-MFEP — was explored but not
adopted; it lives in examples/cr2_benchmark/chi_mfep_experimental.py.)

Options 1/2 live in feature space; `project_to_chi_umap` / `project_to_embedding`
map them onto a fitted χ-UMAP.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ..mep.core import reaction_integrator


class ModeNet(nn.Module):
    """Adapter exposing a single χ_i output as a scalar net (for mep.core)."""
    def __init__(self, net, i):
        super().__init__()
        self.net = net.eval()
        self.i = i

    def forward(self, x):
        return self.net(x)[:, self.i:self.i + 1]


def _identity(x):
    return x


# ── transition state ────────────────────────────────────────────────────────────

def transition_state_medoid(chi_col, X, lo=0.45, hi=0.55):
    """χ_i≈0.5 transition state: medoid (closest-to-centroid) cell of the 0.5 band,
    in feature space. Returns (index, x_feature)."""
    chi_col = np.asarray(chi_col)
    mask = (chi_col >= lo) & (chi_col <= hi)
    if mask.sum() == 0:
        idx = int(np.argmin(np.abs(chi_col - 0.5)))
        return idx, X[idx].astype(np.float64)
    sub = X[mask]
    c = sub.mean(0)
    j = int(np.argmin(((sub - c) ** 2).sum(1)))
    idx = int(np.where(mask)[0][j])
    return idx, X[idx].astype(np.float64)


# ── Option 1: medoid spline ─────────────────────────────────────────────────────

def medoid_path(chi_col, X, emb, levels=None, band=0.05, min_cells=8, n_spline=100,
                smooth=None):
    """
    Per χ-isosurface medoid in FEATURE space; returns the medoids' EMBEDDING
    coordinates and a smooth spline through them (n_spline points). The per-band
    medoid can jump in the embedding, so `smooth` (splprep `s`) defaults to a
    fairly strong value; raise it for a smoother trend, lower it to track medoids.
    """
    chi_col = np.asarray(chi_col)
    if levels is None:
        levels = np.linspace(0.1, 0.9, 17)
    pts = []
    for l in levels:
        mask = np.abs(chi_col - l) < band
        if mask.sum() < min_cells:
            continue
        sub = X[mask]
        c = sub.mean(0)
        gi = int(np.where(mask)[0][np.argmin(((sub - c) ** 2).sum(1))])
        pts.append(emb[gi])
    pts = np.asarray(pts)
    if len(pts) < 3:
        return pts, pts
    try:
        from scipy.interpolate import splprep, splev
        k = min(3, len(pts) - 1)
        s = (len(pts) * 3.0) if smooth is None else smooth
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, k=k)
        u = np.linspace(0, 1, n_spline)
        spline = np.column_stack(splev(u, tck))
    except Exception:
        spline = pts
    return pts, spline


# ── Option 2: ∇χ descent/ascent (no energy) ─────────────────────────────────────

def gradient_path(net, X, x0, mode, steps=40, stepsize=0.025,
                  retract_steps=15, retract_tol=1e-6):
    """From x0 (transition state), integrate −∇χ_i to χ→0 and +∇χ_i to χ→1,
    retracting to level sets. Returns the feature-space path (P, F)."""
    mn = ModeNet(net, mode)
    down = reaction_integrator(mn, _identity, x0, steps=steps, stepsize=stepsize,
                               direction=-1, retract_steps=retract_steps, retract_tol=retract_tol)
    up = reaction_integrator(mn, _identity, x0, steps=steps, stepsize=stepsize,
                             direction=+1, retract_steps=retract_steps, retract_tol=retract_tol)
    return np.vstack([down[::-1], x0[None], up])


# NOTE: the ensemble-averaged-gradient path (χ-MFEP) explored during the CR2
# benchmark was moved out of the maintained package — it denoises by averaging ∇χ
# over each isosurface but is no more manifold-faithful than the local MEP above, so
# it is superfluous to the final results. It is preserved as a record in
# examples/cr2_benchmark/chi_mfep_experimental.py.


# ── projection to a fitted χ-UMAP ───────────────────────────────────────────────

def project_to_chi_umap(net, path_feature, reducer, device="cpu"):
    """Map a feature-space path onto a χ-UMAP: net(path) -> χ, then reducer.transform.
    NOTE: reducer.transform can scatter off-manifold interpolated points; prefer
    `project_to_embedding` for a clean on-manifold overlay."""
    net.eval()
    with torch.no_grad():
        chi_path = net(torch.tensor(np.asarray(path_feature, np.float32), device=device)).cpu().numpy()
    return reducer.transform(chi_path)


def project_to_embedding(path_feature, X_data, emb, k=15):
    """
    Project a feature-space path onto an existing 2-D embedding by snapping each
    path point to the DATA MANIFOLD: a distance-weighted average of the embedding
    coordinates of its k nearest real cells (in feature space). Unlike
    `umap.transform`, this guarantees the overlay stays on the visible cloud and
    never scatters to off-manifold positions.
    """
    from sklearn.neighbors import NearestNeighbors
    P = np.asarray(path_feature, np.float64)
    nn = NearestNeighbors(n_neighbors=k).fit(np.asarray(X_data, np.float64))
    d, idx = nn.kneighbors(P)
    w = np.exp(-(d / (d.mean(1, keepdims=True) + 1e-9)) ** 2)
    w /= w.sum(1, keepdims=True)
    return np.einsum("pk,pkd->pd", w, np.asarray(emb)[idx])


def smooth_2d(coords, n_out=100, smooth=None):
    """Spline-smooth a 2-D polyline (for clean overlay of projected paths)."""
    coords = np.asarray(coords)
    if len(coords) < 4:
        return coords
    try:
        from scipy.interpolate import splprep, splev
        s = len(coords) * 2.0 if smooth is None else smooth
        tck, _ = splprep([coords[:, 0], coords[:, 1]], s=s, k=3)
        return np.column_stack(splev(np.linspace(0, 1, n_out), tck))
    except Exception:
        return coords


def draw_path(ax, coords, *, color="k", lw=1.1, alpha=0.8, arrow=False, label=None,
              smooth=False, halo=True):
    """Overlay a path as a white-outlined line connecting its points, thin and slightly
    translucent so the χ-map underneath stays visible. `halo` draws the white outline for
    legibility on any background; `arrow` (off by default) adds an arrowhead at the χ=1
    end; `smooth=True` spline-smooths the polyline first."""
    import matplotlib.patheffects as pe
    coords = np.asarray(coords)
    if smooth:
        coords = smooth_2d(coords)
    eff = [pe.Stroke(linewidth=lw + 2.0, foreground="white"), pe.Normal()] if halo else None
    ax.plot(coords[:, 0], coords[:, 1], color=color, lw=lw, alpha=alpha, label=label,
            zorder=5, path_effects=eff)
    if arrow and len(coords) >= 2:
        ax.annotate("", xy=coords[-1], xytext=coords[-3 if len(coords) >= 3 else -2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha), zorder=6)
    return ax
