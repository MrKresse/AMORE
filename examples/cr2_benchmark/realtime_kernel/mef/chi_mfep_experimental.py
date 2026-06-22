"""
chi_mfep_experimental.py — the χ-MFEP / reaction-path attribution idea that was
explored during the CR2 benchmark but did NOT make it into the headline method.

Kept here (out of `amore.src`) so the exploration is reproducible without bloating
the maintained package. The notebook's final results use, instead:
  * the LOCAL χ-MEP (`amore.scrna.gradient_path`) for pathway visualisation, and
  * the χ-binned averaged gradient (`amore.scrna.binned_gradient_sensitivity`) for
    drivers.

Contents
--------
  ensemble_gradient_path     : the χ-MFEP — at each χ level, step along the
                               ENSEMBLE-AVERAGED ∇χ over that isosurface, then
                               Newton-retract; optional centroid "anchor".
  reaction_path_attribution  : ∫ ∂χ/∂gene along a supplied gene-space path
                               (completeness Σ ≈ Δχ); the path-integral analogue
                               of Integrated Gradients on the χ-MFEP.
  integrated_gradients       : straight-line baseline→cell Integrated Gradients
                               for χ (completeness Σ IG ≈ mean Δχ).

CORRECTION (why this is "experimental", not the headline)
---------------------------------------------------------
We originally framed the ensemble χ-MFEP as winning because the centroid `anchor`
keeps the path "on the data manifold". Direct measurement said the OPPOSITE: in the
1576-D HVG space the anchored MFEP sits at the per-isosurface band *centroids* — which
are themselves averages BETWEEN cells, ~13 units from the nearest real cell — whereas
the local χ-MEP hugs actual cells (~1 unit away). So the MFEP is LESS manifold-faithful,
not more. Its only genuine benefit for attribution is DENOISING: averaging ∂χ over each
isosurface smooths the gradient. Because the simpler local MEP already gives smooth,
cell-hugging paths and the χ-binned gradient already denoises the driver readout, this
whole module is superfluous to the final results and lives here as a record only.

Run-standalone example at the bottom (`python chi_mfep_experimental.py`) reproduces the
reaction-path-vs-IG-vs-GPCCA driver comparison on the cached PC and HVG models.
"""
from __future__ import annotations
import numpy as np
import torch

from amore.scrna.mfep import ModeNet, _identity, gradient_path
from amore.mep.core import levelset_retract, _chi_and_grad, _chi_val


# ── the χ-MFEP (ensemble-averaged-gradient path) ─────────────────────────────────

def ensemble_gradient_path(net, X, chi_col, x0, mode, n_levels=200, band=0.06,
                           min_cells=8, chi_clip=(0.05, 0.95), chi_pct=(0.1, 99.9), anchor=0.2,
                           denom_floor=1e-2, retract_steps=15, retract_tol=1e-6, device="cpu"):
    """
    From x0, walk toward high/low χ_i by, at each target level, stepping along the
    ENSEMBLE-AVERAGED ∇χ_i (mean over cells on the current isosurface) to first order,
    then Newton-retracting to the exact level. Three guards stop the χ→0/1 blow-up:

      * χ-window clip: integrate only over the TRANSITION window — `chi_clip` intersected
        with the data percentiles `chi_pct`. Deep in the basins ∇χ→0 (committor is flat
        there, a known MD effect), so the first-order step would explode; we do not
        integrate into that region.
      * denominator floor: clamp |∇χ·direction| ≥ `denom_floor` so a small gradient
        cannot produce a runaway step.
      * centroid anchor: blend a fraction `anchor` toward the per-isosurface band
        centroid, then re-retract. NOTE (corrected): the band centroid is an average
        BETWEEN cells, not a real cell — anchoring pulls the path OFF the data manifold
        in high-D (measured ~13 units from the nearest cell vs ~1 for the local MEP).
        Its effect is to DENOISE / regularise the trajectory, not to keep it manifold-
        faithful. anchor=0 follows the raw ensemble gradient. Default 0.2.

    More `n_levels` ⇒ smaller χ increments ⇒ lower first-order error. Returns (P, F).
    """
    mn = ModeNet(net, mode)
    chi_col = np.asarray(chi_col)
    Xn = np.asarray(X, np.float32)
    p_lo, p_hi = np.percentile(chi_col, chi_pct)
    chi_lo = max(chi_clip[0], float(p_lo))
    chi_hi = min(chi_clip[1], float(p_hi))

    def _band_idx(level):
        mask = np.abs(chi_col - level) < band
        return (np.where(mask)[0] if mask.sum() >= min_cells
                else np.argsort(np.abs(chi_col - level))[:max(min_cells, 20)])

    def ens_dir(level):
        idx = _band_idx(level)
        xb = torch.tensor(Xn[idx], device=device, requires_grad=True)
        mn(xb).sum().backward()
        g = xb.grad.detach().cpu().numpy().mean(0)
        return g / (np.linalg.norm(g) + 1e-12)

    def walk(targets):
        x = x0.copy().astype(np.float64)
        pts = []
        for lvl in targets:
            cur, gloc = _chi_and_grad(mn, _identity, x)
            gd = ens_dir(cur)
            denom = float(np.dot(gd, gloc))
            if abs(denom) < 1e-8:                       # ensemble dir ⟂ ∇χ -> use local
                gd, denom = gloc, float(np.dot(gloc, gloc))
            denom = np.sign(denom) * max(abs(denom), denom_floor)   # floor: no runaway step
            x = x + (lvl - cur) * gd / denom            # first-order step to reach lvl
            x = levelset_retract(mn, _identity, x, lvl, max_steps=retract_steps, tol=retract_tol)
            if anchor > 0:                              # denoise toward the band centroid
                centroid = Xn[_band_idx(lvl)].mean(0).astype(np.float64)
                x = (1 - anchor) * x + anchor * centroid
                x = levelset_retract(mn, _identity, x, lvl, max_steps=retract_steps, tol=retract_tol)
            pts.append(x.copy())
        return pts

    start = float(np.clip(_chi_val(mn, _identity, x0), chi_lo, chi_hi))
    up = walk(np.linspace(start, chi_hi, n_levels))
    down = walk(np.linspace(start, chi_lo, n_levels))
    return np.vstack([down[::-1], x0[None], up])


# ── reaction-path attribution along a supplied gene-space path ───────────────────

def reaction_path_attribution(chi_fn, gene_path, mode, *, device="cpu"):
    """
    Path-integral ("reaction-path") attribution: integrate ∂χ_mode/∂gene along a supplied
    GENE-space path (e.g. the ensemble χ-MFEP from progenitor → terminal),

        attribution_j = Σ_t ∂χ/∂gene_j(midpoint_t) · (γ_{t+1,j} − γ_{t,j}).

    Like Integrated Gradients this satisfies completeness (Σ_j attribution_j ≈
    χ(end) − χ(start)); the path is curved rather than IG's straight line. `chi_fn` maps a
    (b, G) gene tensor to χ. Returns (attribution (G,) signed, delta_chi float).
    """
    P = np.asarray(gene_path, np.float32)
    attr = np.zeros(P.shape[1])
    for t in range(len(P) - 1):
        mid = (0.5 * (P[t] + P[t + 1]))[None]
        x = torch.tensor(mid, device=device, requires_grad=True)
        g, = torch.autograd.grad(chi_fn(x)[:, mode].sum(), x)
        attr += g.detach().cpu().numpy()[0] * (P[t + 1] - P[t])
    with torch.no_grad():
        c0 = float(chi_fn(torch.tensor(P[[0]], device=device))[0, mode])
        c1 = float(chi_fn(torch.tensor(P[[-1]], device=device))[0, mode])
    return attr, c1 - c0


def integrated_gradients(net, X, baseline, target_rows, mode, *,
                         m_steps=32, cell_batch=24, device="cpu"):
    """
    Integrated Gradients for membership `mode`, integrating d chi_mode/d x along the
    straight path baseline -> x for each target cell, averaged over target cells:

        IG(gene) = mean_x  (x - b) ⊙ (1/M) Σ_α d chi_mode/d x |_{b + α (x-b)}

    Completeness:  Σ_gene IG ≈ mean_x [chi_mode(x) - chi_mode(b)].
    Returns (ig (F,) signed, completeness_delta float).
    """
    net.eval()
    Xn = np.asarray(X, np.float32)
    F_ = Xn.shape[1]
    b = torch.as_tensor(np.asarray(baseline, np.float32), device=device)
    alphas = torch.as_tensor((np.arange(m_steps) + 0.5) / m_steps, dtype=torch.float32, device=device)
    ig = np.zeros(F_); comp = 0.0
    rows = np.asarray(target_rows)
    for s in range(0, len(rows), cell_batch):
        idx = rows[s:s + cell_batch]
        x = torch.as_tensor(Xn[idx], device=device)
        diff = x - b
        c = x.shape[0]
        pts = (b[None, None] + alphas[None, :, None] * diff[:, None, :]).reshape(c * m_steps, F_)
        pts.requires_grad_(True)
        g, = torch.autograd.grad(net(pts)[:, mode].sum(), pts)
        g = g.reshape(c, m_steps, F_).mean(1)
        ig += (g.detach().cpu().numpy() * diff.detach().cpu().numpy()).sum(0)
        with torch.no_grad():
            comp += float((net(x)[:, mode] - net(b[None])[:, mode]).sum())
    return ig / len(rows), comp / len(rows)


# ── ensemble of local χ-MEPs seeded across the separatrix + reaction-path drivers ──

def _farthest_point_seeds(P, n, seed=0):
    """Farthest-point sampling of n row-indices of P — spreads seeds across the set."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    d = np.full(len(P), np.inf)
    for _ in range(1, min(n, len(P))):
        d = np.minimum(d, ((P - P[idx[-1]]) ** 2).sum(1))
        idx.append(int(d.argmax()))
    return np.array(idx)


def separatrix_mep_drivers(net, X, chi_col, mode, *, loadings=None, n_seeds=10, steps=150,
                           sep=(0.45, 0.55), seed=0, device="cpu"):
    """
    Seed an ENSEMBLE of local χ-MEPs across the separatrix (the χ_mode≈0.5 decision
    surface), and integrate ∂χ_mode/∂gene along each curve, then average — a
    reaction-path driver readout that samples the whole transition front rather than
    one medoid path.

    For each of `n_seeds` cells chosen by farthest-point sampling from the χ∈`sep` band,
    run `gradient_path` (the local χ-MEP, χ→0 ↔ χ→1), and accumulate the path integral

        attr_j = Σ_t ∂χ/∂gene_j(mid_t) · Δgene_{j,t}.

    For a PC-input net pass `loadings` (50×G): gradients and displacements are mapped to
    gene space via @ loadings (PCA rows orthonormal ⇒ Σ_j attr_j ≈ Δχ still holds). For
    an HVG net leave `loadings=None`. Returns dict(attr (G,) signed, paths [feature-space
    arrays], seeds, completeness = mean Δχ over the ensemble).
    """
    chi_col = np.asarray(chi_col); Xn = np.asarray(X, np.float32)
    band = np.where((chi_col >= sep[0]) & (chi_col <= sep[1]))[0]
    if len(band) < n_seeds:
        band = np.argsort(np.abs(chi_col - 0.5))[:max(n_seeds, 20)]
    seeds = band[_farthest_point_seeds(Xn[band], n_seeds, seed)]
    L = np.asarray(loadings, np.float32) if loadings is not None else None
    G = L.shape[1] if L is not None else Xn.shape[1]
    net.eval()
    attr = np.zeros(G); comp = 0.0; paths = []
    for i in seeds:
        P = gradient_path(net, Xn, Xn[i].astype(np.float64), mode,
                          steps=steps, stepsize=0.96 / 2 / steps)
        paths.append(P)
        Pf = np.asarray(P, np.float32)
        for t in range(len(Pf) - 1):
            mid = torch.tensor((0.5 * (Pf[t] + Pf[t + 1]))[None], device=device, requires_grad=True)
            g, = torch.autograd.grad(net(mid)[:, mode].sum(), mid)
            g = g.detach().cpu().numpy()[0]; dx = Pf[t + 1] - Pf[t]
            if L is not None:
                g = g @ L; dx = dx @ L
            attr += g * dx
        with torch.no_grad():
            c0 = float(net(torch.tensor(Pf[[0]], device=device))[0, mode])
            c1 = float(net(torch.tensor(Pf[[-1]], device=device))[0, mode])
        comp += c1 - c0
    return dict(attr=attr / len(seeds), paths=paths, seeds=seeds, completeness=comp / len(seeds))


def binned_gradient_along_paths(net, paths, mode, *, loadings=None, nbins=20, device="cpu"):
    """
    χ-binned sensitivity evaluated on the POINTS of an MEP ensemble — the direction-only
    counterpart to `separatrix_mep_drivers`'s reaction-path integral. Concatenate all path
    points, bin them by χ_mode, average ∂χ_mode/∂gene per bin, then average over bins (no
    displacement weighting). Dropping the displacement weight recovers most of the driver
    signal the path integral loses, isolating the magnitude bias as the path integral's
    failure mode. `loadings` (50×G) chains a PC net's gradient to genes. Returns signed (G,).
    """
    L = np.asarray(loadings, np.float32) if loadings is not None else None
    G = L.shape[1] if L is not None else np.asarray(paths[0]).shape[1]
    P = np.vstack([np.asarray(p, np.float32) for p in paths])
    xt = torch.tensor(P, device=device, requires_grad=True)
    chiv = net(xt)[:, mode]
    g, = torch.autograd.grad(chiv.sum(), xt)
    g = g.detach().cpu().numpy(); cv = chiv.detach().cpu().numpy()
    if L is not None:
        g = g @ L
    edges = np.linspace(0.0, 1.0, nbins + 1)
    b = np.clip(np.digitize(cv, edges) - 1, 0, nbins - 1)
    bsum = np.zeros((nbins, G)); bcnt = np.zeros(nbins)
    for k in np.unique(b):
        m = b == k; bsum[k] = g[m].sum(0); bcnt[k] = m.sum()
    return (bsum[bcnt > 0] / bcnt[bcnt > 0, None]).mean(0)


if __name__ == "__main__":
    print(__doc__)
