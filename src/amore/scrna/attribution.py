"""
amore.scrna.attribution — gene-level attribution of ISOKANN chi memberships, and
the driver-recovery scoring used to benchmark them.

Readouts of "what genes drive membership i":
  * gene_gradient   : per-cell d chi_i/d gene. For an HVG-input net this is a direct
                      input gradient; for a PCA-input net pass `loadings` to chain
                      d chi/d gene = (d chi/d PC) @ loadings. NOTE: the raw cell-mean
                      gradient is a POOR driver readout — the PCA version is dominated
                      by genes' PC-representation (loading norm, measured rho~0.98).
  * binned_gradient_sensitivity : the HEADLINE driver readout. Bin cells by chi_i,
                      average d chi_i/d gene over the cells in each bin, then average
                      the per-bin means. Density-correcting the gradient this way
                      removes the loading-norm bias (rho~0.98 -> ~0.62) and recovers
                      curated lineage drivers competitively with / better than a
                      fate-probability correlation — from the same network.
  * signed_corr     : Pearson/Spearman corr(gene, chi_i) — scale-invariant marker
                      score; competitive with GPCCA fate-prob correlation.

`recovery_at_k` scores any ranking against a curated/TF target set (CR2's overlap@k).
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import rankdata


# ── gradient attribution ────────────────────────────────────────────────────────

def gene_gradient(net, X, *, loadings=None, mode=None, batch=512, device="cpu"):
    """
    Per-gene attribution magnitude (max over cells of |d chi_i/d gene|) and signed
    aggregate (sign of mean gradient), for each membership i.

    Parameters
    ----------
    net : trained chi network (eval mode set internally).
    X   : (N, F) features the net was trained on.
    loadings : (F, G) PCA loadings to map PC-gradients to genes, OR None if the net
               already takes genes directly (F == G).
    mode : int membership index, or None for all k.

    Returns (gmax, gsigned) each (k, G) [or (G,) if mode given]: gmax = max_cell
    |d chi/d gene|, gsigned = sign(mean_cell d chi/d gene) * gmax.
    """
    net.eval()
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    N = Xt.shape[0]
    k = net(Xt[:2]).shape[1]
    L = torch.as_tensor(loadings, dtype=torch.float32, device=device) if loadings is not None else None
    G = (loadings.shape[1] if loadings is not None else Xt.shape[1])
    modes = [mode] if mode is not None else list(range(k))
    gmax = {i: np.zeros(G) for i in modes}
    gsum = {i: np.zeros(G) for i in modes}
    for s in range(0, N, batch):
        for i in modes:
            xb = Xt[s:s + batch].clone().requires_grad_(True)
            gi, = torch.autograd.grad(net(xb)[:, i].sum(), xb)
            dg = gi.detach()
            if L is not None:
                dg = dg @ L
            dg = dg.cpu().numpy()
            gmax[i] = np.maximum(gmax[i], np.abs(dg).max(0))
            gsum[i] += dg.sum(0)
    GMAX = np.stack([gmax[i] for i in modes])
    GSIG = np.stack([np.sign(gsum[i]) * gmax[i] for i in modes])
    if mode is not None:
        return GMAX[0], GSIG[0]
    return GMAX, GSIG


# NOTE: straight-line Integrated Gradients and the χ-MFEP reaction-path attribution
# were explored during the CR2 benchmark but superseded by `binned_gradient_sensitivity`
# below; they are preserved in examples/cr2_benchmark/chi_mfep_experimental.py.


# ── χ-sensitivity / correlation drivers ──────────────────────────────────────────

def binned_gradient_sensitivity(net, X, chi_col, mode, *, loadings=None, nbins=20,
                                batch=512, device="cpu"):
    """
    χ-sensitivity driver score (the headline AMORE driver measure).

    Bin cells by χ_mode, average ∂χ_mode/∂gene over the cells IN each bin (the
    ensemble gradient on each χ-isosurface), then average those per-bin means over
    bins. Equal weight per χ-level up-weights the sparse transition region where ∂χ
    is large — undoing the cell-density bias that sinks the plain cell-mean gradient
    (which is drowned by the many ~0-gradient committed cells). Simple, robust, and
    needs no path/anchor machinery.

    `loadings` (F, G): if the net takes PCs, chain to genes via (∂χ/∂PC) @ loadings;
    None if the net already takes genes. Returns a signed (G,) sensitivity (rank
    descending for positive drivers).
    """
    net.eval()
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    N = Xt.shape[0]
    L = torch.as_tensor(loadings, dtype=torch.float32, device=device) if loadings is not None else None
    G = (loadings.shape[1] if loadings is not None else Xt.shape[1])
    edges = np.linspace(0.0, 1.0, nbins + 1)
    binid = np.clip(np.digitize(np.asarray(chi_col), edges) - 1, 0, nbins - 1)
    bsum = np.zeros((nbins, G)); bcnt = np.zeros(nbins)
    for s in range(0, N, batch):
        xb = Xt[s:s + batch].clone().requires_grad_(True)
        g, = torch.autograd.grad(net(xb)[:, mode].sum(), xb)
        dg = g.detach()
        if L is not None:
            dg = dg @ L
        dg = dg.cpu().numpy()
        bsl = binid[s:s + batch]
        for b in np.unique(bsl):
            m = bsl == b
            bsum[b] += dg[m].sum(0); bcnt[b] += m.sum()
    return (bsum[bcnt > 0] / bcnt[bcnt > 0, None]).mean(0)


def signed_corr(X, target, *, spearman=False):
    """Signed (Pearson or Spearman) corr of each column of X with `target`. -> (F,)."""
    Xn = np.asarray(X, np.float64); t = np.asarray(target, np.float64)
    if spearman:
        Xn = np.column_stack([rankdata(Xn[:, j]) for j in range(Xn.shape[1])])
        t = rankdata(t)
    Xz = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-12)
    tz = (t - t.mean()) / (t.std() + 1e-12)
    return (Xz * tz[:, None]).mean(0)


# ── scoring ─────────────────────────────────────────────────────────────────────

def rank_genes(genes, score, descending=True):
    """Return gene names ordered by score (descending = positive drivers first)."""
    order = np.argsort(-np.asarray(score) if descending else np.asarray(score))
    return np.asarray(genes)[order]


def recovery_at_k(ranked_genes, target_set, ks=(10, 20, 50, 100)):
    """overlap@k: how many of target_set appear in the top-k of ranked_genes."""
    t = set(target_set)
    return {int(k): int(len(t.intersection(ranked_genes[:k]))) for k in ks}
