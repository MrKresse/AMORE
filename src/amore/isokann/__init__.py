"""
amore.isokann — multi-dimensional ISOKANN for simultaneous Koopman eigenfunction learning.

The single-dimensional ISOKANN (Rabben, Ray, Weber 2020) learns one committor-like
function chi: X -> [0,1].  This module extends it to k simultaneous chi functions
that together form a membership simplex analogous to PCCA+:

    chi_i(x) >= 0,   sum_i chi_i(x) = 1.

Each vertex of this simplex corresponds to one metastable state.  The edges and
faces give the reaction pathways between them.  The simplex can be related to
UMAP / diffusion maps but is derived from actual Koopman dynamics rather than a
k-NN graph.

Quick start
-----------
>>> from amore.isokann import ChiNetMulti, power_method_multi, implied_timescales
>>> chi = ChiNetMulti(in_dim=2, k=3, hidden=[64, 32])
>>> result = power_method_multi(chi, x0, x1, n_iter=40, epochs_per_iter=200)
>>> evals, ts = implied_timescales(chi(x0), chi(x1), lagtime=0.01)
"""

from .network import ChiNetMulti, ChiNetMultiRaw
from .power   import power_method_multi, implied_timescales, koopman_matrix, whiten

__all__ = [
    "ChiNetMulti",
    "ChiNetMultiRaw",
    "power_method_multi",
    "implied_timescales",
    "koopman_matrix",
    "whiten",
]
