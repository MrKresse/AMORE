# -*- coding: utf-8 -*-
"""
nonrev_targets.py — adapt the two rough non-reversible isotarget ideas
(TransformSchurISA, TransformGPCCA) to the existing AMORE isotarget pipeline.

The two transforms live in `examples/non_reversible/schur_isotargets.py`. They were
written with an (n, k) row-major convention and return `(target, diag)`, whereas the
AMORE isotarget suite (`src/amore/isotarget.py`, `apply_target`) is (k, n) and returns
just `target`. They are also STATEFUL (each warm-starts the membership matrix A from the
previous fixed-point iterate), so one instance must persist across a whole training run.

`PipelineTarget` bridges both gaps: it transposes (k,n)<->(n,k), keeps one transform
instance per (variant, seed) run, and stashes the latest diagnostics in `.last_diag`
(eigenvalue imag parts, |λ_k|-|λ_{k+1}| gap, min membership before/after the feasibility
projection, cond(A)) so the harness can log the cyclic-feasibility story.

On a reversible system the dominant coarse-propagator spectrum is real, the inner-simplex
pivot is already feasible, and SchurISA reduces to ISA — exactly the intended control.
"""
from __future__ import annotations
import os, sys
import numpy as np

# Import the two rough ideas. They are vendored into this directory
# (schur_isotargets.py) so the benchmark is self-contained; the canonical copy also lives
# in examples/non_reversible/. Prefer the local vendored copy, fall back to the canonical.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from schur_isotargets import (                   # noqa: E402  (local vendored copy)
        TransformSchurISA, TransformGPCCA,
        _coarse_propagator, _sorted_real_schur, _inner_simplex_vertices, _feasible_k_flag,
    )
except ModuleNotFoundError:                          # fall back to canonical location
    sys.path.insert(0, os.path.join(_HERE, "..", "non_reversible"))
    from schur_isotargets import (                   # noqa: E402
        TransformSchurISA, TransformGPCCA,
        _coarse_propagator, _sorted_real_schur, _inner_simplex_vertices, _feasible_k_flag,
    )

NONREV_VARIANTS = ["schurisa", "gpcca"]


class PipelineTarget:
    """(k,n)->(k,n) stateful wrapper around a non-reversible Schur isotarget transform."""

    def __init__(self, kind: str, **kw):
        # simplex_normalize=False: return the feasibility-projected target in ISA-like
        # natural scale (the probability-simplex normalization collapses the linear ChiNet
        # in the fixed-point loop). The diagnostics still report the membership feasibility.
        kw.setdefault("simplex_normalize", False)
        if kind == "schurisa":
            self.t = TransformSchurISA(**kw)
        elif kind == "gpcca":
            # a bit more feasibility weight than the default keeps the crispness term
            # from dragging memberships negative on the cyclic toy
            kw.setdefault("feas_weight", 20.0)
            self.t = TransformGPCCA(**kw)
        else:
            raise ValueError(f"unknown non-reversible variant '{kind}'")
        self.kind = kind
        self.last_diag: dict | None = None

    def __call__(self, chi_kn: np.ndarray, kchi_kn: np.ndarray, k: int | None = None) -> np.ndarray:
        chi  = np.asarray(chi_kn, float).T            # (n, k)
        kchi = np.asarray(kchi_kn, float).T           # (n, k)
        # Reset the feasibility warm-start each call: the transform was designed to
        # warm-start A from the previous iterate to "track one optimum smoothly", but in
        # the AMORE fixed-point loop chi itself moves every iteration, so the persistent
        # warm-start drifts A toward a spread-killing optimum and collapses chi (SD->0.02
        # over ~150 iters). Warm-starting from the freshly-recomputed ISA pivot instead
        # keeps chi alive (k_eff=3, SD~0.28, vs ISA's 0.39) — see _probe_loop.py.
        self.t._A_prev = None
        target, diag = self.t(chi, kchi, k=k or chi.shape[1])
        self.last_diag = diag
        return target.T                               # (k, n)


def plain_isa_feasibility(chi_kn: np.ndarray, kchi_kn: np.ndarray) -> dict:
    """One-shot probe: what would PLAIN ISA (inner-simplex pivot, no feasibility
    projection) do on this (chi, kchi)? Returns the most-negative membership and the
    fraction of negative memberships — the quantity the feasibility projection fixes,
    and the diagnostics of the coarse propagator spectrum (imag parts reveal a cycle)."""
    chi  = np.asarray(chi_kn, float).T
    kchi = np.asarray(kchi_kn, float).T
    n, k = kchi.shape
    w = np.full(n, 1.0 / n)
    That = _coarse_propagator(chi, kchi, w)
    U, T, eigs, blocks = _sorted_real_schur(That)
    ok, msg = _feasible_k_flag(eigs, blocks, k)
    X = kchi @ U; X[:, 0] = 1.0
    idx, A0 = _inner_simplex_vertices(X)
    isa_target = X @ A0
    return dict(
        eigs=eigs,
        max_abs_imag=float(np.abs(np.imag(eigs)).max()),
        min_membership=float(isa_target.min()),
        frac_negative=float(np.mean(isa_target < -1e-6)),
        feasible_k=ok, k_message=msg,
    )
