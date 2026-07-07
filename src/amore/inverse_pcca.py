# -*- coding: utf-8 -*-
"""
src/amore/inverse_pcca.py

Inverse PCCA+: recover the dominant Koopman eigenvalues, implied timescales,
and eigenfunctions from a learned membership function chi via the coarse
coupling Lambda_S, without ever forming the full transfer matrix.

chi is a partition of unity (chi >= 0, rows sum to 1 -- a softmax-head
ChiNetMulti output, trained e.g. with `amore.isotarget.isa_target`). Its
Galerkin/Rayleigh-Ritz projection onto the tau-lagged Koopman operator is

    G_hat = (1/N) chi^T W chi
    C_hat = (1/N) chi^T W (K_tau chi)
    Lambda_S = G_hat^{-1} C_hat

Because chi sums to 1 and K_tau 1 = 1, Lambda_S inherits row-sums equal to 1,
so the constant vector is always a right eigenvector at eigenvalue 1 (the
Perron mode). Diagonalising Lambda_S (an m x m matrix, m = number of
membership functions) recovers Ritz approximations to the dominant Koopman
eigenvalues/eigenfunctions -- exact when span(chi) is K_tau-invariant, with
error controlled by the invariance residual ``||K_tau chi - chi @ Lambda_S||``
(see `InversePCCAResult.residual`).

Non-reversible systems
-----------------------
Off equilibrium (e.g. a directed cycle), Lambda_S is generally not symmetric
and may carry a complex-conjugate eigenvalue pair. Diagonalising it naively
can hand the "Perron slot" to a rotation when the stationary mode and a
conjugate pair share modulus 1 (a degenerate cycle). Eigen-recovery there
goes through the pinned real-Schur ordering from the ISOKANN benchmark's
non-reversible isotarget module (`_sorted_real_schur`), which pins the
stationary mode first by closeness to +1 (not by modulus alone) and never
bisects a complex-conjugate block -- see that function's docstring. This
module imports it rather than re-deriving the same pinning logic; see the
NOTE below.

NOTE on the `schur_isotargets` import
--------------------------------------
`_coarse_propagator` (exactly G_hat^{-1} C_hat, weighted by an arbitrary
sampling measure) and `_sorted_real_schur` (the pinned real-Schur ordering)
live in `examples/isokann_benchmark/lib/schur_isotargets.py`, not in
`src/amore` proper -- that module is vendored benchmark code, reached by the
benchmark's own `harness.py` / `nonrev_targets.py` only via a `sys.path`
insert. This was confirmed by inspection before writing this file. Per an
explicit decision (reuse the existing pinning/ordering machinery rather than
re-deriving it), this module reaches it the same way.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import logm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_BENCHMARK_LIB = os.path.join(_REPO_ROOT, "examples", "isokann_benchmark", "lib")

if os.path.isdir(_BENCHMARK_LIB) and _BENCHMARK_LIB not in sys.path:
    sys.path.insert(0, _BENCHMARK_LIB)

try:
    from schur_isotargets import _coarse_propagator, _sorted_real_schur, _as_weights
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "amore.inverse_pcca requires "
        "examples/isokann_benchmark/lib/schur_isotargets.py "
        f"(looked under {_BENCHMARK_LIB}). This vendored module supplies the "
        "pinned real-Schur ordering and coarse-propagator machinery used "
        "here; it is intentionally not reimplemented in src/amore."
    ) from exc


__all__ = [
    "InversePCCAResult", "inverse_pcca",
    "SpectralProcess", "group_conjugate_pairs",
    "SpectralGap", "find_spectral_gap",
    "plot_complex_plane",
    "rate_matrix", "plot_rate_matrix",
]


@dataclass
class InversePCCAResult:
    """Result of `inverse_pcca`.

    lam : (m,) complex ndarray
        Eigenvalues of Lambda_S, ordered with the Perron (stationary) mode
        first, then descending by modulus.
    timescales : (m,) float ndarray
        Implied timescales t_i = -tau / log|lambda_i|. Index 0 (the Perron
        mode) is flagged as `np.inf` rather than dropped, so `lam` and
        `timescales` stay index-aligned.
    xi : (N, m) complex ndarray
        Recovered eigenfunctions evaluated at the N anchors, `xi = chi @ V`
        where V's columns are the right eigenvectors of Lambda_S. Real for
        `reversible=True`; may be genuinely complex for `reversible=False`
        (a conjugate pair) -- use `.real` / `.imag`.
    Lambda_S : (m, m) ndarray
        The coarse Koopman coupling in the chi basis (rows sum to 1).
    residual : float
        Invariance residual ``||K_tau chi - chi @ Lambda_S||_F``. The Ritz
        values/vectors above are exact only when this is small; eigenvalue
        error scales with it.
    """

    lam: np.ndarray
    timescales: np.ndarray
    xi: np.ndarray
    Lambda_S: np.ndarray
    residual: float


def inverse_pcca(
    chi: np.ndarray,
    propagate: Callable[[], np.ndarray],
    tau: float,
    *,
    reversible: bool = True,
    weights: Optional[np.ndarray] = None,
    transform: object = None,
) -> InversePCCAResult:
    """
    Recover dominant Koopman eigenvalues/timescales/eigenfunctions from a
    learned membership function chi, via the coarse coupling
    ``Lambda_S = G_hat^{-1} C_hat`` -- the Rayleigh-Ritz projection of the
    tau-lagged Koopman operator onto span(chi) -- without ever forming the
    full transfer matrix.

    Parameters
    ----------
    chi : (N, m) ndarray
        Membership function evaluated at N anchor points x0. Rows should sum
        to 1 (a softmax-head `ChiNetMulti` output); m is the number of
        membership functions / metastable states.
    propagate : callable, () -> (N, m) ndarray
        Returns K_tau chi at the SAME anchors and row ordering as `chi` (the
        existing Monte-Carlo burst-averaging estimator, bound via closure --
        see `examples/isokann_benchmark/inverse_pcca.ipynb` for the pattern).
        Not re-implemented here.
    tau : float
        Physical lag time used by `propagate`, for the timescale formula.
    reversible : bool, default True
        If True, Lambda_S is diagonalised directly and its spectrum is
        asserted to be (near-)real with the dominant eigenvalue near 1.
        If False, eigen-recovery uses the pinned real-Schur ordering so a
        complex-conjugate pair is handled correctly and a rotation is never
        assigned the Perron slot.
    weights : (N,) ndarray or None
        Sampling-measure weights W (pi-weighted for a reversible system;
        uniform/empirical otherwise). Defaults to uniform 1/N.
    transform : object or None
        If it exposes the coarse coupling as a `Lambda_S` attribute, that is
        used directly (skipping the G_hat/C_hat solve) and asserted to agree
        with the recomputed value within tolerance.

    Returns
    -------
    InversePCCAResult
    """
    chi = np.asarray(chi, dtype=np.float64)
    if chi.ndim != 2:
        raise ValueError(f"chi must be (N, m), got shape {chi.shape}")
    N, m = chi.shape

    Kchi = np.asarray(propagate(), dtype=np.float64)
    if Kchi.shape != chi.shape:
        raise ValueError(
            f"propagate() returned shape {Kchi.shape}, expected {chi.shape} "
            "(same anchors/ordering as chi)"
        )

    w = _as_weights(weights, N)
    Lambda_S = _coarse_propagator(chi, Kchi, w)

    transform_Lambda_S = getattr(transform, "Lambda_S", None) if transform is not None else None
    if transform_Lambda_S is not None:
        transform_Lambda_S = np.asarray(transform_Lambda_S, dtype=np.float64)
        if not np.allclose(transform_Lambda_S, Lambda_S, atol=1e-6, rtol=1e-4):
            raise AssertionError(
                "transform.Lambda_S disagrees with the recomputed G_hat^{-1} C_hat "
                f"coupling (max abs diff {np.max(np.abs(transform_Lambda_S - Lambda_S)):.3e})"
            )
        Lambda_S = transform_Lambda_S

    residual = float(np.linalg.norm(Kchi - chi @ Lambda_S))

    if reversible:
        lam, V = _eig_reversible(Lambda_S)
    else:
        lam, V = _eig_nonreversible(Lambda_S)

    xi = chi @ V
    timescales = _implied_timescales(lam, tau)

    return InversePCCAResult(lam=lam, timescales=timescales, xi=xi,
                              Lambda_S=Lambda_S, residual=residual)


def _implied_timescales(lam: np.ndarray, tau: float) -> np.ndarray:
    """t_i = -tau / log|lambda_i|; index 0 (Perron) is flagged as inf, not dropped."""
    timescales = np.full(len(lam), np.inf)
    abs_lam = np.clip(np.abs(lam[1:]), 1e-300, 1 - 1e-15)
    timescales[1:] = -tau / np.log(abs_lam)
    return timescales


def _eig_reversible(Lambda_S: np.ndarray, perron_tol: float = 1e-2):
    """Diagonalise Lambda_S directly, sorted by descending modulus. Asserts a
    (near-)real spectrum with the dominant eigenvalue near 1 (the Perron mode,
    right-eigenvector ~constant since chi is a partition of unity)."""
    lam, V = np.linalg.eig(Lambda_S)
    order = np.argsort(-np.abs(lam))
    lam, V = lam[order], V[:, order]

    max_imag = float(np.max(np.abs(lam.imag)))
    if max_imag > perron_tol:
        raise ValueError(
            "inverse_pcca(reversible=True): Lambda_S has a non-negligible "
            f"imaginary eigenvalue component (max |Im| = {max_imag:.3e}) -- this "
            "looks non-reversible; call with reversible=False."
        )
    lam = lam.real.astype(np.complex128)
    if abs(lam[0].real - 1.0) > perron_tol:
        raise ValueError(
            f"inverse_pcca(reversible=True): dominant eigenvalue {lam[0].real:.6f} "
            "is not close to 1 -- Lambda_S does not look like a valid coarse "
            "Koopman coupling (row sums should be 1)."
        )
    return lam, V.real.astype(np.complex128)


def _eig_nonreversible(Lambda_S: np.ndarray):
    """Eigen-recovery via the pinned real-Schur route: `_sorted_real_schur` fixes
    the ordering (stationary mode pinned to the Perron slot first by closeness
    to +1, never bisecting a complex-conjugate block). Eigenvectors are then
    read off `np.linalg.eig` (a full, numerically exact eigendecomposition) and
    matched to the Schur-ordered eigenvalues by nearest match -- Schur VECTORS
    themselves are only genuine eigenvectors for the leading invariant
    subspace, not eigenvector-by-eigenvector, so we use them for ordering only.
    """
    _, _, eigs, _ = _sorted_real_schur(Lambda_S)
    lam_raw, V_raw = np.linalg.eig(Lambda_S)

    used = np.zeros(len(lam_raw), dtype=bool)
    order = []
    for e in eigs:
        d = np.abs(lam_raw - e)
        d[used] = np.inf
        j = int(np.argmin(d))
        order.append(j)
        used[j] = True
    lam = lam_raw[order]
    V = V_raw[:, order]
    return lam, V


# ---------------------------------------------------------------------------
# Spectral-gap reading: how many distinct processes does a recovered spectrum
# actually contain, and where's the gap (e.g. to choose k for an
# overparametrized-chi training run)? See
# `examples/isokann_benchmark/inverse_pcca.ipynb`'s overparametrization
# section and `examples/2cm2/spectral_gap.ipynb` for worked examples.
# ---------------------------------------------------------------------------

@dataclass
class SpectralProcess:
    """One distinct physical process recovered from an `inverse_pcca` spectrum.

    A genuine complex-conjugate pair (same real part, opposite-sign imaginary
    part -- e.g. a rotation/oscillatory mode) occupies two raw entries of
    `InversePCCAResult.lam` but is ONE process; counting raw array positions
    instead over-counts by one per pair.

    kind : str
        "real" (a real eigenvalue), "complex_pair" (a genuine conjugate
        pair), or "unpaired_complex" (a complex eigenvalue with no conjugate
        partner found within `tol` -- `inverse_pcca`'s eigenvalues should
        always be real or come in conjugate pairs, so this is a red flag for
        numerical noise, not a genuine process; it is still returned rather
        than silently dropped or miscounted as real).
    modulus : float
        |lambda|, the sort key used across `group_conjugate_pairs` /
        `find_spectral_gap`.
    lam : complex
        A representative eigenvalue (for a pair, whichever half was
        encountered first in the descending-modulus sort -- both halves
        share the same modulus by construction).
    indices : tuple[int, ...]
        Index/indices of this process's eigenvalue(s) in the ORIGINAL
        (unsorted) `lam` array passed to `group_conjugate_pairs`.
    """

    modulus: float
    lam: complex
    kind: str
    indices: tuple[int, ...]


def group_conjugate_pairs(lam: np.ndarray, tol: float = 1e-6) -> list:
    """Group a (possibly complex) eigenvalue array into distinct physical
    processes, sorted by descending modulus: real singletons and genuine
    complex-conjugate pairs counted once each, not per raw array slot (see
    `SpectralProcess`). Robust to `lam`'s incoming order/sort convention --
    resorts by modulus internally.
    """
    lam = np.asarray(lam)
    order = np.argsort(-np.abs(lam))
    lam_sorted = lam[order]
    n = len(lam_sorted)
    seen = set()
    processes = []
    for i in range(n):
        if i in seen:
            continue
        li = lam_sorted[i]
        if abs(li.imag) < tol:
            processes.append(SpectralProcess(modulus=abs(li), lam=complex(li), kind="real",
                                              indices=(int(order[i]),)))
            seen.add(i)
            continue
        partner = next((j for j in range(i + 1, n)
                        if j not in seen and abs(lam_sorted[j] - li.conjugate()) < tol), None)
        if partner is not None:
            processes.append(SpectralProcess(
                modulus=abs(li), lam=complex(li), kind="complex_pair",
                indices=(int(order[i]), int(order[partner]))))
            seen.add(i); seen.add(partner)
        else:
            processes.append(SpectralProcess(modulus=abs(li), lam=complex(li),
                                              kind="unpaired_complex", indices=(int(order[i]),)))
            seen.add(i)
    return processes


@dataclass
class SpectralGap:
    """The largest drop in modulus between consecutive `SpectralProcess`es
    (as returned by `group_conjugate_pairs`, sorted descending): `k`
    processes sit above the gap, the rest below it -- the overparametrization
    read-off. Train chi with more membership functions than you think you
    need (k_trained > k_true), then use this instead of assuming k up front.
    """

    k: int
    modulus_above: float
    modulus_below: float
    gap: float


def find_spectral_gap(processes: list) -> SpectralGap:
    """Largest modulus drop between consecutive entries of `processes`
    (descending order, as returned by `group_conjugate_pairs`). `k` counts
    PROCESSES, not raw eigenvalue array slots -- a complex-conjugate pair
    already counts once. Ties are broken by the first (largest-k) occurrence.
    """
    if len(processes) < 2:
        raise ValueError("need at least 2 processes to find a gap")
    moduli = [p.modulus for p in processes]
    drops = [moduli[i] - moduli[i + 1] for i in range(len(moduli) - 1)]
    i = int(np.argmax(drops))
    return SpectralGap(k=i + 1, modulus_above=moduli[i], modulus_below=moduli[i + 1],
                       gap=drops[i])


def plot_complex_plane(ax, lam: np.ndarray, tol: float = 1e-6,
                       ref: Optional[np.ndarray] = None,
                       ref_label: str = "numerical reference") -> None:
    """Scatter `lam`'s eigenvalues in the complex plane (unit circle for
    reference), with genuine complex-conjugate pairs (`group_conjugate_pairs`)
    connected by a line so they read as one process, not two unrelated
    points. Optionally overlay a reference spectrum `ref` as open diamonds.
    Takes an existing `ax` (no matplotlib import here) -- caller owns the
    figure.
    """
    lam = np.asarray(lam)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=1, ls=":")
    ax.axhline(0, color="lightgray", lw=0.8)
    ax.axvline(0, color="lightgray", lw=0.8)

    colors = {"real": "C3", "complex_pair": "C0", "unpaired_complex": "C1"}
    seen_labels = set()
    for p in group_conjugate_pairs(lam, tol=tol):
        pts = lam[list(p.indices)]
        label = p.kind.replace("_", " ")
        if p.kind == "complex_pair":
            ax.plot(pts.real, pts.imag, color=colors[p.kind], lw=1, zorder=2)
        ax.scatter(pts.real, pts.imag, s=90, color=colors[p.kind], zorder=3,
                  label=label if label not in seen_labels else None)
        seen_labels.add(label)

    if ref is not None:
        ref = np.asarray(ref)
        ax.scatter(ref.real, ref.imag, s=70, facecolors="none", edgecolors="k",
                  marker="D", zorder=3, label=ref_label)

    ax.set_xlabel("Re(lambda)"); ax.set_ylabel("Im(lambda)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Coarse-grained rate matrix: a principled "which edge matters" readout.
# ---------------------------------------------------------------------------

def rate_matrix(Lambda_S: np.ndarray, tau: float) -> np.ndarray:
    """Coarse-grained continuous-time rate matrix Q = logm(Lambda_S) / tau, the
    generator of Lambda_S. Off-diagonal Q[i, j] (i != j) is the coarse
    transition RATE from state i to state j -- a principled "this edge
    matters, this edge doesn't" readout, in the same units as `1/timescales`,
    derived from `Lambda_S` alone rather than an ad-hoc heuristic (e.g.
    counting on-edge simplex frames) or a separate theory (TPT/Kramers).

    Works identically regardless of `reversible=True/False`: `Lambda_S` is
    always a real m x m matrix (chi and `propagate` are both real; only its
    EIGENvalues can be complex for a non-reversible system), so this needs no
    eigendecomposition and no Schur route -- it is a direct matrix function
    of `Lambda_S` itself, computed the same way for every system.

    Because `Lambda_S`'s rows sum to 1 (chi is a partition of unity), the
    all-ones vector is a right eigenvector at eigenvalue 1; `logm` preserves
    that eigenvector at eigenvalue log(1) = 0, so Q's rows sum to 0 exactly
    -- the defining property of a valid continuous-time generator/rate
    matrix, not something enforced separately here.

    Falls back to the first-order approximation `(Lambda_S - I) / tau` if
    `logm`'s principal branch is not real (can happen if `Lambda_S` has a
    negative real eigenvalue) -- accurate whenever `tau` is small relative to
    the coarse relaxation times, the same regime `inverse_pcca`'s own Ritz
    approximation is valid in.
    """
    Lambda_S = np.asarray(Lambda_S, dtype=np.float64)
    L = logm(Lambda_S)
    if np.max(np.abs(L.imag)) > 1e-6:
        L = Lambda_S - np.eye(len(Lambda_S))
    else:
        L = L.real
    return L / tau


def plot_rate_matrix(ax, Q: np.ndarray, labels: Optional[list] = None,
                     cmap: str = "coolwarm", title: Optional[str] = None) -> None:
    """Heatmap of a coarse rate matrix `Q` (see `rate_matrix`): `imshow` with the numeric
    rate annotated in each cell, on a diverging colormap centered at 0 -- positive
    off-diagonal rates and the negative diagonal escape rate read apart at a glance.
    Off-diagonal entries should be non-negative for a valid generator; a value that
    prints noticeably negative is itself a diagnostic (see `rate_matrix`'s docstring on
    when `Lambda_S` is far enough from `expm` of a true generator for this to happen).
    Takes an existing `ax` (no matplotlib import here) -- caller owns the figure.
    """
    Q = np.asarray(Q)
    m = len(Q)
    qmax = np.abs(Q).max()
    ax.imshow(Q, cmap=cmap, vmin=-qmax, vmax=qmax)
    for i in range(m):
        for j in range(m):
            ax.text(j, i, f"{Q[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(m)); ax.set_yticks(range(m))
    if labels is not None:
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    if title is not None:
        ax.set_title(title)
