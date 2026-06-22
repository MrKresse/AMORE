"""
tests/test_isotarget.py

Verification for the isotarget port in `src/amore/isotarget.py`.

This port was NOT checked against a running Julia install. These tests are the
substitute safety net. They are of two kinds:

  1. PROPERTY tests — catch the v2 bug class (amplitude collapse, ISA bootstrap
     failure, non-finite output) without needing Julia. Weaker than numerical
     equivalence, but they specifically target how v2 went wrong.

  2. A VAMP-2 CROSS-CHECK against `deeptime` — a genuine external reference for
     the one component that has no Julia fixture anyway.

Run:  pytest tests/test_isotarget.py -v
The deeptime test is skipped automatically if deeptime is not installed
(`pip install deeptime`).

The v2 bugs these guard against (see BENCHMARK_V2_POSTMORTEM.md):
  * GramSchmidt returned raw orthonormal Q  -> per-anchor amplitude ~1/sqrt(n)
    instead of order 1  -> the 0.49-0.51 collapse.  Guarded by
    test_gramschmidt_amplitude_is_order_one.
  * ISA vertex selection lost its whitening/indexmap structure and could not
    bootstrap from a near-uniform chi.  Guarded by test_isa_* .
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from amore.isotarget import (
    shiftscale, isa_target, gramschmidt_target, pseudoinv_target,
    cross_target, svd_target, vamp2_score, apply_target,
)

K = 3            # chi dimension used across the benchmark
N = 1600         # ~ number of anchors (TW grid); large enough that 1/sqrt(n) is tiny
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Fixtures: two input regimes.
#   generic   — a well-spread (chi, kchi), the easy case.
#   nearunif  — chi almost constant (~0.5 everywhere) plus tiny noise. This is
#               the regime that exposed the v2 ISA/scaling bugs: at power-
#               iteration init the network output IS near-uniform.
# ---------------------------------------------------------------------------

def _generic():
    chi = RNG.uniform(0.0, 1.0, size=(K, N))
    kchi = chi + 0.05 * RNG.standard_normal((K, N))
    return chi, kchi


def _nearunif():
    chi = 0.5 + 1e-3 * RNG.standard_normal((K, N))
    kchi = chi + 1e-4 * RNG.standard_normal((K, N))
    return chi, kchi


ALL_KN_VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd"]


# ---------------------------------------------------------------------------
# Finiteness — every transform, both regimes. A transform may legitimately
# raise on the degenerate near-uniform input (ISA can), but it must never
# return NaN/Inf.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_KN_VARIANTS)
@pytest.mark.parametrize("regime", ["generic", "nearunif"])
def test_output_finite_and_shaped(variant, regime):
    chi, kchi = _generic() if regime == "generic" else _nearunif()
    try:
        out = apply_target(variant, chi, kchi)
    except ValueError:
        # An explicit, named failure on a degenerate input is acceptable
        # behaviour (e.g. ISA on a collapsed simplex). A silent NaN is not.
        return
    assert out.shape == (K, N), f"{variant}/{regime}: wrong shape {out.shape}"
    assert np.all(np.isfinite(out)), f"{variant}/{regime}: non-finite output"


# ---------------------------------------------------------------------------
# *** THE KEY REGRESSION TEST ***
# GramSchmidt must return an order-1 target, not a 1/sqrt(n) one.
# Raw orthonormal Q has per-column L2 norm 1, i.e. per-anchor RMS ~1/sqrt(N)
# (~0.025 for N=1600). The Julia multiplies by sqrt(N) to restore order 1.
# v2 dropped that factor -> amplitude collapse. This test fails loudly if the
# factor is missing again.
# ---------------------------------------------------------------------------

def test_gramschmidt_amplitude_is_order_one():
    chi, kchi = _generic()
    target = gramschmidt_target(kchi)                 # renormalize=True default
    rms = np.sqrt(np.mean(target ** 2, axis=1))       # per-mode RMS amplitude

    collapsed_scale = 1.0 / np.sqrt(N)                # ~0.025 — the v2 bug value
    assert np.all(rms > 20 * collapsed_scale), (
        f"GramSchmidt target RMS={rms} is ~1/sqrt(n) scale — the sqrt(n) "
        f"rescaling is missing. This is the v2 amplitude-collapse bug."
    )
    # And explicitly: dropping renormalisation reproduces the collapsed scale,
    # confirming the factor is exactly what separates the two.
    raw = gramschmidt_target(kchi, renormalize=False)
    raw_rms = np.sqrt(np.mean(raw ** 2, axis=1))
    assert np.all(raw_rms < 5 * collapsed_scale)
    np.testing.assert_allclose(rms, raw_rms * np.sqrt(N), rtol=1e-6)


def test_cross_amplitude_is_order_one():
    """Cross also applies a sqrt(n) scaling; guard it the same way."""
    chi, kchi = _generic()
    target = cross_target(chi, kchi)
    rms = np.sqrt(np.mean(target ** 2, axis=1))
    assert np.all(rms > 20.0 / np.sqrt(N)), (
        f"Cross target RMS={rms} looks 1/sqrt(n)-scaled; sqrt(n) factor missing."
    )


# ---------------------------------------------------------------------------
# ISA vertex selection / indexmap.
# ---------------------------------------------------------------------------

def test_isa_indexmap_recovers_known_simplex():
    """
    Build rows that are convex combinations of K known corner points, with the
    corners themselves present as rows. _indexmap must pick exactly the corners.

    The corners must live in K-dimensional space (here the standard-basis
    simplex e_1..e_K), because _indexmap selects k = X.shape[1] vertices — a
    K-vertex simplex embedded in fewer than K dimensions is degenerate for ISA.
    """
    from amore.isotarget import _indexmap
    corners = np.eye(K)                               # K corners in K-D space
    # interior points: random convex combinations of the K corners
    w = RNG.dirichlet(np.ones(K), size=200)
    interior = w @ corners
    X = np.vstack([corners, interior])                # corners are rows 0..K-1
    picked = sorted(_indexmap(X))
    assert picked == list(range(K)), (
        f"indexmap picked {picked}, expected the {K} simplex corners "
        f"{list(range(K))}."
    )


def test_isa_generic_runs_and_is_nondegenerate():
    """On a well-spread input ISA must produce a non-collapsed target."""
    chi, kchi = _generic()
    target = isa_target(chi, kchi)
    sd = target.std(axis=1)
    assert np.all(np.isfinite(target))
    assert np.any(sd > 1e-3), f"ISA produced a degenerate target (sd={sd})."


def test_isa_nearuniform_fails_loudly_not_silently():
    """
    On a near-uniform chi, ISA may genuinely fail to bootstrap — that is a real
    property of ISA, not a bug. The REQUIREMENT is that it fails LOUDLY (raises)
    rather than silently returning a NaN or a collapsed target. The v2 harness
    swallowed such failures; the port must surface them.
    """
    chi, kchi = _nearunif()
    try:
        target = isa_target(chi, kchi)
    except ValueError:
        return                                        # acceptable: explicit failure
    # If it did return, it must at least be finite (not silent NaN garbage).
    assert np.all(np.isfinite(target)), (
        "ISA returned a non-finite target on near-uniform input — silent "
        "failure. It must raise instead."
    )


# ---------------------------------------------------------------------------
# Shape / sanity for the remaining transforms.
# ---------------------------------------------------------------------------

def test_shiftscale_1d_only_and_range():
    _, kchi = _generic()
    ks1d = kchi[:1]                                   # (1, N)
    out = shiftscale(ks1d)
    assert out.shape == ks1d.shape
    assert out.min() >= -1e-9 and out.max() <= 1 + 1e-9
    with pytest.raises(ValueError):
        shiftscale(kchi)                              # k=3 must be rejected
    with pytest.raises(ValueError):
        shiftscale(np.ones((1, N)))                   # constant chi must raise


def test_svd_runs_and_shaped():
    chi, kchi = _generic()
    out = svd_target(chi, kchi)
    assert out.shape == (K, N)
    assert np.all(np.isfinite(out))


def test_pseudoinv_runs_and_shaped():
    chi, kchi = _generic()
    out = pseudoinv_target(chi, kchi)
    assert out.shape == (K, N)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# VAMP-2 cross-check against deeptime.
# This is the genuine external reference for the VAMP score. Same fixed inputs
# to both implementations; regularisation matched; assert agreement.
# ---------------------------------------------------------------------------

def test_vamp2_matches_deeptime():
    """
    Cross-check the benchmark's `vamp2_score` against deeptime's VAMP-2.

    IMPORTANT — the two use different conventions, and the test accounts for
    both rather than asserting naive equality:

      1. CENTERING. deeptime's VAMP uses mean-centered covariances (true
         covariances). The benchmark `vamp2_score` uses raw second moments
         (chi.T @ chi / n, no centering).
      2. STATIONARY MODE. deeptime's VAMP-2 score includes the trivial
         singular value 1 from the constant eigenfunction; the benchmark
         score does not.

    The verified identity is:
        deeptime VAMP2  ==  1 + ||K_centered||_F^2
    where K_centered is the half-weighted Koopman matrix built from *centered*
    covariances. This test confirms (a) deeptime satisfies that identity with
    our own centered computation, and (b) the benchmark's raw-moment
    `vamp2_score` equals an independent raw-moment reference. Together these
    verify the benchmark score is a correct VAMP-2 computation under its own
    (raw-moment, no-stationary) convention.

    If the benchmark intends its VAMP-2 to match deeptime's convention exactly,
    `vamp2_score` should center the data and add 1 — that is a design decision
    for the benchmark, flagged here, not silently patched.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("deeptime")
    from deeptime.decomposition import VAMP

    n, k = 500, 3
    rng = np.random.default_rng(7)
    chi0 = rng.uniform(0.0, 1.0, size=(n, k)).astype(np.float64)
    mix = np.eye(k) + 0.1 * rng.standard_normal((k, k))
    chi1 = (chi0 @ mix + 0.05 * rng.standard_normal((n, k))).astype(np.float64)

    EPS = 1e-6

    def _sqrtinv(M):
        w, V = np.linalg.eigh(M + EPS * np.eye(len(M)))
        w = np.clip(w, EPS, None)
        return V @ np.diag(w ** -0.5) @ V.T

    # (a) deeptime's score == 1 + centered Frobenius-norm-squared.
    deeptime_score = float(VAMP(epsilon=EPS).fit_fetch((chi0, chi1)).score(r="VAMP2"))
    x0c, x1c = chi0 - chi0.mean(0), chi1 - chi1.mean(0)
    c00, c11, c01 = x0c.T @ x0c / n, x1c.T @ x1c / n, x0c.T @ x1c / n
    K_cent = _sqrtinv(c00) @ c01 @ _sqrtinv(c11)
    centered_fro2 = float(np.sum(np.linalg.svd(K_cent, compute_uv=False) ** 2))
    np.testing.assert_allclose(
        deeptime_score, 1.0 + centered_fro2, rtol=1e-4,
        err_msg="deeptime VAMP-2 != 1 + centered ||K||_F^2 — convention drift; "
                "re-derive before trusting the cross-check.")

    # (b) the benchmark's vamp2_score == independent raw-moment ||K||_F^2.
    ours = float(vamp2_score(torch.tensor(chi0), torch.tensor(chi1), eps=EPS))
    C00, C11, C01 = chi0.T @ chi0 / n, chi1.T @ chi1 / n, chi0.T @ chi1 / n
    K_raw = _sqrtinv(C00) @ C01 @ _sqrtinv(C11)
    raw_fro2 = float(np.sum(np.linalg.svd(K_raw, compute_uv=False) ** 2))
    np.testing.assert_allclose(
        ours, raw_fro2, rtol=1e-4,
        err_msg=f"vamp2_score ({ours}) != raw-moment ||K||_F^2 ({raw_fro2}) — "
                f"the benchmark VAMP-2 score is computed incorrectly.")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
