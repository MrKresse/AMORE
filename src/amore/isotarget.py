"""
src/amore/isotarget.py

Python port of the ISOKANN isotarget transforms, ported directly from the
Julia reference `axsk/ISOKANN.jl` → `src/isotarget.jl`.

Why this file exists
--------------------
The previous benchmark (v2) used `examples/benchmark/targets.py`, a hand-written
paraphrase that carried transcription bugs (a dropped order-1 rescaling in
GramSchmidt; dropped whitening in ISA vertex selection). This module is a fresh
port made directly from the Julia, with those bug classes explicitly guarded
against. See `BENCHMARK_V2_POSTMORTEM.md`.

Verification status
-------------------
This port was NOT checked numerically against a running Julia install (the
isolated-Julia route was judged not worth the environment overhead for a one-time
check). Instead:
  * Each transform is a direct, documented translation of the Julia.
  * `tests/test_isotarget.py` provides PROPERTY tests that catch the v2 bug class
    (amplitude, non-degeneracy, finiteness) without needing Julia.
  * The VAMP-2 score is cross-checked against `deeptime` (see the test file).
Property tests are weaker than numerical equivalence. If a Julia install is ever
available, generating fixtures and asserting equivalence remains the gold check.

Convention
----------
All transforms take and return arrays shaped (k, n):
    chi_x0 : (k, n)  chi network evaluated at anchor points x0
    kchi   : (k, n)  E[chi(x_tau) | x0], the Koopman expectation, already
                     burst-averaged before being passed in
    target : (k, n)  regression target for the next power-iteration step
k = chi output dimension, n = number of anchors.

The Julia works column-major and largely in "row space"; this port works with
explicit (k, n) arrays and transposes locally where the Julia would. Each
transform's docstring notes the corresponding Julia struct.

NOTE on Cross: `cross_target` is the most intricate transform (residual-weighted
Rayleigh-Ritz). It is ported faithfully but, having the most moving parts, is the
one most worth eyeballing against the Julia `rr_cross` by hand.

NOTE on SVD slot in benchmark: the v3 benchmark's "SVD" slot uses
`amore.isokann.power_method_multi` (subspace power iteration), NOT `svd_target`.
`svd_target` is the faithful port of Julia TransformSVD (DMD-style) and is
included so this module is complete, but the benchmark uses the subspace method.
See BENCHMARK_V2_POSTMORTEM.md, bug 3.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

try:
    import torch
except ImportError:  # torch only needed for the VAMP-2 score
    torch = None


# =============================================================================
# Shared helpers
# =============================================================================

def _fixperm(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """
    Permute the ROWS of `new` to minimise L1 distance to `old`.

    Port of Julia `fixperm`. The Julia brute-forces all permutations
    (Combinatorics.permutations) and is only used for small k; we do the same
    and skip permutation for k > 8 (matching the spirit of the Julia TODO that
    suggests Hungarian for larger systems).
    """
    from itertools import permutations as _perms

    k = new.shape[0]
    if k > 8:
        return new
    best = min(_perms(range(k)), key=lambda p: np.sum(np.abs(new[list(p)] - old)))
    return new[list(best)]


def _sign_align(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Flip the sign of each row of `target` so it agrees in orientation with the
    corresponding row of `ref`. Port of the `target .*= sign.(sum(...))` idiom
    used in several Julia transforms. A zero sign is treated as +1.
    """
    s = np.sign(np.sum(ref * target, axis=1, keepdims=True))
    s[s == 0] = 1.0
    return target * s


# =============================================================================
# V1  ShiftScale  (1D only) — Julia TransformShiftscale
# =============================================================================

def shiftscale(kchi: np.ndarray) -> np.ndarray:
    """
    Classical 1D ISOKANN min-max shift-scale of the Koopman expectation to [0, 1].

    Julia `shiftscale`:  chi = (ks - min) / (max - min)
    Only valid for a 1D chi (k == 1). Raises if chi is constant (the Julia
    throws a DomainError in the same case).
    """
    ks = np.asarray(kchi, dtype=np.float64)
    if ks.ndim == 2 and ks.shape[0] != 1:
        raise ValueError("ShiftScale only works with a 1D chi function (k == 1)")
    mn, mx = ks.min(), ks.max()
    if mx <= mn:
        raise ValueError("ShiftScale: chi is constant — power iteration collapsed")
    return (ks - mn) / (mx - mn)


# =============================================================================
# V2  ISA  (Inner Simplex Algorithm) — Julia TransformISA / myisa
# =============================================================================

def _indexmap(X: np.ndarray) -> list[int]:
    """
    Select the k simplex-vertex rows of X (shape (n, k)).

    This is a direct port of the vertex-selection logic that the Julia reaches
    through `PCCAPlus.indexmap`. It is inlined here deliberately (no external
    PCCA+ dependency) BECAUSE the v2 bug was a missing piece of exactly this
    step. Inlined, visible, and covered by a property test.

    Algorithm (standard inner-simplex vertex search):
      1. The first vertex is the row farthest from the CENTROID of all rows.
         (Distance from the centroid, not the origin — the construction must be
         translation-invariant; a corner sitting at the origin must still be
         findable. Picking "largest norm" instead silently fails for a simplex
         that straddles the origin.)
      2. Each subsequent vertex is the row maximising the distance to the affine
         subspace already spanned by the chosen vertices, found by Gram-Schmidt
         orthogonalisation of the row differences.
    This is the construction PCCA+ uses to locate simplex corners.
    """
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    if n < k:
        raise ValueError(f"ISA: need at least k={k} rows, got n={n}")

    # First vertex: row farthest from the centroid (translation-invariant).
    centre = X.mean(axis=0)
    idx = [int(np.argmax(np.linalg.norm(X - centre, axis=1)))]

    # ortho holds an orthonormal basis of the span of (X[v] - X[idx[0]]).
    # VECTORISED inner-simplex search: each step projects ALL n row-differences onto
    # the current basis at once (BLAS, multi-threaded) instead of a Python `for i in
    # range(n)` loop. This is the real training bottleneck on large n — ~80x faster on
    # 165k rows — and selects bit-for-bit the SAME vertices as the reference loop
    # (verified on random + real kchi; the maths is identical Gram-Schmidt).
    ortho: list[np.ndarray] = []
    for _ in range(1, k):
        base = X[idx[0]]
        D = X - base                                   # (n, k) all row differences at once
        for q in ortho:                                # remove already-spanned part
            D = D - np.outer(D @ q, q)
        dist = np.linalg.norm(D, axis=1)
        dist[idx] = -1.0                               # exclude already-chosen rows
        best_i = int(np.argmax(dist))
        if dist[best_i] < 0:
            raise ValueError("ISA: could not find a non-degenerate next vertex "
                             "(subspace collapsed)")
        idx.append(best_i)
        # the residual D[best_i] is exactly (X[best_i]-base) projected through ortho
        v = D[best_i]
        nv = np.linalg.norm(v)
        if nv > 1e-300:
            ortho.append(v / nv)
    return idx


def isa_target(chi_x0: np.ndarray, kchi: np.ndarray,
               permute: bool = True, whitening: bool = False) -> np.ndarray:
    """
    V2 — Inner Simplex Algorithm.  Julia `TransformISA` / `myisa`.

    RECOMMENDED USAGE (gold standard, see examples/isokann_benchmark): pair this target
    with a **softmax output head** (`amore.isokann.ChiNetMulti`) and **no warm-up**. The
    softmax architecturally enforces the membership simplex (chi>=0, sum=1), which removes
    the amplitude-collapse / mode-selection that a linear head suffers on high-dimensional
    inputs — softmax-ISA recovers all dominant slow modes from a random init (e.g. both the
    phi-flip and psi processes of vacuum alanine dipeptide), whereas linear-ISA needed a
    converged 1-D ShiftScale warm-up and still mode-selected. Use the linear head
    (`ChiNetMultiLinear`) only for the signed eigenfunction/basis targets (gramschmidt,
    pseudoinv, cross, svd).

    Julia:
        target = inv(K[vertices])' * K         (K = kchi', rows = anchors)
        optionally row-permuted to match chi for stability.

    *** BUG FIX (transpose) ***
    The ISA condition is A · (vertex value vectors) = I, where the vertex value
    vectors are the COLUMNS of K[verts]^T. Hence A = inv(K[verts]^T) = inv(K[verts])^T.
    The Julia `myisa(...)' * ks` includes that transpose; an earlier version of this
    port wrote `inv(K[verts]) @ kchi` (no transpose), which is correct only when the
    vertex submatrix is symmetric. Without it, the target is not one-hot at the
    vertices and ISA collapses to a constant (k_eff=0) once the GramSchmidt warm-up
    is removed. Verified against the Julia original; see benchmark_v3 ISA ablation.

    `whitening` mirrors the Julia `TransformISA.whitening` option: if set, the
    vertex search is run on a whitened copy of the rows (C^{-1/2} applied),
    while the inversion/target still use the unwhitened rows — exactly as the
    Julia `myisa` does.
    """
    K = np.asarray(kchi, dtype=np.float64).T.copy()      # (n, k), rows = anchors
    if K.shape[1] < 2:
        raise ValueError("ISA does not work with a 1D chi function")

    if whitening:
        C = (K.T @ K) / K.shape[0]
        # symmetric inverse square root C^{-1/2}
        evals, evecs = np.linalg.eigh(C)
        evals = np.clip(evals, 1e-12, None)
        W = evecs @ np.diag(evals ** -0.5) @ evecs.T
        verts = _indexmap(K @ W)
    else:
        verts = _indexmap(K)

    C_sub = K[verts]                                     # (k, k)
    try:
        A = np.linalg.inv(C_sub).T                        # inv(K[verts])^T  (see docstring)
    except np.linalg.LinAlgError:
        raise ValueError("ISA: simplex submatrix is singular (collapsed chi)")

    target = A @ np.asarray(kchi, dtype=np.float64)      # (k, n)
    if permute:
        target = _fixperm(target, np.asarray(chi_x0, dtype=np.float64))
    return target


# =============================================================================
# V3  GramSchmidt — Julia TransformGramSchmidt2
# =============================================================================

def gramschmidt_target(kchi: np.ndarray, renormalize: bool = True) -> np.ndarray:
    """
    V3 — QR orthonormalisation of the Koopman image.  Julia `TransformGramSchmidt2`.

    Julia:
        q, r = qr(chi')                  # chi here = kchi
        t    = q' .* diag(sign.(r))
        if renormalize:  c = sqrt(size(chi,2));  t .*= c

    *** v2 BUG GUARD ***
    The v2 port returned the raw orthonormal Q, whose columns have L2 norm 1,
    i.e. per-anchor amplitude ~1/sqrt(n). The Julia `renormalize` branch
    multiplies by c = sqrt(n) to bring the target back to order 1. Dropping that
    factor was the cause of the 0.49-0.51 amplitude collapse. It is restored
    here and `renormalize` defaults to True.

    `c = sqrt(size(chi, 2))` in the Julia: `chi` is (k, n) there, so `size(.,2)`
    is n, the number of anchors. Hence c = sqrt(n).
    """
    Y = np.asarray(kchi, dtype=np.float64).T             # (n, k)
    n = Y.shape[0]
    Q, R = np.linalg.qr(Y)                               # Q: (n, k), R: (k, k)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    target = (Q * signs).T                               # (k, n)
    if renormalize:
        target = target * np.sqrt(n)                     # <-- the restored factor
    return target


# =============================================================================
# V4  PseudoInv — Julia TransformPseudoInv  (direct=True, eigenvecs=True branch)
# =============================================================================

def pseudoinv_target(chi_x0: np.ndarray, kchi: np.ndarray,
                     normalize: bool = True, permute: bool = True) -> np.ndarray:
    """
    V4 — Moore-Penrose pseudoinverse + real Schur vectors.
    Julia `TransformPseudoInv` with the default `direct=True, eigenvecs=True`.

    Julia (direct branch):
        kchi_inv = pinv(kchi)
        Kinv     = chi * kchi_inv
        T        = schur(Kinv).vectors
        target   = T * Kinv * kchi
        if normalize: target = target ./ norm.(eachrow(target),1) .* size(target,2)
        if permute:   target = fixperm(target, chi)

    Note the Julia normalisation uses the **L1** row norm and multiplies by
    size(target, 2) == n. Reproduced exactly.
    """
    chi = np.asarray(chi_x0, dtype=np.float64)
    Kchi = np.asarray(kchi, dtype=np.float64)
    if chi.shape[0] < 2:
        raise ValueError("PseudoInv does not work with a 1D chi function")

    try:
        Kchi_inv = np.linalg.pinv(Kchi)                  # (n, k)
    except np.linalg.LinAlgError:
        raise ValueError("PseudoInv: pinv failed (degenerate kchi)")

    Kinv = chi @ Kchi_inv                                # (k, k)
    # real Schur decomposition; scipy returns (T, Z) with Z the Schur vectors
    _, Z = scipy.linalg.schur(Kinv, output="real")
    target = Z.T @ Kinv @ Kchi                           # (k, n)

    if normalize:
        l1 = np.sum(np.abs(target), axis=1, keepdims=True)   # L1 row norm
        l1 = np.clip(l1, 1e-12, None)
        target = target / l1 * target.shape[1]               # * n

    if permute:
        target = _fixperm(target, chi)
    return target


# =============================================================================
# V5  Cross — Julia TransformCross / rr_cross
# =============================================================================
# Most intricate transform. Ported faithfully from `rr_cross`; the Julia
# default keyword values are reproduced exactly. Worth a hand-check vs Julia.

def cross_target(chi_x0: np.ndarray, kchi: np.ndarray,
                 X_hist: np.ndarray | None = None,
                 Y_hist: np.ndarray | None = None,
                 alpha: float = 1e-8, tau: float = 1e-3, p: float = 2.0,
                 wmin: float = 1e-3,
                 clip_s: tuple[float, float] = (1e-2, 10.0)) -> np.ndarray:
    """
    V5 — Residual-weighted Rayleigh-Ritz.  Julia `rr_cross` (used by `TransformCross`).

    `X_hist`, `Y_hist` are the accumulated (n, cols) history matrices the Julia
    `TransformCross` keeps. If None, only the current batch is used (X = chi_x0',
    Y = kchi'). The benchmark harness owns the history bookkeeping; this function
    is the pure transform.

    Julia `rr_cross` (defaults alpha=1e-8, tau=1e-3, p=2.0, wmin=1e-3,
    clip_s=(1e-2,10.0)):
        Q, R   = qr(Y)
        C      = X'X + alpha*I
        M      = X' * Q
        T      = R * (C \\ M)
        vals, vecs = eigen(T)
        V      = Q * vecs
        residuals  = ||X*vecs - (Y*vecs)*diag(vals)||  (per column)
        relres     = residuals / (|vals|*(||Y vecs|| + eps) + ||X vecs|| + eps)
        w          = 1 / (1 + (relres/tau)^p)        clamped to [wmin, 1]
        s          = clamp(sqrt(w), clip_s)
        # NB: in the current Julia, Vscaled = V  (the s-scaling line is
        #     commented out); we reproduce the ACTIVE Julia behaviour.
        target = V                                    (then * sqrt(n), sign-align)

    The benchmark `targets.py` v2 applied `V * sqrt(w)`; the *active* Julia code
    does NOT (that line is commented out, `Vscaled = V`). This port follows the
    active Julia. `w`/`s` are still computed and returned for diagnostics.
    """
    if X_hist is not None and Y_hist is not None:
        X = np.asarray(X_hist, dtype=np.float64)
        Y = np.asarray(Y_hist, dtype=np.float64)
    else:
        X = np.asarray(chi_x0, dtype=np.float64).T       # (n, k)
        Y = np.asarray(kchi, dtype=np.float64).T         # (n, k)

    n, d = X.shape
    k = np.asarray(chi_x0).shape[0]

    Q, R = np.linalg.qr(Y, mode="reduced")               # Q:(n,d), R:(d,d)
    C = X.T @ X + alpha * np.eye(d)
    M = X.T @ Q
    T = R @ np.linalg.solve(C, M)                        # (d, d)

    vals, vecs = np.linalg.eig(T)
    order = np.argsort(-vals.real)
    vals = vals[order]
    vecs = vecs[:, order]
    V = Q @ vecs                                         # (n, d) ambient Ritz vectors

    # residuals (kept for diagnostics; do not scale the target by them — the
    # active Julia leaves Vscaled = V)
    Lam = np.diag(vals)
    Rres = X @ vecs - (Y @ vecs) @ Lam
    residuals = np.sqrt(np.sum(np.abs(Rres) ** 2, axis=0))
    Ynorm = np.sqrt(np.sum(np.abs(Y @ vecs) ** 2, axis=0))
    Xnorm = np.sqrt(np.sum(np.abs(X @ vecs) ** 2, axis=0))
    denom = np.abs(vals) * (Ynorm + 1e-300) + Xnorm + 1e-300
    relres = residuals / denom
    w = 1.0 / (1.0 + (relres.real / tau) ** p)
    w = np.clip(w, wmin, 1.0)
    _s = np.clip(np.sqrt(w), clip_s[0], clip_s[1])       # diagnostics only

    target = V[:, :k].real.T                             # (k, n)
    target = target * np.sqrt(n)                         # Julia: target .*= sqrt(size,1)
    target = _sign_align(target, np.asarray(chi_x0, dtype=np.float64))
    return target


# =============================================================================
# V6  SVD  (DMD-style) — Julia TransformSVD
# =============================================================================
# NOTE FOR THE BENCHMARK: the v3 benchmark's "SVD" slot uses
# `amore.isokann.power_method_multi` (subspace power iteration), NOT this
# transform. This `svd_target` is the faithful port of the Julia `TransformSVD`
# and is included so this module is complete. See BENCHMARK_V2_POSTMORTEM.md.

def svd_target(chi_x0: np.ndarray, kchi: np.ndarray) -> np.ndarray:
    """
    V6 — DMD-style Koopman eigenstructure.  Julia `TransformSVD`.

    Julia:
        L = model(xs)'      # = chi_x0'   (n, k)
        R = expectation()'  # = kchi'     (n, k)
        U, S, V = svd(L)
        H = U' * R * V * Diagonal(inv.(S))
        vl, vc = eigen(H, sortby = -)     # sort by descending value
        target = U * vc[:, 1:d]
        return target'

    Faithful port. SVD is of L = chi_x0' (the instantaneous chi), NOT of kchi.
    """
    L = np.asarray(chi_x0, dtype=np.float64).T           # (n, k)
    R = np.asarray(kchi, dtype=np.float64).T             # (n, k)
    n, d = L.shape

    U, S, Vh = np.linalg.svd(L, full_matrices=False)     # U:(n,d) S:(d,) Vh:(d,d)
    V = Vh.T
    H = (U.T @ R) @ V @ np.diag(1.0 / np.maximum(S, 1e-12))

    vals, vecs = np.linalg.eig(H)
    order = np.argsort(-vals.real)                       # Julia sortby = -
    vecs = vecs[:, order[:d]]

    target = (U @ vecs.real).T                           # (k, n)
    # The Julia TransformSVD returns target' directly with no extra scaling or
    # sign-alignment; reproduced as-is.
    return target


# =============================================================================
# VAMP-2 score  (training objective, not an isotarget transform)
# =============================================================================

def _sym_sqrt_inv(M, eps: float = 1e-6):
    """Symmetric inverse square root of a PSD matrix, with diagonal regularisation."""
    k = M.shape[0]
    eye = torch.eye(k, dtype=M.dtype, device=M.device)
    M_safe = torch.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0) + eps * eye
    evals, evecs = torch.linalg.eigh(M_safe)
    evals = evals.clamp(min=eps)
    return evecs @ torch.diag(evals.pow(-0.5)) @ evecs.T


def vamp2_score(chi_x0, chi_x1, eps: float = 1e-6):
    """
    VAMP-2 score = || C00^{-1/2} C01 C11^{-1/2} ||_F^2.

    Equivalently the sum of squared singular values of the half-weighted Koopman
    matrix — this is exactly `deeptime`'s 'VAMP2' score. Maximise it (negate for
    a loss). chi_x0, chi_x1 are (n, k): rows = samples (NOTE: different layout
    from the isotarget transforms above, which are (k, n)).

    VERIFICATION: cross-check against deeptime in tests/test_isotarget.py.
    deeptime's score has its own regularisation (`epsilon`, default 1e-6, and
    `score_mode`); for the cross-check to be meaningful, deeptime's `epsilon`
    must be set equal to `eps` here. With matched regularisation the two should
    agree to tight tolerance.
    """
    if torch is None:
        raise ImportError("vamp2_score requires PyTorch")
    n = chi_x0.shape[0]
    C00 = (chi_x0.T @ chi_x0) / n
    C11 = (chi_x1.T @ chi_x1) / n
    C01 = (chi_x0.T @ chi_x1) / n
    K_hat = _sym_sqrt_inv(C00, eps) @ C01 @ _sym_sqrt_inv(C11, eps)
    return torch.linalg.matrix_norm(K_hat, "fro") ** 2


# =============================================================================
# Dispatch
# =============================================================================

VARIANT_NAMES = {
    "shiftscale":  "V1-ShiftScale",
    "isa":         "V2-ISA",
    "gramschmidt": "V3-GramSchmidt",
    "pseudoinv":   "V4-PseudoInv",
    "cross":       "V5-Cross",
    "svd":         "V6-SVD-DMD",
    "vamp2":       "B-VAMP2",
}


def apply_target(variant: str, chi_x0: np.ndarray, kchi: np.ndarray,
                 cross_hist: tuple[np.ndarray, np.ndarray] | None = None
                 ) -> np.ndarray:
    """
    Dispatch to an isotarget transform. chi_x0, kchi are (k, n).

    `cross_hist` is the (X_hist, Y_hist) tuple for the Cross variant only;
    ignored by the others. VAMP2 is NOT dispatched here — it is a training loss,
    not a target; call `vamp2_score` directly.
    """
    if variant == "shiftscale":
        return shiftscale(kchi)
    if variant == "isa":
        return isa_target(chi_x0, kchi)
    if variant == "gramschmidt":
        return gramschmidt_target(kchi)
    if variant == "pseudoinv":
        return pseudoinv_target(chi_x0, kchi)
    if variant == "cross":
        if cross_hist is not None:
            return cross_target(chi_x0, kchi, X_hist=cross_hist[0],
                                Y_hist=cross_hist[1])
        return cross_target(chi_x0, kchi)
    if variant == "svd":
        return svd_target(chi_x0, kchi)
    raise ValueError(f"Unknown variant '{variant}'. "
                     f"Choose from {list(VARIANT_NAMES)} (vamp2 is a loss, "
                     f"not a target).")
