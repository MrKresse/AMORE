"""
Non-reversible isotargets for ISOKANN: TransformSchurISA and TransformGPCCA.

These extend the reversible ISA transform to directional / cyclic transfer
operators (e.g. CellRank-style kernels), where a complex-conjugate pair can sit
in the dominant invariant subspace. They share ISA's machinery and reduce to it
exactly when the dominant top-k spectrum is real.

(Vendored into examples/nonrev_benchmark/ so the benchmark is self-contained; the
canonical copy also lives in examples/non_reversible/. The two `simplex_normalize`
kwargs are a backward-compatible pipeline adjustment — default True = original
behaviour.)

Conventions (matched to the rest of the AMORE isotarget suite, NOT copied
line-for-line from isotarget.jl since that file was not available here):

  * Operate on the propagated Koopman image  kchi = K^tau chi   (shape (n, k)).
    `chi` (n, k) is the current network output; the returned `target` is the
    regression target for the fixed-point step. The target is meant to be
    detached -- no gradient flows through this transform.
  * All inner products are weighted by D = diag(w), w_i = rho0(x_i), sum w = 1.
    This is the shooting/sampling measure, NOT the stationary pi. Off
    equilibrium these differ; using rho0 is the consistent choice for ISOKANN.
  * Partition of unity is the linear constraint  A @ 1 = e_0  given a constant
    leading basis column. Feasibility is  (basis @ A) >= 0  elementwise.

Escalation ladder (all share the same feasible-k constraint):
    ISA                  : inner-simplex pivot only        (reversible / real top-k)
    TransformSchurISA    : inner simplex + feasibility proj (cyclic, robust default)
    TransformGPCCA       : feasibility + crispness opt      (cyclic, crisper chi)

Author note: keep the diagnostics. On a directional kernel the eigenvalue
imaginary parts, the |lambda_k|-|lambda_{k+1}| gap, the most-negative membership
before projection, and cond(A) are the panels that tell you whether the chosen k
is feasible and whether plain ISA would have been silently wrong.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur, rsf2csf
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# weighting helpers
# --------------------------------------------------------------------------- #
def _as_weights(w, n):
    if w is None:
        w = np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float).reshape(-1)
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / s


def _coarse_propagator(chi, kchi, w):
    """rho0-Galerkin coarse propagator  That = (chi' D chi)^{-1} chi' D kchi.

    Real k x k. Its spectrum is the diagnostic object: real top-k -> ISA valid,
    a conjugate pair -> cyclic, needs feasibility projection / GPCCA.
    """
    D = w[:, None]
    G = chi.T @ (D * chi)            # (k, k) Gram in D
    M = chi.T @ (D * kchi)           # (k, k)
    return np.linalg.solve(G, M)


# --------------------------------------------------------------------------- #
# ordered real Schur (block-safe), constant mode first
# --------------------------------------------------------------------------- #
def _sorted_real_schur(M):
    """Real Schur of M, eigenvalues ordered by DESCENDING modulus, never
    splitting a 2x2 (complex-conjugate) block.

    Returns
    -------
    U : (k, k) real orthogonal Schur vectors (rotation within the subspace)
    T : (k, k) real quasi-triangular Schur form
    eigs : (k,) eigenvalues in the ordered diagonal-block sense
    blocks : list of (start, size) describing 1x1 / 2x2 diagonal blocks
    """
    # scipy's `sort` callable clusters the selected eigenvalues to the top-left
    # and keeps conjugate pairs intact; we get a full descending order by
    # selecting "modulus above a sweeping threshold" is fragile, so instead we
    # sort once with a key callable that scipy applies stably.
    k = M.shape[0]

    # First do an unsorted real Schur, then reorder by repeated selection.
    T, U = schur(M, output="real")

    # Identify diagonal blocks of the (unsorted) form.
    def block_structure(T):
        blocks = []
        i = 0
        n = T.shape[0]
        tol = 1e-12 * (1.0 + np.abs(T).max())
        while i < n:
            if i + 1 < n and abs(T[i + 1, i]) > tol:
                blocks.append((i, 2))
                i += 2
            else:
                blocks.append((i, 1))
                i += 1
        return blocks

    # Eigenvalue modulus per block (use complex form for 2x2 blocks).
    Tc, Uc = rsf2csf(T, U)
    diag = np.diag(Tc)

    blocks = block_structure(T)
    # modulus key per block; stationary mode (real eigenvalue closest to +1)
    # must be pinned FIRST -- on a cycle the conjugate pair shares modulus 1
    # with the stationary mode, so modulus alone does not separate them.
    mod_key = []
    stat_score = []
    for (s, sz) in blocks:
        if sz == 1:
            mod_key.append(abs(diag[s]))
            # closeness to the real eigenvalue +1, only for (near-)real eigs
            im = abs(np.imag(diag[s]))
            stat_score.append(abs(np.real(diag[s]) - 1.0) + 10.0 * im)
        else:
            mod_key.append(max(abs(diag[s]), abs(diag[s + 1])))
            stat_score.append(np.inf)  # a 2x2 block is never the stationary mode
    mod_key = np.asarray(mod_key)
    stat_score = np.asarray(stat_score)
    stat_block = int(np.argmin(stat_score))

    rest = [b for b in range(len(blocks)) if b != stat_block]
    rest_sorted = sorted(rest, key=lambda b: -mod_key[b])
    order = [stat_block] + rest_sorted

    # Rebuild U, T by permuting whole blocks. Because Schur form is
    # quasi-triangular, a clean block permutation generally requires Schur
    # reordering; for the small k here we reconstruct via an eigh/eig-free
    # route: project onto the ordered invariant subspaces using U columns.
    cols = []
    for bi in order:
        s, sz = blocks[bi]
        cols.extend(range(s, s + sz))
    P = np.eye(k)[:, cols]            # column permutation
    U_ord = U @ P
    T_ord = P.T @ T @ P

    # eigenvalues in ordered sense
    Tc2, _ = rsf2csf(T_ord, U_ord)
    eigs = np.diag(Tc2).copy()
    blocks_ord = block_structure(T_ord)
    return U_ord, T_ord, eigs, blocks_ord


def _feasible_k_flag(eigs, blocks, k):
    """Does the requested k bisect a 2x2 block? Returns (ok, message)."""
    # cumulative size up to the boundary
    csum = 0
    for (s, sz) in blocks:
        csum += sz
        if csum == k:
            return True, "k aligns with a block boundary"
        if csum > k:
            # the block that crossed the boundary
            if sz == 2 and (csum - sz) < k < csum:
                return False, (f"k={k} bisects a complex-conjugate pair "
                               f"(eigs {eigs[s]:.4g}, {eigs[s+1]:.4g}); "
                               f"snap to {csum-sz} or {csum}")
            return True, "k aligns with a block boundary"
    return True, "k uses full subspace"


# --------------------------------------------------------------------------- #
# inner simplex (ISA pivot) -- the closed-form warm start
# --------------------------------------------------------------------------- #
def _inner_simplex_vertices(X):
    """Greedy inner-simplex vertex search (Gram-Schmidt / max-volume pivot),
    the same pivot ISA uses. X is (n, k) with constant leading column.

    Returns vertex row indices (length k) and A0 = inv(X[idx]).
    """
    n, k = X.shape
    idx = np.zeros(k, dtype=int)

    # first vertex: row of largest norm
    idx[0] = int(np.argmax(np.einsum("ij,ij->i", X, X)))

    # affine shift to the first vertex, then Gram-Schmidt: at each step pick the
    # row with the largest residual orthogonal to the chosen vertex directions.
    ortho = X - X[idx[0]]
    for j in range(1, k):
        rownorm = np.einsum("ij,ij->i", ortho, ortho)
        idx[j] = int(np.argmax(rownorm))
        if j < k - 1:
            v = ortho[idx[j]].copy()
            nv = np.linalg.norm(v)
            if nv > 0:
                v = v / nv
                ortho = ortho - np.outer(ortho @ v, v)

    A0 = np.linalg.inv(X[idx])
    return idx, A0


# --------------------------------------------------------------------------- #
# partition-of-unity parameterization of A
# --------------------------------------------------------------------------- #
# Constraint A @ 1 = e_0  (first row sums to 1, other rows sum to 0) given a
# constant leading basis column. Free variables: A[:, :k-1]; last column fixed.
def _pack(A):
    return A[:, :-1].reshape(-1)


def _unpack(free, k):
    A = np.empty((k, k))
    A[:, :-1] = free.reshape(k, k - 1)
    rowtarget = np.zeros(k)
    rowtarget[0] = 1.0
    A[:, -1] = rowtarget - A[:, :-1].sum(axis=1)
    return A


# --------------------------------------------------------------------------- #
# escalation 1: feasibility-projected ISA
# --------------------------------------------------------------------------- #
class TransformSchurISA:
    """Robust default for non-reversible dynamics.

    inner-simplex pivot (closed form) -> soft feasibility projection onto
    {A : (X A) >= 0, A 1 = e_0}, warm-started from the ISA pivot and from the
    previous iterate's A. Reduces to ISA exactly when the pivot is already
    feasible (real top-k, metastable geometry).

    The feasibility step is the *mandatory* generalization for cyclic systems:
    on loop geometry the inner-simplex polygon excludes arc points, sending some
    memberships negative. Crispness optimization is deliberately omitted here.
    """

    def __init__(self, feas_weight=1.0, maxiter=200, tol=1e-10,
                 simplex_normalize=True):
        self.feas_weight = feas_weight
        self.maxiter = maxiter
        self.tol = tol
        # simplex_normalize=True (default): clip negatives and row-normalize to a
        # probability membership (PCCA+ output). =False: return the feasibility-projected
        # affine target X@A in natural scale, like the ISA isotarget -- this is the form
        # that trains a linear-output ChiNet stably in the AMORE fixed-point loop (the hard
        # simplex normalization squashes the target toward the centroid and collapses chi).
        self.simplex_normalize = simplex_normalize
        self._A_prev = None  # warm start across fixed-point iterations

    def _objective(self, free, X, k):
        A = _unpack(free, k)
        Y = X @ A
        viol = np.minimum(Y, 0.0)
        f = self.feas_weight * np.sum(viol * viol)
        # gradient
        dY = 2.0 * self.feas_weight * viol            # (n, k)
        dA = X.T @ dY                                 # (k, k)
        # project gradient onto free coords (last column is dependent)
        dfree = (dA[:, :-1] - dA[:, -1:]).reshape(-1)
        return f, dfree

    def __call__(self, chi, kchi, w=None, k=None):
        chi = np.asarray(chi, float)
        kchi = np.asarray(kchi, float)
        n, kk = kchi.shape
        k = kk if k is None else k
        w = _as_weights(w, n)

        That = _coarse_propagator(chi, kchi, w)
        U, T, eigs, blocks = _sorted_real_schur(That)
        ok, msg = _feasible_k_flag(eigs, blocks, k)

        # real ordered basis; pin constant leading column to enforce p.o.u.
        X = kchi @ U
        X[:, 0] = 1.0

        idx, A0 = _inner_simplex_vertices(X)
        target_isa = X @ A0
        min_before = float(target_isa.min())

        A_start = self._A_prev if self._A_prev is not None else A0
        res = minimize(self._objective, _pack(A_start), args=(X, k),
                       jac=True, method="L-BFGS-B",
                       options=dict(maxiter=self.maxiter, ftol=self.tol))
        A = _unpack(res.x, k)
        self._A_prev = A.copy()

        raw = X @ A                                   # feasibility-projected affine target
        if self.simplex_normalize:
            target = np.clip(raw, 0.0, None)
            rs = target.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            target = target / rs
        else:
            target = raw

        diag = dict(
            eigs=eigs, gap=_gap(eigs, k), feasible_k=ok, k_message=msg,
            min_membership_before_proj=min_before,
            min_membership_after_proj=float(raw.min()),
            condA=float(np.linalg.cond(A)),
            feas_residual=float(res.fun),
        )
        return target, diag


# --------------------------------------------------------------------------- #
# escalation 2: full GPCCA (feasibility + crispness), warm-started
# --------------------------------------------------------------------------- #
class TransformGPCCA:
    """Crispness variant. Adds the GPCCA-style crispness reward on top of the
    feasibility objective:

        minimize   feas_weight * ||relu(-X A)||^2  -  crisp_weight * crispness(A)

    crispness here is the Roeblitz-Weber I2 objective  sum_{i,j} A_ij^2 / A_0j
    (the optimized-A criterion in pyGPCCA), which sharpens memberships toward
    the simplex corners. Non-convex -> warm-start from the previous iterate to
    track one optimum smoothly and avoid target jitter in the fixed-point loop.

    Use when cyclic structure is strong enough that sharper chi (hence sharper
    ||grad chi|| driver score) is worth the extra loop fragility. Benchmark its
    driver recovery against TransformSchurISA rather than assuming it wins.
    """

    def __init__(self, feas_weight=10.0, crisp_weight=1.0,
                 maxiter=300, tol=1e-10, simplex_normalize=True):
        self.feas_weight = feas_weight
        self.crisp_weight = crisp_weight
        self.maxiter = maxiter
        self.tol = tol
        self.simplex_normalize = simplex_normalize   # see TransformSchurISA note
        self._A_prev = None

    def _objective(self, free, X, k, w):
        A = _unpack(free, k)
        Y = X @ A

        # feasibility penalty
        viol = np.minimum(Y, 0.0)
        f_feas = np.sum(viol * viol)
        dY = 2.0 * viol

        # crispness:  sum_ij A_ij^2 / A_0j   (maximize -> subtract)
        a0 = A[0, :].copy()
        eps = 1e-8
        a0s = np.where(np.abs(a0) < eps, np.sign(a0) * eps + eps, a0)
        crisp = np.sum((A * A) / a0s[None, :])
        # gradients of crisp wrt A
        dcrisp = 2.0 * A / a0s[None, :]
        dcrisp[0, :] += -np.sum(A * A, axis=0) / (a0s * a0s)

        f = self.feas_weight * f_feas - self.crisp_weight * crisp
        dA = self.feas_weight * (X.T @ dY) - self.crisp_weight * dcrisp
        dfree = (dA[:, :-1] - dA[:, -1:]).reshape(-1)
        return f, dfree

    def __call__(self, chi, kchi, w=None, k=None):
        chi = np.asarray(chi, float)
        kchi = np.asarray(kchi, float)
        n, kk = kchi.shape
        k = kk if k is None else k
        w = _as_weights(w, n)

        That = _coarse_propagator(chi, kchi, w)
        U, T, eigs, blocks = _sorted_real_schur(That)
        ok, msg = _feasible_k_flag(eigs, blocks, k)

        X = kchi @ U
        X[:, 0] = 1.0

        idx, A0 = _inner_simplex_vertices(X)
        min_before = float((X @ A0).min())

        A_start = self._A_prev if self._A_prev is not None else A0
        res = minimize(self._objective, _pack(A_start), args=(X, k, w),
                       jac=True, method="L-BFGS-B",
                       options=dict(maxiter=self.maxiter, ftol=self.tol))
        A = _unpack(res.x, k)
        self._A_prev = A.copy()

        raw = X @ A
        if self.simplex_normalize:
            target = np.clip(raw, 0.0, None)
            rs = target.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            target = target / rs
        else:
            target = raw

        diag = dict(
            eigs=eigs, gap=_gap(eigs, k), feasible_k=ok, k_message=msg,
            min_membership_before_proj=min_before,
            min_membership_after_proj=float(raw.min()),
            condA=float(np.linalg.cond(A)),
            crispness=float(_crispness(A)),
        )
        return target, diag


# --------------------------------------------------------------------------- #
# small diagnostics
# --------------------------------------------------------------------------- #
def _gap(eigs, k):
    mod = np.sort(np.abs(eigs))[::-1]
    if k < len(mod):
        return float(mod[k - 1] - mod[k])
    return float("inf")


def _crispness(A):
    a0 = A[0, :]
    eps = 1e-8
    a0s = np.where(np.abs(a0) < eps, eps, a0)
    return np.sum((A * A) / a0s[None, :])
