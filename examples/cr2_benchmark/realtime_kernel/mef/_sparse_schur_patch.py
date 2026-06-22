"""
Sparse partial real-Schur backend for pyGPCCA, implemented with SciPy's ARPACK
(`scipy.sparse.linalg.eigs`) so GPCCA runs on the 165 892-cell MEF transition matrix
WITHOUT SLEPc/PETSc (which have no Windows build — see the cr2 benchmark env notes).

Why this is needed
------------------
CellRank's `GPCCA.compute_schur` falls back to `method='brandts'` when slepc4py is
absent, and brandts densifies the transition matrix — 165892**2 * 8 B ≈ 205 GiB.
pyGPCCA's only sparse path is `method='krylov'`, which calls SLEPc's Krylov-Schur.

What we patch (numerically equivalent to the SLEPc path)
-------------------------------------------------------
1. `cellrank._utils._linear_solver._is_petsc_slepc_available` -> True, so CellRank
   keeps `method='krylov'` and does NOT densify the matrix. (The absorption solve in
   `compute_fate_probabilities` is still called with `use_petsc=False`, so no PETSc
   code path is ever exercised.)
2. `pygpcca._sorted_schur.sorted_schur` (and the name bound in `pygpcca._gpcca`) ->
   a SciPy implementation that, for a sparse input P, computes the `m` dominant
   eigenpairs with ARPACK, builds a REAL orthonormal basis Q of that invariant
   subspace (Re/Im parts for complex-conjugate pairs), forms R = Qᵀ P Q, and returns
   `(R, Q, eigenvalues)` — exactly the contract pyGPCCA's `_do_schur` consumes
   (constant Perron vector first, Q spans the dominant invariant subspace). The same
   `_check_conj_split` guard and `_check_schur` validation are applied. Dense inputs
   fall through to the original brandts path unchanged.

This changes neither the model nor the result: it is the identical dominant Schur
subspace SLEPc would return, obtained with a different (ARPACK) eigensolver.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

import pygpcca._sorted_schur as _ss
import pygpcca._gpcca as _gp
import cellrank._utils._linear_solver as _ls

# (1) tell CellRank petsc/slepc is available so it keeps the sparse krylov path.
_ls._is_petsc_slepc_available = lambda: True

_orig_sorted_schur = _ss.sorted_schur


def _real_invariant_basis(w: np.ndarray, V: np.ndarray, m: int):
    """Build a real basis of the invariant subspace of the >=m dominant eigenpairs.

    `w` (eigenvalues) and `V` (eigenvectors, columns) are assumed sorted by the
    selection criterion (descending). Complex-conjugate pairs are assumed adjacent
    (ARPACK returns them so); for each pair we take Re(v) and Im(v) — together they
    span the same 2-D real invariant subspace as {v, v̄}. Returns at least `m`
    columns, never splitting a conjugate pair.
    """
    cols, eig = [], []
    i, n = 0, len(w)
    while len(cols) < m and i < n:
        wi, vi = w[i], V[:, i]
        if abs(wi.imag) < 1e-10:
            cols.append(np.real(vi).copy())
            eig.append(complex(wi.real, 0.0))
            i += 1
        else:
            cols.append(np.real(vi).copy())
            cols.append(np.imag(vi).copy())
            eig.append(wi)
            eig.append(np.conj(wi))
            i += 2  # skip the adjacent conjugate partner
    return np.column_stack(cols), np.asarray(eig)


def sorted_schur(P, m, z="LM", method="krylov", tol_krylov=1e-16):
    # dense -> original (brandts) behaviour, untouched.
    if not sp.issparse(P):
        return _orig_sorted_schur(P, m, z=z, method="brandts", tol_krylov=tol_krylov)

    n = P.shape[0]
    if m > n:
        raise ValueError(f"Requested more groups than states: {m} > {n}.")
    k = int(min(n - 2, m + 4))   # a small buffer over m for convergence / pair safety
    w, V = eigs(P.astype(np.float64), k=k, which="LM" if z == "LM" else "LR", tol=0)
    order = np.argsort(-np.abs(w)) if z == "LM" else np.argsort(-w.real)
    w, V = w[order], V[:, order]

    B, eigvals = _real_invariant_basis(w, V, m)
    Q, _ = np.linalg.qr(B)                       # real orthonormal basis of the subspace
    Q = np.asarray(Q[:, : B.shape[1]], dtype=np.float64)
    R = Q.T @ (P @ Q)                            # (c, c) projected Schur form

    # replicate pyGPCCA's conjugate-split guard + slice to exactly m.
    if m < n:
        if _ss._check_conj_split(eigvals[:m]):
            raise ValueError(
                f"Clustering into {m} clusters will split complex conjugate eigenvalues. "
                "Request one cluster more or less."
            )
        Q, R, eigvals = Q[:, :m], R[:m, :m], eigvals[:m]

    _ss._check_schur(P=P, Q=Q, R=R, eigenvalues=eigvals, method="krylov")
    return R, Q, eigvals


# (2) install the sparse Schur backend everywhere pyGPCCA looks it up.
_ss.sorted_schur = sorted_schur
_gp.sorted_schur = sorted_schur


# =============================================================================
# (3) Sparse direct fate-probability solve (SuperLU) instead of densifying.
# -----------------------------------------------------------------------------
# CellRank's absorption solve `(I - Q) X = S` falls back to scipy's *dense* solver
# without PETSc (densifying the ~162k x 162k transient block -> 205 GiB), and its
# scipy gmres path fails to converge for the hardest MEF lineage. We route any
# SPARSE system through a single SuperLU factorization (exact, all RHS at once) —
# the same linear system CellRank defines, solved with a reliable sparse backend.
# =============================================================================
import scipy.sparse.linalg as _spla
import cellrank._utils._linear_solver as _lsmod
import cellrank.estimators.mixins._fate_probabilities as _fpmod

_orig_solve_lin_system = _lsmod._solve_lin_system


def _solve_lin_system(mat_a, mat_b, solver=_lsmod._DEFAULT_SOLVER, use_petsc=False,
                      preconditioner=None, n_jobs=None, backend=_lsmod.DEFAULT_BACKEND,
                      tol=1e-5, use_eye=False, show_progress_bar=True):
    if not sp.issparse(mat_a):
        return _orig_solve_lin_system(
            mat_a, mat_b, solver=solver, use_petsc=use_petsc, preconditioner=preconditioner,
            n_jobs=n_jobs, backend=backend, tol=tol, use_eye=use_eye,
            show_progress_bar=show_progress_bar)
    A = (sp.eye(mat_a.shape[0], format="csc") - mat_a) if use_eye else mat_a
    A = sp.csc_matrix(A)
    B = mat_b.toarray() if sp.issparse(mat_b) else np.asarray(mat_b)
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 1:
        B = B[:, None]
    try:
        lu = _spla.splu(A)                       # one factorization, all RHS columns
        X = lu.solve(B)
    except (MemoryError, RuntimeError):          # fall back to preconditioned lgmres
        M = _spla.LinearOperator(A.shape, _spla.spilu(A).solve)
        X = np.column_stack([_spla.lgmres(A, B[:, j], M=M, rtol=tol)[0]
                             for j in range(B.shape[1])])
    return X


_lsmod._solve_lin_system = _solve_lin_system
_fpmod._solve_lin_system = _solve_lin_system
