"""
Python re-implementation of ISOKANN isotarget transforms from
NEWISOKANN/ISOKANN.jl/src/isotarget.jl, plus VAMP-2 / VAMPnet.

Each transform receives:
    chi_x0 : (k, n)  chi network evaluated at anchor points x0
    kchi   : (k, n)  E[chi(x_tau) | x0] — Koopman expectation
                     already averaged over bursts before this call

and returns:
    target : (k, n)  regression target for the next power iteration step

Convention: k = output dimension of chi network, n = number of anchors.

Variants
--------
V1  ShiftScale   — 1D min-max normalisation (classical ISOKANN-1)
V2  ISA          — Inner-Simplex Algorithm; PCCA+-style rotation
V3  GramSchmidt  — QR orthonormalisation of the Koopman image
V4  PseudoInv    — Schur-based inversion of the Koopman action
V5  Cross        — Rayleigh-Ritz with residual weighting (newest, Oct 2025)
V6  SVD          — DMD-style: SVD(chi), eigen of reduced Koopman H (new in v2)
B   VAMP2        — VAMP-2 score loss (training objective, not a target)
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# V1  TransformShiftScale (1D only)
# ──────────────────────────────────────────────────────────────────────────────

def shiftscale(kchi: np.ndarray) -> np.ndarray:
    """
    Min-max scale the Koopman expectation to [0, 1].
    kchi shape: (1, n) or (n,).
    """
    mn, mx = kchi.min(), kchi.max()
    if mx <= mn:
        raise ValueError("chi is constant — power iteration collapsed")
    return (kchi - mn) / (mx - mn)


# ──────────────────────────────────────────────────────────────────────────────
# V2  TransformISA (Inner Simplex Algorithm)
# ──────────────────────────────────────────────────────────────────────────────

def _isa_vertices(K: np.ndarray) -> list[int]:
    """
    Iteratively select k vertices of the simplex spanned by rows of K (n, k)
    via max-distance selection (equivalent to PCCAPlus.indexmap).
    """
    centre = K.mean(0)
    vertices = [int(np.argmax(np.linalg.norm(K - centre, axis=1)))]
    for _ in range(K.shape[1] - 1):
        dists = np.min(
            np.stack([np.linalg.norm(K - K[v], axis=1) for v in vertices]), axis=0
        )
        vertices.append(int(np.argmax(dists)))
    return vertices


def isa_target(chi_x0: np.ndarray, kchi: np.ndarray,
               permute: bool = True) -> np.ndarray:
    """
    V2 — Inner Simplex Algorithm.

    1. Find the k extreme rows of kchi' (= the simplex vertices).
    2. Invert the k×k submatrix to get a rotation A.
    3. target = A @ kchi  (rotates the Koopman image into the simplex).

    chi_x0, kchi : (k, n)
    """
    K = kchi.T.copy().astype(np.float64)   # (n, k)
    vertices = _isa_vertices(K)
    C = K[vertices]                          # (k, k)
    try:
        A = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        raise ValueError("ISA: simplex submatrix is singular (collapsed chi)")
    target = A @ kchi                        # (k, n)
    if permute:
        target = _fixperm(target, chi_x0)
    return target


def _fixperm(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """Permute rows of `new` to minimise L1 distance to `old`."""
    from itertools import permutations as _perms
    k = new.shape[0]
    if k > 8:                               # skip for large k
        return new
    best_perm = min(_perms(range(k)), key=lambda p: np.sum(np.abs(new[list(p)] - old)))
    return new[list(best_perm)]


# ──────────────────────────────────────────────────────────────────────────────
# V3  TransformGramSchmidt2  (QR orthonormalisation)
# ──────────────────────────────────────────────────────────────────────────────

def gramschmidt_target(kchi: np.ndarray) -> np.ndarray:
    """
    V3 — QR-based orthonormalisation of the Koopman image.

    Compute thin QR of kchi' → target = Q' scaled by sign(diag R).
    kchi : (k, n)  →  target : (k, n)
    """
    Y = kchi.T.astype(np.float64)           # (n, k)
    Q, R = np.linalg.qr(Y)                  # Q: (n, k), R: (k, k)
    signs = np.sign(np.diag(R))
    target = (Q * signs).T                  # (k, n)
    return target


# ──────────────────────────────────────────────────────────────────────────────
# V4  TransformPseudoInv  (Schur-based inversion of K)
# ──────────────────────────────────────────────────────────────────────────────

def pseudoinv_target(chi_x0: np.ndarray, kchi: np.ndarray,
                     normalize: bool = True,
                     permute: bool = True) -> np.ndarray:
    """
    V4 — Moore-Penrose pseudoinverse + real Schur vectors.

    Construct K^{-1} ≈ chi_x0 @ pinv(kchi), take its real Schur vectors Z,
    then target = Z' @ K^{-1} @ kchi.

    chi_x0, kchi : (k, n)
    """
    chi  = chi_x0.astype(np.float64)
    Kchi = kchi.astype(np.float64)

    try:
        Kchi_inv = np.linalg.pinv(Kchi)         # (n, k)
    except np.linalg.LinAlgError:
        raise ValueError("PseudoInv: pinv failed (degenerate kchi)")

    Kinv = chi @ Kchi_inv                        # (k, k)  ≈ K^{-1} in chi basis

    # Real Schur decomposition for numerical stability
    T, Z = scipy.linalg.schur(Kinv, output='real')   # Z: Schur vectors (k, k)
    target = Z.T @ Kinv @ Kchi                         # (k, n)

    if normalize:
        norms = np.linalg.norm(target, axis=1, keepdims=True)
        target = target / np.clip(norms, 1e-8, None) * target.shape[1]

    if permute:
        target = _fixperm(target, chi_x0)

    return target


# ──────────────────────────────────────────────────────────────────────────────
# V5  TransformCross  (Rayleigh-Ritz with residual weighting)
# ──────────────────────────────────────────────────────────────────────────────

class CrossHistory:
    """
    Maintains a rolling history of (chi, kchi) column matrices for
    TransformCross.  Start with maxcols=0 to use only the current batch.
    """

    def __init__(self, maxcols: int = 0):
        self.maxcols = maxcols
        self._X: np.ndarray | None = None   # (n, cols) history
        self._Y: np.ndarray | None = None

    def reset(self):
        self._X = None
        self._Y = None

    def update(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append (x', y') columns to history, trim to maxcols."""
        x = x.T.astype(np.float64)   # (n, k)
        y = y.T.astype(np.float64)

        if self._X is None:
            self._X, self._Y = x, y
        else:
            self._X = np.hstack([self._X, x])
            self._Y = np.hstack([self._Y, y])

        if self.maxcols > 0 and self._X.shape[1] > self.maxcols:
            self._X = self._X[:, -self.maxcols:]
            self._Y = self._Y[:, -self.maxcols:]

        return self._X, self._Y   # (n, total_cols)


def cross_target(chi_x0: np.ndarray, kchi: np.ndarray,
                 history: CrossHistory | None = None,
                 alpha: float = 1e-8,
                 tau: float = 1e-3,
                 p: float = 2.0) -> np.ndarray:
    """
    V5 — Rayleigh-Ritz with residual weighting (TransformCross).

    Builds a (optionally accumulated) basis from kchi columns, solves the
    reduced eigenvalue problem, and weights each Ritz vector by how well
    it satisfies the eigenvalue equation.

    chi_x0, kchi : (k, n)
    """
    k = chi_x0.shape[0]

    if history is not None:
        X, Y = history.update(chi_x0, kchi)
    else:
        X = chi_x0.T.astype(np.float64)    # (n, k)
        Y = kchi.T.astype(np.float64)

    n, d = X.shape

    # QR basis of Y
    Q, R = np.linalg.qr(Y, mode='reduced')  # Q: (n, d), R: (d, d)

    # Reduced eigenvalue problem
    C = X.T @ X + alpha * np.eye(d)          # (d, d)
    M = X.T @ Q                              # (d, d)
    T_mat = R @ np.linalg.solve(C, M)        # (d, d)

    vals, vecs = np.linalg.eig(T_mat)
    idx  = np.argsort(-np.abs(vals.real))[:k]
    vals = vals[idx]
    vecs = vecs[:, idx]
    V    = Q @ vecs                          # (n, k) ambient Ritz vectors

    # Residual weighting
    Lam   = np.diag(vals)
    Rres  = X @ vecs - (Y @ vecs) @ Lam     # (n, k)
    res   = np.sqrt(np.sum(Rres**2, axis=0))
    Ynorm = np.sqrt(np.sum((Y @ vecs)**2, axis=0))
    Xnorm = np.sqrt(np.sum((X @ vecs)**2, axis=0))
    denom = np.abs(vals) * (Ynorm + 1e-12) + Xnorm + 1e-12
    relres = res / denom
    w = 1.0 / (1 + (relres.real / tau) ** p)
    w = np.clip(w.real, 1e-3, 1.0)

    # Scale and sign-align
    target = (V * np.sqrt(w)).T              # (k, n)
    target *= np.sqrt(n)
    s = np.sign(np.sum(chi_x0 * target, axis=1, keepdims=True))
    target *= s

    return target.real


# ──────────────────────────────────────────────────────────────────────────────
# V6  TransformSVD  (DMD-style eigenstructure)
# ──────────────────────────────────────────────────────────────────────────────

def svd_target(chi_x0: np.ndarray, kchi: np.ndarray) -> np.ndarray:
    """
    V6 — DMD-style Koopman eigenstructure (TransformSVD in isotarget.jl).

    1. Thin SVD of L = chi_x0':  L = U S V'
    2. Reduced Koopman matrix:   H = U' R V S^{-1}   where R = kchi'
    3. Eigendecompose H; sort by descending Re(lambda).
    4. Ambient Ritz vectors:     target = (U @ eigvecs)' * sqrt(n)

    chi_x0, kchi : (k, n)
    Returns target : (k, n)
    """
    L = chi_x0.T.astype(np.float64)   # (n, k)
    R = kchi.T.astype(np.float64)     # (n, k)
    n, d = L.shape

    U, S, Vh = np.linalg.svd(L, full_matrices=False)   # U:(n,d), S:(d,), Vh:(d,d)
    V  = Vh.T                                           # (d, d) — Julia convention
    H  = (U.T @ R) @ V @ np.diag(1.0 / np.maximum(S, 1e-10))  # (d, d)

    vals, vecs = np.linalg.eig(H)
    idx   = np.argsort(-vals.real)[:d]
    vecs  = vecs[:, idx]

    target = (U @ vecs.real).T         # (k, n)

    # Sign-align with chi_x0 for stability
    s = np.sign(np.sum(chi_x0 * target, axis=1, keepdims=True))
    s[s == 0] = 1.0
    return (target * s) * np.sqrt(n)


# ──────────────────────────────────────────────────────────────────────────────
# VAMP-2 loss (training objective — replaces power iteration)
# ──────────────────────────────────────────────────────────────────────────────

def _sym_sqrt_inv(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Symmetric matrix square-root inverse via eigendecomposition."""
    k = M.shape[0]
    eye = torch.eye(k, dtype=M.dtype, device=M.device)
    # Clamp NaN/Inf before adding regularisation.
    M_safe = torch.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0)
    M_reg  = M_safe + eps * eye
    try:
        L, V = torch.linalg.eigh(M_reg)
    except RuntimeError:
        # Last resort: return identity scaled by eps (gradient will be near-zero
        # but finite, letting the network recover from the degenerate state).
        return eye * (eps ** -0.5)
    L = L.clamp(min=eps)
    return V @ torch.diag(L.pow(-0.5)) @ V.T


def vamp2_score(chi_x0: torch.Tensor, chi_x1: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    """
    VAMP-2 score  =  ||C_{00}^{-1/2} C_{01} C_{11}^{-1/2}||_F^2.

    Maximising this is equivalent to finding the dominant Koopman eigenfunctions
    (Mardt, Pasquali, Wu, Noe 2018 — VAMPnets).

    chi_x0, chi_x1 : (n, k)  — note rows=samples, unlike the other targets
    Returns: scalar (positive; caller should negate to get a loss).
    """
    n   = chi_x0.shape[0]
    C00 = (chi_x0.T @ chi_x0) / n
    C11 = (chi_x1.T @ chi_x1) / n
    C01 = (chi_x0.T @ chi_x1) / n

    C00_inv_sqrt = _sym_sqrt_inv(C00, eps)
    C11_inv_sqrt = _sym_sqrt_inv(C11, eps)
    K_hat        = C00_inv_sqrt @ C01 @ C11_inv_sqrt
    return torch.linalg.matrix_norm(K_hat, 'fro') ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ──────────────────────────────────────────────────────────────────────────────

VARIANT_NAMES = {
    "shiftscale"  : "V1-ShiftScale",
    "isa"         : "V2-ISA",
    "gramschmidt" : "V3-GramSchmidt",
    "pseudoinv"   : "V4-PseudoInv",
    "cross"       : "V5-Cross",
    "svd"         : "V6-SVD",
    "vamp2"       : "B-VAMP2",
}

def apply_target(variant: str, chi_x0: np.ndarray, kchi: np.ndarray,
                 history: CrossHistory | None = None) -> np.ndarray:
    """
    Dispatch to the correct isotarget function.

    chi_x0, kchi : (k, n)  numpy arrays
    Returns target : (k, n)
    """
    if variant == "shiftscale":
        assert chi_x0.shape[0] == 1, "ShiftScale is 1D only"
        return shiftscale(kchi)
    elif variant == "isa":
        return isa_target(chi_x0, kchi)
    elif variant == "gramschmidt":
        return gramschmidt_target(kchi)
    elif variant == "pseudoinv":
        return pseudoinv_target(chi_x0, kchi)
    elif variant == "cross":
        return cross_target(chi_x0, kchi, history=history)
    elif variant == "svd":
        return svd_target(chi_x0, kchi)
    else:
        raise ValueError(f"Unknown variant '{variant}'. "
                         f"Choose from: {list(VARIANT_NAMES)}")
