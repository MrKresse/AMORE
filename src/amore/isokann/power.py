"""
Multi-dimensional ISOKANN power iteration with orthogonal deflation.

Algorithm
---------
The Koopman operator K acts on functions: Kf(x) = E[f(x_tau) | x].
Its dominant k eigenfunctions chi_1,...,chi_k satisfy K chi_i = lambda_i chi_i.

We learn them simultaneously via orthogonal power iteration:

  For each outer iteration n:
    1. target(x) = Chi_n(x_tau)          [approximate K[Chi_n]]
    2. Orthogonalise columns of target   [deflate to distinct eigenfunctions]
    3. Scale target to a consistent range
    4. Train Chi_{n+1}(x) to fit target  [inner SGD loop]

The orthogonalisation step (whitening) prevents all k functions from
collapsing to the dominant eigenfunction.  It implements simultaneous
orthogonal iteration, which converges to the invariant subspace spanned
by the k eigenvectors with largest |lambda|.

References
----------
Rabben, Ray, Weber (2020) J. Chem. Phys. 153 — ISOKANN (k=1)
Mardt, Pasquali, Wu, Noe (2018) Nat. Comm. — VAMPnets
axsk/ISOKANN.jl — Julia reference implementation (multi-D)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


# ── Orthogonalisation ─────────────────────────────────────────────────────────

def whiten(Y: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Whiten the columns of Y so they have identity covariance.

    Computes the symmetric whitening matrix W = C^{-1/2} where
    C = (Y - mean(Y)).T @ (Y - mean(Y)) / n, then returns Y_w = Y_c @ W.

    Parameters
    ----------
    Y : (n, k)
    eps : float
        Eigenvalue floor to avoid numerical blow-up.

    Returns
    -------
    Y_w : (n, k)  — whitened, zero-mean columns
    W   : (k, k)  — whitening matrix (for evaluation on new points)
    """
    Yc = Y - Y.mean(0, keepdim=True)
    C  = (Yc.T @ Yc) / len(Y)                       # (k, k)
    D, V = torch.linalg.eigh(C)                      # eigenvalues in ascending order
    D    = D.clamp(min=eps)
    W    = V @ torch.diag(D.pow(-0.5)) @ V.T         # (k, k) — C^{-1/2}
    return Yc @ W, W


def scale_to_unit(Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-column min-max scaling to [0, 1]."""
    y_min = Y.min(0).values
    y_max = Y.max(0).values
    return (Y - y_min) / (y_max - y_min + eps)


# ── Koopman matrix and rates ───────────────────────────────────────────────────

def koopman_matrix(chi_x0: torch.Tensor,
                   chi_x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate the finite-dimensional Koopman matrix in the chi basis.

    K ≈ A^{-1} C  where:
      A_ij = E[chi_i(x0) chi_j(x0)]   (auto-correlation / stationary avg)
      C_ij = E[chi_i(x1) chi_j(x0)]   (cross-correlation)

    Parameters
    ----------
    chi_x0, chi_x1 : (n, k)

    Returns
    -------
    K : (k, k)  — Koopman matrix in the chi basis
    A : (k, k)
    C : (k, k)
    """
    n  = chi_x0.shape[0]
    A  = (chi_x0.T @ chi_x0) / n
    C  = (chi_x1.T @ chi_x0) / n
    K  = torch.linalg.solve(A + 1e-6 * torch.eye(A.shape[0], device=A.device), C)
    return K, A, C


def implied_timescales(chi_x0: torch.Tensor,
                       chi_x1: torch.Tensor,
                       lagtime: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute implied timescales from the eigenvalues of the Koopman matrix.

    tau_i = -lagtime / log(|lambda_i|)

    The first eigenvalue is always 1 (stationary state); skip it.

    Parameters
    ----------
    chi_x0, chi_x1 : (n, k)  evaluated on Koopman pairs
    lagtime : float            physical lag time

    Returns
    -------
    eigenvalues : (k,) complex numpy array, sorted by |lambda| descending
    timescales  : (k-1,) float array (all except stationary mode)
    """
    with torch.no_grad():
        K, _, _ = koopman_matrix(chi_x0, chi_x1)
    evals = torch.linalg.eigvals(K).cpu().numpy()
    evals = evals[np.argsort(-np.abs(evals))]         # sort by magnitude
    abs_evals = np.clip(np.abs(evals[1:]), 1e-12, 1 - 1e-12)  # skip stationary
    timescales = -lagtime / np.log(abs_evals)
    return evals, timescales


# ── Inner training loop ────────────────────────────────────────────────────────

def _train_one_iter(chi: nn.Module,
                    x0: torch.Tensor,
                    targets: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    epochs: int,
                    batch: int) -> float:
    """SGD inner loop: fit chi(x0) → targets."""
    chi.train()
    n = len(x0)
    total_loss = 0.0
    for _ in range(epochs):
        idx  = torch.randperm(n, device=x0.device)[:batch]
        pred = chi(x0[idx])
        loss = F.mse_loss(pred, targets[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / epochs


# ── Main power iteration ───────────────────────────────────────────────────────

def power_method_multi(
    chi:            nn.Module,
    x0:             torch.Tensor,
    x1:             torch.Tensor,
    n_iter:         int   = 40,
    epochs_per_iter: int  = 300,
    lr:             float = 1e-3,
    lr_decay:       float = 0.97,
    batch:          int   = 2048,
    collapse_eps:   float = 1e-3,
    verbose:        bool  = True,
) -> dict:
    """
    Multi-dimensional ISOKANN power iteration.

    Each outer iteration is one step of the orthogonal Koopman power method:
      1. Compute targets  Y = chi(x1)                     [Koopman action]
      2. Whiten Y columns (symmetric orthogonalisation)    [deflation]
      3. Scale to [0,1]                                    [normalisation]
      4. Train chi(x0) to fit Y_scaled                    [inner SGD]

    Parameters
    ----------
    chi : nn.Module with forward(x) -> (batch, k)
    x0, x1 : (n, in_dim) Koopman pairs on the appropriate device
    n_iter : int
    epochs_per_iter : int
    lr : float
    lr_decay : float   multiplicative LR decay per iteration
    batch : int        SGD minibatch size
    collapse_eps : float
        If the variance of any chi column drops below this, reinitialise
        that column's targets with small noise (collapse guard).
    verbose : bool

    Returns
    -------
    dict with keys:
      'losses'      : list[float] — avg loss per iteration
      'spans'       : (n_iter, k) array — chi range per function per iter
      'eigenvalues' : (k,) complex — final Koopman eigenvalues
      'timescales'  : (k-1,) float — implied timescales (in arbitrary units)
    """
    opt       = torch.optim.Adam(chi.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)
    k         = chi(x0[:1]).shape[-1]

    losses: list[float] = []
    spans_all: list[np.ndarray] = []

    if verbose:
        print(f"Multi-D ISOKANN  k={k}  ({n_iter} iters x {epochs_per_iter} epochs)")
        print(f"{'iter':>5}  {'loss':>10}  " +
              "  ".join(f"chi{i}_span" for i in range(k)))

    for it in range(n_iter):
        # ── 1. Compute Koopman-propagated targets ─────────────────────────
        chi.eval()
        with torch.no_grad():
            Y = chi(x1)   # (n, k) — approximate K[chi](x0)

        # ── 2. Orthogonalise columns via SVD ─────────────────────────────
        # Y = U @ S @ V.T  →  U is (n, k) orthonormal, robust to rank deficiency.
        # This is numerically stabler than whitening when some eigenfunctions
        # have small amplitude (avoids division by near-zero eigenvalues).
        try:
            Yc = Y - Y.mean(0)
            U, S, Vh = torch.linalg.svd(Yc, full_matrices=False)  # U: (n,k)
            Y_orth = U
        except torch.linalg.LinAlgError:
            Y_orth = Y - Y.mean(0)

        # Collapse guard: if any column has very small singular value, inject noise
        with torch.no_grad():
            col_std = Y_orth.std(0)
        for j in range(k):
            if col_std[j].item() < collapse_eps:
                Y_orth = Y_orth.clone()
                Y_orth[:, j] = Y_orth[:, j] + collapse_eps * torch.randn(
                    len(Y_orth), device=Y_orth.device)

        # ── 3. Scale each column to [0, 1] ───────────────────────────────
        targets = scale_to_unit(Y_orth)

        # ── 4. Inner SGD ─────────────────────────────────────────────────
        avg_loss = _train_one_iter(chi, x0, targets, opt, epochs_per_iter, batch)
        scheduler.step()
        losses.append(avg_loss)

        # ── Diagnostics ──────────────────────────────────────────────────
        chi.eval()
        with torch.no_grad():
            chi_all = chi(x0)
        spans = (chi_all.max(0).values - chi_all.min(0).values).cpu().numpy()
        spans_all.append(spans)

        if verbose and ((it + 1) % 5 == 0 or it == 0):
            span_str = "  ".join(f"{s:.4f}" for s in spans)
            print(f"{it+1:>5}  {avg_loss:>10.5f}  {span_str}")

    # ── Final Koopman eigenvalues ─────────────────────────────────────────
    chi.eval()
    with torch.no_grad():
        chi_x0_f = chi(x0)
        chi_x1_f = chi(x1)

    # Use a reasonable lagtime placeholder (actual value depends on caller)
    evals, timescales = implied_timescales(chi_x0_f, chi_x1_f, lagtime=1.0)

    if verbose:
        print(f"\nKoopman eigenvalues: {evals}")
        print(f"Implied timescales:  {timescales}")

    return {
        "losses":      losses,
        "spans":       np.array(spans_all),    # (n_iter, k)
        "eigenvalues": evals,
        "timescales":  timescales,
    }
