"""
Euler-Maruyama Langevin integrator.

SDE:  dx = -∇U(x) dt + σ dW_t

Discretised (Euler-Maruyama):
    x_{n+1} = x_n + F(x_n) dt + σ √dt  Z_n,   Z_n ~ N(0, I)
"""

import numpy as np


class LangevinSimulator:
    """
    General-purpose overdamped Langevin integrator.

    Parameters
    ----------
    potential_fn : callable
        U(x) -> float, where x is a 1-D numpy array of length `dim`.
    grad_fn : callable or None
        ∇U(x) -> 1-D array of length `dim`.
        If None, the gradient is estimated via central finite differences.
    dim : int
        Phase-space dimension.
    sigma : float
        Noise amplitude (diffusion coefficient).
    dt : float
        Integration time step.
    lagtime : float
        Default lag time used to separate lagged pairs.
    support : array-like of shape (2, dim) or None
        [lo, hi] bounds used when sampling a random initial condition.
    """

    def __init__(self, potential_fn, grad_fn, dim, sigma, dt, lagtime, support=None):
        self.potential_fn = potential_fn
        self._grad_fn     = grad_fn
        self.dim          = dim
        self.sigma        = sigma
        self.dt           = dt
        self.lagtime      = lagtime
        self.support      = np.asarray(support) if support is not None else None

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def potential(self, x):
        return self.potential_fn(x)

    def gradient(self, x):
        if self._grad_fn is not None:
            return self._grad_fn(x)
        # Central finite differences fallback
        eps = 1e-5
        grad = np.empty_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (self.potential_fn(xp) - self.potential_fn(xm)) / (2 * eps)
        return grad

    def force(self, x):
        return -self.gradient(x)

    # ------------------------------------------------------------------
    # Integrator
    # ------------------------------------------------------------------

    def step(self, x, rng):
        """Single Euler-Maruyama step."""
        noise = rng.standard_normal(self.dim)
        return x + self.force(x) * self.dt + self.sigma * np.sqrt(self.dt) * noise

    def trajectory(self, x0, T, saveat=None, rng=None):
        """
        Integrate for time T starting from x0.

        Parameters
        ----------
        x0 : array-like of shape (dim,)
        T : float
            Total integration time.
        saveat : float or None
            Time interval between saved frames.  Defaults to `dt` (save every step).
        rng : numpy.random.Generator or None

        Returns
        -------
        traj : np.ndarray of shape (n_frames, dim)
        """
        if rng is None:
            rng = np.random.default_rng()
        if saveat is None:
            saveat = self.dt

        n_total    = int(round(T / self.dt))
        save_every = max(1, int(round(saveat / self.dt)))

        x      = np.asarray(x0, dtype=float).copy()
        frames = []

        for k in range(n_total):
            x = self.step(x, rng)
            if (k + 1) % save_every == 0:
                frames.append(x.copy())

        return np.stack(frames)   # (n_frames, dim)

    def lagged_pairs(self, traj, lag=1):
        """
        Slice a trajectory into aligned (X0, X_tau) pairs.

        Parameters
        ----------
        traj : np.ndarray of shape (n_frames, dim)
        lag : int
            Number of saved frames to skip.

        Returns
        -------
        X0, Xtau : np.ndarray of shape (n_samples, dim)
        """
        return traj[:-lag], traj[lag:]

    def rand_x0(self, rng=None):
        """Sample a random initial condition uniformly within `support`."""
        if rng is None:
            rng = np.random.default_rng()
        if self.support is not None:
            lo, hi = self.support[0], self.support[1]
            return rng.uniform(lo, hi)
        return rng.standard_normal(self.dim)

    def koopman_pairs(self, n_samples, lagtime=None, rng=None):
        """
        Generate (X0, X_tau) pairs by sampling x0 uniformly from `support`
        and propagating each for `lagtime` with the Langevin integrator.

        This gives uniform coverage of the PES domain, complementing
        trajectory-based pairs which may under-sample high-energy regions.

        Parameters
        ----------
        n_samples : int
            Number of independent (x0, x_tau) pairs to generate.
        lagtime : float or None
            Propagation time per sample.  Defaults to ``self.lagtime``.
        rng : numpy.random.Generator or None

        Returns
        -------
        X0, Xtau : np.ndarray of shape (n_samples, dim)
        """
        if rng is None:
            rng = np.random.default_rng()
        if lagtime is None:
            lagtime = self.lagtime

        n_steps = max(1, int(round(lagtime / self.dt)))

        X0   = np.empty((n_samples, self.dim))
        Xtau = np.empty((n_samples, self.dim))

        for i in range(n_samples):
            x = self.rand_x0(rng)
            X0[i] = x
            for _ in range(n_steps):
                x = self.step(x, rng)
            Xtau[i] = x

        return X0, Xtau
