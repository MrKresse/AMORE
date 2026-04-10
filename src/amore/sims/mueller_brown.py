"""
Mueller-Brown potential and preconfigured Langevin simulator.

Potential (4-Gaussian form, standard parametrisation):

    U(x, y) = Σ_k  A_k · exp(a_k (x−x0_k)²  +  b_k (x−x0_k)(y−y0_k)  +  c_k (y−y0_k)²)

Parameters
----------
k   A      a      b      c     x0    y0
0  -200   -1      0    -10     1     0
1  -100   -1      0    -10     0     0.5
2  -170   -6.5   11    -6.5  -0.5   1.5
3    15    0.7    0.6   0.7  -1     1

Minima are near (-0.55, 1.45) and (0.62, 0.03).
Suggested display window: x ∈ [-1.5, 1.0], y ∈ [-0.5, 2.0].
"""

import numpy as np
from .langevin import LangevinSimulator

# -------------------------------------------------------------------
# Parameters (match exactly the Julia ISOKANN implementation)
# -------------------------------------------------------------------
_A  = np.array([-200.0, -100.0, -170.0,  15.0])
_a  = np.array([  -1.0,   -1.0,   -6.5,   0.7])
_b  = np.array([   0.0,    0.0,   11.0,   0.6])
_c  = np.array([ -10.0,  -10.0,   -6.5,   0.7])
_x0 = np.array([   1.0,    0.0,   -0.5,  -1.0])
_y0 = np.array([   0.0,    0.5,    1.5,   1.0])


def potential(xy):
    """
    Mueller-Brown potential energy.

    Parameters
    ----------
    xy : array-like of length 2
        Coordinates [x, y].

    Returns
    -------
    float
    """
    x, y = float(xy[0]), float(xy[1])
    dx = x - _x0
    dy = y - _y0
    exponents = _a * dx**2 + _b * dx * dy + _c * dy**2
    return float(np.dot(_A, np.exp(exponents)))


def gradient(xy):
    """
    Analytic gradient of the Mueller-Brown potential.

    Returns
    -------
    grad : np.ndarray of shape (2,)
        [∂U/∂x, ∂U/∂y]
    """
    x, y = float(xy[0]), float(xy[1])
    dx = x - _x0
    dy = y - _y0
    exponents = _a * dx**2 + _b * dx * dy + _c * dy**2
    e = _A * np.exp(exponents)           # (4,) weighted exponentials

    dfdx = 2 * _a * dx + _b * dy        # ∂f_k/∂x  for each k
    dfdy = _b * dx + 2 * _c * dy        # ∂f_k/∂y  for each k

    return np.array([np.dot(e, dfdx), np.dot(e, dfdy)])


def MuellerBrown(dt=1e-4, lagtime=1e-3, sigma=1.0):
    """
    Preconfigured Langevin simulator for the Mueller-Brown potential.

    Parameters
    ----------
    dt : float
        Integration time step.
    lagtime : float
        Default lag time for lagged-pair generation.
    sigma : float
        Noise amplitude.  Increase (e.g. sigma=5) for faster barrier crossings.

    Returns
    -------
    LangevinSimulator
    """
    support = np.array([[-1.5, -0.5],   # [lo_x, lo_y]
                        [ 1.0,  2.0]])  # [hi_x, hi_y]

    return LangevinSimulator(
        potential_fn = potential,
        grad_fn      = gradient,
        dim          = 2,
        sigma        = sigma,
        dt           = dt,
        lagtime      = lagtime,
        support      = support,
    )
