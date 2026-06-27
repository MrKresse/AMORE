"""
chi-MEP: minimum-energy path traced through chi-level sets.

Algorithm (port of ISOKANN.jl/src/minimumpath.jl)
--------------------------------------------------
Starting from a point x₀, alternate between:

  1. Gradient step along ±∇χ(x):
         x ← x + direction · (∇χ / ‖∇χ‖²) · stepsize

     This moves x to a new chi level χ_new ≈ χ(x) ± stepsize.

  2. Level-set projection back to χ(x) = χ_new:
     - If a potential is given: minimize U(x) subject to χ(x) = χ_new (SLSQP).
     - Otherwise: Newton retraction  x += (χ_new − χ(x)) · ∇χ / ‖∇χ‖².

`reaction_path_minimum` runs the integrator in both directions from x₀,
allocating steps proportionally to χ(x₀) ∈ [0, 1]:
  backward : ⌊steps · χ₀⌋  steps  (decreasing chi, toward state A)
  forward  : ⌊steps · (1−χ₀)⌋ steps (increasing chi, toward state B)
"""

import numpy as np
import torch as pt
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Low-level torch helpers (operate on single points as 1-D numpy arrays)
# ---------------------------------------------------------------------------

def _device(nu):
    """Return the device the network's parameters live on."""
    return next(nu.parameters()).device


def _chi_val(nu, featurizer, x_np):
    """χ(x) as a Python float.  x_np: 1-D numpy array."""
    x_t = pt.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(_device(nu))
    with pt.no_grad():
        return nu(featurizer(x_t)).reshape(-1)[0].item()


def _chi_and_grad(nu, featurizer, x_np):
    """
    Returns (χ(x): float, ∇_x χ(x): 1-D float64 numpy array).
    Gradient is computed via autograd through the featurizer and network.
    """
    x_t = pt.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(_device(nu)).requires_grad_(True)
    chi = nu(featurizer(x_t)).reshape(-1)[0]
    (grad,) = pt.autograd.grad(chi, x_t)
    return chi.item(), grad.squeeze().detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Level-set projection
# ---------------------------------------------------------------------------

def levelset_retract(nu, featurizer, x, chi_target, max_steps=20, tol=1e-8):
    """
    Newton iteration to enforce χ(x) ≈ chi_target.

        x ← x + (χ_target − χ(x)) · ∇χ(x) / ‖∇χ(x)‖²

    Parameters
    ----------
    x : np.ndarray of shape (dim,)
    chi_target : float
    """
    x = x.copy()
    for _ in range(max_steps):
        chi_val, g = _chi_and_grad(nu, featurizer, x)
        err = chi_target - chi_val
        if abs(err) < tol:
            break
        x = x + err * g / (np.dot(g, g) + 1e-12)
    return x


def energy_min_on_levelset(nu, featurizer, potential_fn, x0, chi_target,
                            grad_fn=None, tol=1e-5, max_iter=100):
    """
    Minimize potential_fn(x) subject to χ(x) = chi_target using SLSQP.

    Parameters
    ----------
    potential_fn : callable
        U(x) -> float, x is a 1-D float64 numpy array.
    grad_fn : callable or None
        ∇U(x) -> 1-D float64 numpy array.  If None, scipy uses finite differences.
    tol : float
        Function tolerance for SLSQP (ftol).
    max_iter : int

    Returns
    -------
    x_opt : np.ndarray of shape (dim,)
    """
    constraint = {
        "type": "eq",
        "fun": lambda x: _chi_val(nu, featurizer, x) - chi_target,
        "jac": lambda x: _chi_and_grad(nu, featurizer, x)[1],
    }

    result = minimize(
        potential_fn,
        x0.astype(np.float64),
        jac=grad_fn,
        method="SLSQP",
        constraints=constraint,
        options={"ftol": tol, "maxiter": max_iter, "disp": False},
    )
    return result.x


# ---------------------------------------------------------------------------
# Core integrator
# ---------------------------------------------------------------------------

def reaction_integrator(nu, featurizer, x0, steps=50, stepsize=0.01, direction=1,
                        potential_fn=None, grad_fn=None,
                        energy_tol=1e-5, energy_max_iter=100,
                        retract_tol=1e-8, retract_steps=20):
    """
    Integrate along ±∇χ from x0 for `steps` steps, projecting each point
    back onto its chi-level set.

    Parameters
    ----------
    nu : torch.nn.Module
        Trained chi network.
    featurizer : callable
        Maps a (1, dim) torch.Tensor to features consumed by nu.
        Use ``lambda x: x`` for systems where coordinates are the features.
    x0 : np.ndarray of shape (dim,)
        Starting point.
    steps : int
        Number of integration steps.
    stepsize : float
        Step length along ∇χ.  Because the step is normalised by ‖∇χ‖²,
        each step changes χ by approximately `stepsize`.
    direction : +1 or -1
        +1 integrates toward increasing χ (state B),
        -1 integrates toward decreasing χ (state A).
    potential_fn : callable or None
        If provided, each point is found by minimizing U on the chi level set
        rather than pure Newton retraction.
    grad_fn : callable or None
        ∇U for SLSQP.  Required (or None for finite-diff fallback) when
        potential_fn is given.
    energy_tol, energy_max_iter : float, int
        Tolerances for SLSQP energy minimization.
    retract_tol, retract_steps : float, int
        Tolerances for Newton retraction fallback.

    Returns
    -------
    path : np.ndarray of shape (steps, dim)
    """
    x = x0.copy().astype(np.float64)
    path = np.empty((steps, len(x)))

    for i in range(steps):
        _, g = _chi_and_grad(nu, featurizer, x)
        g_norm2 = np.dot(g, g) + 1e-12
        x = x + direction * g / g_norm2 * stepsize

        chi_target = _chi_val(nu, featurizer, x)

        if potential_fn is not None:
            x = energy_min_on_levelset(
                nu, featurizer, potential_fn, x, chi_target,
                grad_fn=grad_fn, tol=energy_tol, max_iter=energy_max_iter,
            )
        else:
            x = levelset_retract(
                nu, featurizer, x, chi_target,
                max_steps=retract_steps, tol=retract_tol,
            )

        path[i] = x

    return path


# ---------------------------------------------------------------------------
# Full reaction path
# ---------------------------------------------------------------------------

def reaction_path_minimum(nu, featurizer, x0, steps=100, stepsize=0.01,
                          potential_fn=None, grad_fn=None, **integrator_kwargs):
    """
    Compute a chi-MEP starting from x0.

    Integrates backward (direction=-1) and forward (direction=+1) from x0,
    allocating steps proportionally to χ(x₀):
      backward : ⌊steps · χ₀⌋   steps  →  approaches χ ≈ 0 (state A)
      forward  : ⌊steps · (1−χ₀)⌋ steps  →  approaches χ ≈ 1 (state B)

    Parameters
    ----------
    nu : torch.nn.Module
    featurizer : callable
    x0 : np.ndarray of shape (dim,)
        Starting point (typically from the transition-state region χ ≈ 0.5).
    steps : int
        Total path points (split between the two directions).
    stepsize : float
    potential_fn, grad_fn : optional
        Passed to reaction_integrator for energy-minimised projection.
    **integrator_kwargs
        Forwarded to reaction_integrator (retract_tol, energy_tol, etc.).

    Returns
    -------
    path : np.ndarray of shape (n_path, dim)
        Full path [A-end, ..., x0, ..., B-end].

    Notes
    -----
    The initial state x0 is first projected onto its own chi level set with the SAME
    operator used for every other image — energy minimisation when a potential is given,
    otherwise Newton retraction — so the central image is treated identically to the
    forward/backward images instead of being patched in raw.
    """
    x0 = np.asarray(x0, dtype=np.float64).copy()
    chi0 = _chi_val(nu, featurizer, x0)

    # Treat the initial state like every other image: project it onto its level set.
    if potential_fn is not None:
        x0 = energy_min_on_levelset(
            nu, featurizer, potential_fn, x0, chi0, grad_fn=grad_fn,
            tol=integrator_kwargs.get("energy_tol", 1e-5),
            max_iter=integrator_kwargs.get("energy_max_iter", 100),
        )
    else:
        x0 = levelset_retract(
            nu, featurizer, x0, chi0,
            max_steps=integrator_kwargs.get("retract_steps", 20),
            tol=integrator_kwargs.get("retract_tol", 1e-8),
        )
    chi0 = _chi_val(nu, featurizer, x0)
    steps_back = max(int(steps * chi0), 1)
    steps_fwd  = max(int(steps * (1 - chi0)), 1)

    path_back = reaction_integrator(
        nu, featurizer, x0,
        steps=steps_back, stepsize=stepsize, direction=-1,
        potential_fn=potential_fn, grad_fn=grad_fn, **integrator_kwargs,
    )[::-1]   # reverse: now ordered from A-end toward x0

    path_fwd = reaction_integrator(
        nu, featurizer, x0,
        steps=steps_fwd, stepsize=stepsize, direction=1,
        potential_fn=potential_fn, grad_fn=grad_fn, **integrator_kwargs,
    )

    return np.concatenate([path_back, x0[None, :], path_fwd], axis=0)


# ---------------------------------------------------------------------------
# Transition-state sampling
# ---------------------------------------------------------------------------

def transition_state(nu, featurizer, xs, chi_lo=0.45, chi_hi=0.55):
    """
    Extract frames with χ(x) ∈ [chi_lo, chi_hi].

    Parameters
    ----------
    xs : np.ndarray of shape (n_frames, dim)
    chi_lo, chi_hi : float
        Chi window defining the transition region.

    Returns
    -------
    selected : np.ndarray of shape (n_selected, dim)
    chi_selected : np.ndarray of shape (n_selected,)
    """
    x_t = pt.from_numpy(xs.astype(np.float32)).to(_device(nu))
    with pt.no_grad():
        chi_vals = nu(featurizer(x_t)).reshape(len(xs), -1)[:, 0].cpu().numpy()

    mask = (chi_vals >= chi_lo) & (chi_vals <= chi_hi)
    return xs[mask], chi_vals[mask]
