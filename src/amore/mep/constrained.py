"""
GPU-resident constrained Langevin sampling on chi level sets.

Uses a stiff harmonic restraint ½κ(ξ(x)−c)² implemented via
TorchForce + CustomCVForce (openmm-torch).  All force evaluations
stay on the GPU — no CPU↔GPU transfers during MD.

Algorithm (sequential χ-MEP construction)
------------------------------------------
1. Start from x₀ on levelset ξ(x₀) = c₀
2. Run constrained Langevin on that levelset; select medoid
3. Euler step from medoid along ±∇χ → seed for next levelset
4. Repeat in +∇χ direction (forward branch) and −∇χ (backward branch)
5. Combine backward[::-1] + initial + forward

Public API
----------
export_cv_torchscript      -- trace chi network → .pt file for TorchForce
build_constrained_system   -- add restraint + optional grad-norm reporter
ConstrainedCVReporter      -- collects xi, lambda, positions during MD
run_constrained_sampling   -- drive the simulation
select_medoid              -- pick representative from level-set samples
euler_step_to_levelset     -- Euler+Newton step to next chi level set
sample_single_levelset     -- sample one level set, return reporter results
build_chi_mep_constrained  -- construct the full sequential χ-MEP
"""

import os
import copy

import numpy as np
import torch as pt
from scipy.spatial.distance import pdist, squareform

try:
    import openmm as mm
    from openmm import app, unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

try:
    from openmmtorch import TorchForce
    HAS_OPENMMTORCH = True
except ImportError:
    HAS_OPENMMTORCH = False

from amore.mep.core import _chi_and_grad, _chi_val, levelset_retract


# ---------------------------------------------------------------------------
# Ensure conda-env OpenMM plugins (CUDA, CPU, …) are registered.
# The default plugin directory points to a system path; for conda installs
# the actual plugins live in $CONDA_ENV/lib/plugins/.
# ---------------------------------------------------------------------------
def _load_openmm_plugins():
    if not HAS_OPENMM:
        return
    # mm.__file__ is .../envs/ENV/lib/python3.X/site-packages/openmm/openmm.py
    # Going up 4 parents reaches .../envs/ENV/lib/
    # The plugins are in .../envs/ENV/lib/plugins/
    import pathlib
    candidates = [
        # conda env lib/plugins  (4 parents from openmm.py → lib/, then plugins/)
        str(pathlib.Path(mm.__file__).parent.parent.parent.parent / "plugins"),
        # OPENMM_PLUGIN_DIR env var override
        os.environ.get("OPENMM_PLUGIN_DIR", ""),
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            mm.Platform.loadPluginsFromDirectory(d)
            break

_load_openmm_plugins()


# ---------------------------------------------------------------------------
# Pairwise-distance featurizer as a nn.Module (traceable / scriptable)
# ---------------------------------------------------------------------------

class _PairDistFeaturizer(pt.nn.Module):
    """
    Computes pairwise Euclidean distances for given atom-index pairs.

    Input : (1, n_atoms*3) float32 tensor (flat coordinates, nm)
    Output: (1, n_pairs)   float32 tensor
    """

    def __init__(self, pairs):
        super().__init__()
        self.register_buffer(
            "pairs", pt.tensor(np.asarray(pairs, dtype=np.int64), dtype=pt.long)
        )

    def forward(self, coords_flat):                       # (1, n_atoms*3)
        n_atoms = coords_flat.shape[1] // 3
        xyz  = coords_flat.view(1, n_atoms, 3)
        i    = self.pairs[:, 0]
        j    = self.pairs[:, 1]
        diff = xyz[:, i, :] - xyz[:, j, :]
        return pt.linalg.norm(diff, dim=-1)               # (1, n_pairs)


class _CVPipeline(pt.nn.Module):
    """
    Full chi pipeline for TorchForce.

    Input : positions (n_atoms, 3) float32, nm
    Output: scalar chi value
    """

    def __init__(self, nu, pairs):
        super().__init__()
        self.feat = _PairDistFeaturizer(pairs)
        self.nu   = nu

    def forward(self, positions):                         # (n_atoms, 3) → ()
        flat  = positions.reshape(1, -1)                  # (1, n_atoms*3)
        feats = self.feat(flat)                           # (1, n_pairs)
        chi   = self.nu(feats)                            # (1,) or ()
        return chi.reshape(())                            # scalar


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------

def export_cv_torchscript(nu, pairs, example_positions_nm, path):
    """
    Trace the chi pipeline and save it as a TorchScript (.pt) file
    suitable for use with openmmtorch.TorchForce.

    The exported module has the signature::

        forward(positions: Tensor[n_atoms, 3]) -> Tensor[()]   (scalar)

    where positions are in nm (float32).

    Parameters
    ----------
    nu : torch.nn.Module
        Trained chi network.  Must already be on the desired device.
    pairs : array-like, shape (n_pairs, 2)
        Atom-index pairs used as pairwise-distance features.
    example_positions_nm : np.ndarray, shape (n_atoms, 3)
        A representative configuration in nanometres.  Used only for tracing.
    path : str
        Output file path for the .pt model.
    """
    device = next(nu.parameters()).device
    nu.eval()

    pipeline = _CVPipeline(nu, pairs).to(device)
    pipeline.eval()

    pos_t = pt.from_numpy(
        example_positions_nm.astype(np.float32)
    ).to(device)                                          # (n_atoms, 3)

    with pt.no_grad():
        traced = pt.jit.trace(pipeline, pos_t)

    traced.save(path)
    return path


# ---------------------------------------------------------------------------
# OpenMM system builder
# ---------------------------------------------------------------------------

def build_constrained_system(system, cv_model_path, target_cv_value, kappa):
    """
    Add a harmonic CV restraint ½κ(ξ−c)² to a copy of *system*.

    The TorchForce is wrapped in a CustomCVForce so that TorchForce serves
    as a collective variable; CustomCVForce handles the force calculation
    entirely within OpenMM (GPU-resident).

    Parameters
    ----------
    system : openmm.System
        Existing physical system.  A deep copy is made so the original is
        not modified.
    cv_model_path : str
        Path to the TorchScript CV model (.pt file).
    target_cv_value : float
        Target level-set value c.
    kappa : float
        Restraint stiffness in kJ/mol (per CV-unit²).

    Returns
    -------
    new_system : openmm.System
        Copy of the input system with the restraint added.
    cv_force_index : int
        Index of the CustomCVForce inside new_system (for querying CV values).
    """
    if not HAS_OPENMM:
        raise ImportError("openmm is required")
    if not HAS_OPENMMTORCH:
        raise ImportError("openmmtorch (openmm-torch plugin) is required")

    new_system = copy.deepcopy(system)

    torch_force = TorchForce(cv_model_path)
    torch_force.setUsesPeriodicBoundaryConditions(
        new_system.usesPeriodicBoundaryConditions()
    )

    cv_restraint = mm.CustomCVForce("0.5 * kappa * (xi - c)^2")
    cv_restraint.addGlobalParameter("kappa", float(kappa))
    cv_restraint.addGlobalParameter("c",     float(target_cv_value))
    cv_restraint.addCollectiveVariable("xi", torch_force)

    cv_force_index = new_system.addForce(cv_restraint)
    return new_system, cv_force_index


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class ConstrainedCVReporter:
    """
    Collect xi, lambda (Lagrange multiplier), deviation, and positions
    during a constrained Langevin run.

    Parameters
    ----------
    report_interval : int
        Steps between reports.
    cv_force_index : int
        Index of the CustomCVForce inside the system.
    kappa : float
        Current restraint stiffness (kJ/mol).
    target_c : float
        Current target CV value.
    total_steps : int
        Total production steps (pre-allocates arrays).
    n_particles : int
        Number of particles.
    """

    def __init__(self, report_interval, cv_force_index,
                 kappa, target_c, total_steps, n_particles):
        self.report_interval = report_interval
        self.cv_force_idx    = cv_force_index
        self.kappa           = kappa
        self.target_c        = target_c

        n_frames = (total_steps + report_interval - 1) // report_interval
        self.xi_values     = np.zeros(n_frames)
        self.lambda_values = np.zeros(n_frames)
        self.deviations    = np.zeros(n_frames)
        self.positions     = np.zeros((n_frames, n_particles, 3))
        self._frame        = 0

    def report(self, simulation):
        ctx   = simulation.context
        state = ctx.getState(getPositions=True)
        pos   = (state.getPositions(asNumpy=True)
                 .value_in_unit(unit.nanometers))         # (n_atoms, 3) nm

        xi_arr = (simulation.system
                  .getForce(self.cv_force_idx)
                  .getCollectiveVariableValues(ctx))
        xi  = float(xi_arr[0])
        dev = xi - self.target_c
        lam = self.kappa * dev

        i = self._frame
        self.xi_values[i]     = xi
        self.lambda_values[i] = lam
        self.deviations[i]    = abs(dev)
        self.positions[i]     = pos
        self._frame += 1

    def get_results(self):
        """Return dict with arrays truncated to collected frames."""
        n = self._frame
        return {
            "xi":        self.xi_values[:n],
            "lambda":    self.lambda_values[:n],
            "deviation": self.deviations[:n],
            "positions": self.positions[:n],          # (n, n_atoms, 3) nm
        }


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_constrained_sampling(simulation, reporter, total_steps,
                              report_interval, warn_threshold=0.01):
    """
    Drive a constrained Langevin simulation, calling ``reporter.report``
    every ``report_interval`` steps.

    Parameters
    ----------
    simulation : openmm.app.Simulation
    reporter : ConstrainedCVReporter
    total_steps : int
    report_interval : int
    warn_threshold : float
        Print a warning when |ξ−c| exceeds this value.

    Returns
    -------
    results : dict  (from reporter.get_results())
    """
    for _ in range(0, total_steps, report_interval):
        simulation.step(report_interval)
        reporter.report(simulation)

        dev = reporter.deviations[reporter._frame - 1]
        if dev > warn_threshold:
            step_idx = reporter._frame * report_interval
            print(f"  WARNING step {step_idx}: |ξ−c| = {dev:.4f}")

    return reporter.get_results()


# ---------------------------------------------------------------------------
# Medoid selection
# ---------------------------------------------------------------------------

def select_medoid(positions):
    """
    Select the medoid (minimum mean pairwise distance) from sampled frames.

    Parameters
    ----------
    positions : np.ndarray, shape (n_frames, n_particles, 3)

    Returns
    -------
    medoid_idx : int
    medoid_pos : np.ndarray, shape (n_particles, 3)
    """
    n_frames = positions.shape[0]
    if n_frames == 1:
        return 0, positions[0]

    flat        = positions.reshape(n_frames, -1)
    dist_matrix = squareform(pdist(flat))
    mean_dists  = dist_matrix.mean(axis=1)
    idx         = int(np.argmin(mean_dists))
    return idx, positions[idx]


# ---------------------------------------------------------------------------
# Euler step to next level set
# ---------------------------------------------------------------------------

def euler_step_to_levelset(nu, featurizer, positions_nm, step_size,
                            max_newton_iters=10, tol=1e-3):
    """
    From medoid at ξ = c, take an Euler step along ±∇χ to reach
    levelset ξ ≈ c + step_size.

    positions_nm : np.ndarray (n_atoms, 3)  –OR–  (n_atoms*3,)
        Particle coordinates in **nanometres**.
    step_size : float
        Signed CV increment.  Positive = toward state B (+∇χ).

    Returns
    -------
    new_pos : np.ndarray, shape (n_atoms, 3) nm
    new_c   : float  (actual CV value after Newton correction)
    """
    flat_nm   = np.asarray(positions_nm, dtype=np.float64).flatten()
    xi_val, g = _chi_and_grad(nu, featurizer, flat_nm)
    target_c  = xi_val + step_size

    g_norm2 = np.dot(g, g) + 1e-30
    x = flat_nm + (step_size / g_norm2) * g      # Euler step (flat nm)

    # Newton correction to land on target_c
    for _ in range(max_newton_iters):
        xi_new, g_new = _chi_and_grad(nu, featurizer, x)
        residual = xi_new - target_c
        if abs(residual) < tol:
            break
        g_norm2_new = np.dot(g_new, g_new) + 1e-30
        x = x - (residual / g_norm2_new) * g_new

    xi_final, _ = _chi_and_grad(nu, featurizer, x)
    n_atoms      = x.size // 3
    return x.reshape(n_atoms, 3), xi_final


# ---------------------------------------------------------------------------
# Sample a single level set
# ---------------------------------------------------------------------------

def sample_single_levelset(
    system_factory,
    cv_model_path,
    initial_positions_nm,
    target_c,
    kappa,
    temperature,
    friction,
    dt,
    steps,
    report_interval,
    equilibration_steps=1000,
    platform_name="CUDA",
):
    """
    Run constrained Langevin on levelset ξ = target_c from a seed configuration.

    Parameters
    ----------
    system_factory : callable() -> (openmm.System, openmm.app.Topology)
        Called once per level set to produce a fresh system and topology.
    cv_model_path : str
        Path to the TorchScript CV model.
    initial_positions_nm : np.ndarray, shape (n_atoms, 3)
        Seed configuration in nanometres.
    target_c : float
    kappa : float
        Restraint stiffness (kJ/mol).
    temperature : float  (K)
    friction : float     (ps⁻¹)
    dt : float           (ps)
    steps : int
        Production steps.
    report_interval : int
    equilibration_steps : int
    platform_name : str

    Returns
    -------
    results : dict (from ConstrainedCVReporter.get_results())
    """
    #if not HAS_OPENMM or not HAS_OPENMMTORCH:
    #    raise ImportError("openmm and openmmtorch are required")

    base_system, topology = system_factory()
    system, cv_idx = build_constrained_system(
        base_system, cv_model_path, target_c, kappa
    )

    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction    / unit.picosecond,
        dt          * unit.picoseconds,
    )

    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        plat_kw  = {"Precision": "mixed"}
        simulation = app.Simulation(topology, system, integrator, platform, plat_kw)
    except Exception:
        # fall back to default platform
        simulation = app.Simulation(topology, system, integrator)

    simulation.context.setPositions(initial_positions_nm)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    # Equilibrate without reporting
    if equilibration_steps > 0:
        simulation.step(equilibration_steps)

    n_particles = system.getNumParticles()
    reporter = ConstrainedCVReporter(
        report_interval, cv_idx, kappa, target_c, steps, n_particles
    )
    results = run_constrained_sampling(simulation, reporter, steps, report_interval)

    mean_dev = np.mean(results["deviation"])
    max_dev  = np.max(results["deviation"])
    print(f"  Levelset ξ={target_c:.4f}: |ξ−c| mean={mean_dev:.5f}, max={max_dev:.5f}")

    return results


# ---------------------------------------------------------------------------
# Sequential χ-MEP construction
# ---------------------------------------------------------------------------

def build_chi_mep_constrained(
    system_factory,
    cv_model_path,
    nu,
    featurizer,
    initial_positions_nm,
    kappa,
    cv_step_size,
    n_images_forward,
    n_images_backward,
    steps_per_levelset,
    report_interval,
    temperature=300.0,
    friction=1.0,
    dt=2e-3,
    equilibration_steps=1000,
    platform_name="CUDA",
):
    """
    Build the sequential χ-MEP from initial_positions_nm.

    The MEP is constructed level set by level set:
      backward branch (−∇χ) → initial level set → forward branch (+∇χ)

    Parameters
    ----------
    system_factory : callable() -> (openmm.System, openmm.app.Topology)
        Creates a fresh physical system for each level set.
    cv_model_path : str
        Path to the TorchScript .pt model (from export_cv_torchscript).
    nu : torch.nn.Module
        Trained chi network (for Euler steps / Newton corrections on CPU).
    featurizer : callable
        Feature function matching the network input.
    initial_positions_nm : np.ndarray, shape (n_atoms, 3) OR (n_atoms*3,)
        Starting configuration in nm.
    kappa : float
        Restraint stiffness (kJ/mol).
    cv_step_size : float
        Unsigned Δc between adjacent images.
    n_images_forward : int
        Steps in the +∇χ direction (toward state B).
    n_images_backward : int
        Steps in the −∇χ direction (toward state A).
    steps_per_levelset : int
        Production MD steps at each level set.
    report_interval : int
        Reporting stride (positions saved every this many steps).
    temperature, friction, dt : float
        Langevin parameters (K, ps⁻¹, ps).
    equilibration_steps : int
        Un-reported equilibration before each level-set production run.
    platform_name : str
        OpenMM platform ("CUDA", "OpenCL", or "CPU").

    Returns
    -------
    dict with keys
      "images"     : list of np.ndarray (n_atoms, 3) — medoid configs, A→B order
      "cv_values"  : list of float — CV value at each image
      "results"    : list of result dicts (one per level set)
    """
    init_pos = np.asarray(initial_positions_nm, dtype=np.float64)
    n_atoms  = init_pos.size // 3
    if init_pos.ndim == 1:
        init_pos = init_pos.reshape(n_atoms, 3)

    _sampling_kwargs = dict(
        kappa              = kappa,
        temperature        = temperature,
        friction           = friction,
        dt                 = dt,
        steps              = steps_per_levelset,
        report_interval    = report_interval,
        equilibration_steps= equilibration_steps,
        platform_name      = platform_name,
    )

    # ── Step 1: initial level set ──────────────────────────────────────────
    xi_init = _chi_val(nu, featurizer, init_pos.flatten())
    print(f"Initial ξ = {xi_init:.4f}")

    res_init = sample_single_levelset(
        system_factory, cv_model_path, init_pos, xi_init, **_sampling_kwargs
    )
    _, medoid_init = select_medoid(res_init["positions"])

    # ── Step 2: forward branch ─────────────────────────────────────────────
    fwd_images, fwd_cv, fwd_results = [], [], []
    current_medoid = medoid_init

    for i in range(n_images_forward):
        print(f"\nForward image {i+1}/{n_images_forward}")
        next_pos, next_c = euler_step_to_levelset(
            nu, featurizer, current_medoid, +cv_step_size
        )
        print(f"  Euler → ξ = {next_c:.4f}")

        res_i = sample_single_levelset(
            system_factory, cv_model_path, next_pos, next_c, **_sampling_kwargs
        )
        _, medoid_i = select_medoid(res_i["positions"])

        fwd_images.append(medoid_i)
        fwd_cv.append(next_c)
        fwd_results.append(res_i)
        current_medoid = medoid_i

    # ── Step 3: backward branch ────────────────────────────────────────────
    bwd_images, bwd_cv, bwd_results = [], [], []
    current_medoid = medoid_init

    for i in range(n_images_backward):
        print(f"\nBackward image {i+1}/{n_images_backward}")
        next_pos, next_c = euler_step_to_levelset(
            nu, featurizer, current_medoid, -cv_step_size
        )
        print(f"  Euler → ξ = {next_c:.4f}")

        res_i = sample_single_levelset(
            system_factory, cv_model_path, next_pos, next_c, **_sampling_kwargs
        )
        _, medoid_i = select_medoid(res_i["positions"])

        bwd_images.append(medoid_i)
        bwd_cv.append(next_c)
        bwd_results.append(res_i)
        current_medoid = medoid_i

    # ── Step 4: combine (A-end → B-end) ───────────────────────────────────
    all_images  = list(reversed(bwd_images))  + [medoid_init] + fwd_images
    all_cv      = list(reversed(bwd_cv))      + [xi_init]     + fwd_cv
    all_results = list(reversed(bwd_results)) + [res_init]    + fwd_results

    print(f"\nχ-MEP complete: {len(all_images)} images, "
          f"ξ ∈ [{all_cv[0]:.4f}, {all_cv[-1]:.4f}]")

    return {
        "images":    all_images,
        "cv_values": all_cv,
        "results":   all_results,
    }


# ---------------------------------------------------------------------------
# CPU-retraction fallback: no TorchForce / openmmtorch required
# ---------------------------------------------------------------------------

def sample_levelset_projected(sim, nu, featurizer, x0, chi_level, steps, burnin=0,
                              time_breakdown=False):
    """
    Projected Langevin on the chi = chi_level surface.

    At every step:
      1. OpenMM forces  (GPU)
      2. Project F orthogonal to ∇χ; record mean-force lambda = −F·∇χ/‖∇χ‖²
      3. Langevin velocity/position update
      4. Newton retraction onto chi_level
      5. Record Fixman Z = Σ_i (∇χ_i)²/m_i  and per-atom sensitivity

    Parameters
    ----------
    sim : OpenMMSimulator
    nu, featurizer : chi network
    x0 : np.ndarray (n_atoms*3,)
    chi_level : float
    steps : int
    burnin : int
        Steps to run before recording begins (discarded; not counted in output).

    Returns
    -------
    dict with keys
      "positions"         : (steps, n_atoms*3)  nm
      "lambdas"           : (steps,)  mean-force samples  [kJ/mol]
      "Zs"                : (steps,)  Fixman Z = Σ (∇χ)²/m  [nm⁻² mol/g]
      "sensitivity"       : (n_atoms,)  time-avg mass-weighted ‖∇χ‖² per atom
    """
    ctx   = sim._sim.context
    integ = ctx.getIntegrator()
    dt    = sim._dt
    temp  = sim._temp
    gamma = integ.getFriction().value_in_unit(unit.picosecond**-1)
    kBT   = 0.008314463 * temp                          # kJ/mol

    system = sim._sim.system
    n_particles = system.getNumParticles()
    m_per_atom = np.array(
        [system.getParticleMass(i).value_in_unit(unit.amu) for i in range(n_particles)],
        dtype=np.float64,
    )                                                   # (n_atoms,)
    m = np.repeat(m_per_atom, 3)                        # (n_atoms*3,)

    x = x0.copy().astype(np.float64).flatten()
    v = np.zeros_like(x)

    import time
    t_openmm = t_chi_grad = t_retract = 0.0

    def _step(x, v):
        nonlocal t_openmm, t_chi_grad, t_retract

        t0 = time.perf_counter()
        ctx.setPositions(x.reshape(n_particles, 3))
        F = (ctx.getState(getForces=True)
             .getForces(asNumpy=True)
             .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
             .flatten().astype(np.float64))
        t_openmm += time.perf_counter() - t0

        t0 = time.perf_counter()
        _, dchi = _chi_and_grad(nu, featurizer, x)
        dchi_sq = np.dot(dchi, dchi) + 1e-30
        F_proj  = np.dot(F, dchi) / dchi_sq
        F      -= F_proj * dchi
        t_chi_grad += time.perf_counter() - t0

        db = np.random.randn(len(x))
        v += (1.0 / m) * ((F - gamma * v) * dt + np.sqrt(2.0 * gamma * kBT * dt) * db)
        x += v * dt

        # Newton retraction: reuse pre-step dchi for the correction direction
        # (saves one full backward pass per step); only the chi value is needed
        # to compute the residual.
        t0 = time.perf_counter()
        chi_new = _chi_val(nu, featurizer, x)
        x -= ((chi_new - chi_level) / dchi_sq) * dchi
        t_retract += time.perf_counter() - t0

        return x, v, F_proj, dchi

    # burn-in: run without recording
    for _ in range(burnin):
        x, v, _, _ = _step(x, v)

    positions   = np.empty((steps, len(x)))
    lambdas     = np.empty(steps)
    Zs          = np.empty(steps)
    sensitivity = np.zeros(n_particles)

    for j in range(steps):
        x, v, F_proj, dchi = _step(x, v)

        dchi_sq_per_atom = (dchi.reshape(n_particles, 3) ** 2).sum(axis=1)
        Z = float(np.sum(dchi_sq_per_atom / m_per_atom))

        positions[j]    = x
        lambdas[j]      = -F_proj
        Zs[j]           = Z
        sensitivity    += dchi_sq_per_atom / m_per_atom

    sensitivity /= steps

    chi_final = _chi_val(nu, featurizer, x)
    total = t_openmm + t_chi_grad + t_retract + 1e-12
    print(f"  Levelset ξ={chi_level:.4f}: final ξ={chi_final:.4f}, "
          f"mean λ={lambdas.mean():.3f} kJ/mol, mean Z={Zs.mean():.4g}")
    if time_breakdown:
        n = steps + burnin
        print(f"    time/step: OpenMM {t_openmm/n*1e3:.2f} ms  "
              f"χ-grad {t_chi_grad/n*1e3:.2f} ms  "
              f"retract {t_retract/n*1e3:.2f} ms  "
              f"| {t_openmm/total*100:.0f}% / {t_chi_grad/total*100:.0f}% / {t_retract/total*100:.0f}%")
    return {
        "positions":   positions,
        "lambdas":     lambdas,
        "Zs":          Zs,
        "sensitivity": sensitivity,
    }


# kept for backward compatibility — wraps sample_levelset_projected with the
# old signature (system_factory, temperature, friction, dt, …)
def sample_single_levelset_projected(
    system_factory,
    nu,
    featurizer,
    initial_positions_nm,
    target_c,
    temperature,
    friction,
    dt,
    steps,
    report_interval,
    retract_interval=1,
    equilibration_steps=1000,
    platform_name="CUDA",
    retract_tol=1e-6,
    retract_max_iter=20,
):
    """
    Run Langevin MD constrained to chi ≈ target_c using periodic CPU retraction.

    Every ``retract_interval`` MD steps:
      1. Pull positions from OpenMM context  (GPU → CPU)
      2. Newton-retract onto chi = target_c  via PyTorch autograd  (GPU chi)
      3. Push corrected positions back       (CPU → GPU)

    No TorchForce or openmmtorch plugin is required.

    Parameters
    ----------
    system_factory : callable() -> (openmm.System, openmm.app.Topology)
    nu : torch.nn.Module
    featurizer : callable
        Maps a (1, n_atoms*3) torch.Tensor → features for *nu*.
    initial_positions_nm : np.ndarray, shape (n_atoms, 3)
    target_c : float
    temperature : float  (K)
    friction : float     (ps⁻¹)
    dt : float           (ps)
    steps : int
        Production steps.
    report_interval : int
        Save a frame every this many steps.
    retract_interval : int
        Apply Newton retraction every this many steps.
    equilibration_steps : int
        Un-reported equilibration (retraction applied here too).
    platform_name : str
    retract_tol, retract_max_iter : float, int
        Tolerances for Newton retraction.

    Returns
    -------
    dict with keys
      "xi"        : np.ndarray (n_frames,)
      "deviation" : np.ndarray (n_frames,) — |xi - target_c|
      "positions" : np.ndarray (n_frames, n_atoms, 3)  nm
    """
    base_system, topology = system_factory()

    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction    / unit.picosecond,
        dt          * unit.picoseconds,
    )

    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        plat_kw  = {"Precision": "mixed"}
        simulation = app.Simulation(
            topology, base_system, integrator, platform, plat_kw
        )
    except Exception:
        simulation = app.Simulation(topology, base_system, integrator)

    simulation.context.setPositions(initial_positions_nm)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    def _retract(sim):
        """Pull positions, Newton-retract, push back. Returns flat pos (nm)."""
        state  = sim.context.getState(getPositions=True)
        pos_nm = (state.getPositions(asNumpy=True)
                  .value_in_unit(unit.nanometers))          # (n_atoms, 3)
        flat   = levelset_retract(
            nu, featurizer, pos_nm.flatten(), target_c,
            max_steps=retract_max_iter, tol=retract_tol,
        )                                                   # (n_atoms*3,)
        n_atoms = flat.size // 3
        sim.context.setPositions(flat.reshape(n_atoms, 3))
        return flat

    # ── Equilibration ──────────────────────────────────────────────────────
    for _ in range(0, max(equilibration_steps, 1), retract_interval):
        simulation.step(min(retract_interval, equilibration_steps))
        _retract(simulation)

    # ── Production ─────────────────────────────────────────────────────────
    n_particles = base_system.getNumParticles()
    n_frames    = max((steps + report_interval - 1) // report_interval, 1)
    positions_out = np.zeros((n_frames, n_particles, 3))
    xi_out        = np.zeros(n_frames)
    frame         = 0
    steps_done    = 0
    last_report   = 0

    while steps_done < steps:
        chunk = min(retract_interval, steps - steps_done)
        simulation.step(chunk)
        steps_done += chunk

        flat = _retract(simulation)

        if steps_done - last_report >= report_interval:
            xi_val = _chi_val(nu, featurizer, flat)
            if frame < n_frames:
                positions_out[frame] = flat.reshape(n_particles, 3)
                xi_out[frame]        = xi_val
                frame    += 1
            last_report = steps_done

    positions_out = positions_out[:frame]
    xi_out        = xi_out[:frame]
    deviations    = np.abs(xi_out - target_c)

    mean_dev = float(np.mean(deviations)) if frame > 0 else 0.0
    max_dev  = float(np.max(deviations))  if frame > 0 else 0.0
    print(f"  Levelset ξ={target_c:.4f} [projected]: "
          f"|ξ−c| mean={mean_dev:.5f}, max={max_dev:.5f}")

    return {
        "xi":        xi_out,
        "deviation": deviations,
        "positions": positions_out,
    }


def build_chi_mep_projected(sim, nu, featurizer, x0, steps=50, steps_per_levelset=500, burnin=0,
                            store_positions=False, time_breakdown=False):
    """
    Sequential χ-MEP via projected Langevin on chi level sets.

    Mirrors ``reaction_path_minimum`` but uses ``sample_levelset_projected``
    instead of energy minimisation:
      - Integrates along ±∇χ from x0 for steps images total
      - At each level set: run projected Langevin, take medoid as next seed
      - Steps split proportionally to chi0 (same convention as reaction_path_minimum)

    Parameters
    ----------
    sim : OpenMMSimulator
    nu, featurizer : chi network
    x0 : np.ndarray (n_atoms*3,)  flat coordinates in nm
    steps : int
        Total number of chi-level images (split ~chi0 backward, ~(1-chi0) forward).
    steps_per_levelset : int
        Langevin steps at each level set.
    burnin : int
        Steps discarded at the start of each level set before recording.
    store_positions : bool
        If False (default), strip the ``positions`` array from each per-levelset
        result dict after use.  Set True only when you need the full ensemble —
        storing all frames for 50+ level sets can exhaust RAM.

    Returns
    -------
    dict with keys
      "images"    : list of np.ndarray (n_atoms*3,) — medoid configs, A→B order
      "cv_values" : list of float
    """
    x0 = np.asarray(x0, dtype=np.float64).flatten()
    chi0 = _chi_val(nu, featurizer, x0)
    cv_step = 1.0 / steps
    steps_back = max(int(steps * chi0), 1)
    steps_fwd  = max(int(steps * (1.0 - chi0)), 1)
    print(f"Initial ξ = {chi0:.4f}  ({steps_back} backward + {steps_fwd} forward images)")

    kBT = 0.008314463 * sim._temp

    # Sample and replace x0 with the medoid of its own level set
    print(f"  Sampling initial level set ξ={chi0:.4f}")
    res0 = sample_levelset_projected(sim, nu, featurizer, x0, chi0, steps_per_levelset,
                                              burnin=burnin, time_breakdown=time_breakdown)
    _, x0_medoid = select_medoid(res0["positions"].reshape(len(res0["positions"]), -1, 3))
    x0_medoid = x0_medoid.flatten()

    all_results = [res0]   # ordered A→B; filled in below then sorted

    def _run_branch(direction, n_steps):
        results, images, cv_vals = [], [], []
        x = x0_medoid.copy()
        for i in range(n_steps):
            x, c = euler_step_to_levelset(nu, featurizer, x, direction * cv_step)
            print(f"  {'Fwd' if direction > 0 else 'Bwd'} {i+1}/{n_steps}  ξ → {c:.4f}")
            res = sample_levelset_projected(sim, nu, featurizer, x, c, steps_per_levelset,
                                           burnin=burnin, time_breakdown=time_breakdown)
            _, x = select_medoid(res["positions"].reshape(len(res["positions"]), -1, 3))
            x = x.flatten()
            results.append(res)
            images.append(x.copy())
            cv_vals.append(c)
        return results, images, cv_vals

    bwd_results, bwd_images, bwd_cv = _run_branch(-1, steps_back)
    fwd_results, fwd_images, fwd_cv = _run_branch(+1, steps_fwd)

    all_images  = list(reversed(bwd_images))  + [x0_medoid] + fwd_images
    all_cv      = list(reversed(bwd_cv))       + [chi0]      + fwd_cv
    all_results = list(reversed(bwd_results))  + [res0]      + fwd_results

    # ── Free energy (thermodynamic integration + Fixman correction) ───────
    chi_arr         = np.array(all_cv)
    mean_forces     = np.array([r["lambdas"].mean()              for r in all_results])
    mean_inv_sqrtZ  = np.array([np.mean(1.0 / np.sqrt(r["Zs"])) for r in all_results])

    # Trapezoid integration of mean force over chi
    F_rigid = np.zeros(len(chi_arr))
    for i in range(1, len(chi_arr)):
        dchi = chi_arr[i] - chi_arr[i - 1]
        F_rigid[i] = F_rigid[i - 1] + 0.5 * (mean_forces[i] + mean_forces[i - 1]) * dchi

    # Fixman correction: F_std = F_rigid − kBT·log(<1/√Z>)
    F_free = F_rigid - kBT * np.log(mean_inv_sqrtZ + 1e-300)
    F_free -= F_free.min()                              # shift minimum to zero

    if not store_positions:
        for r in all_results:
            del r["positions"]

    print(f"\nχ-MEP complete: {len(all_images)} images, "
          f"ξ ∈ [{all_cv[0]:.4f}, {all_cv[-1]:.4f}], "
          f"ΔF = {F_free.max():.2f} kJ/mol")
    return {
        "images":       all_images,                     # list of (n_atoms*3,) arrays
        "cv_values":    all_cv,                         # list of floats
        "results":      all_results,                    # list of per-levelset dicts
        "mean_forces":  mean_forces,                    # (n_images,)
        "F_rigid":      F_rigid,                        # (n_images,) PMF without Fixman
        "F_free":       F_free,                         # (n_images,) PMF + Fixman
        "sensitivity":  np.array([r["sensitivity"] for r in all_results]),  # (n_images, n_atoms)
    }
