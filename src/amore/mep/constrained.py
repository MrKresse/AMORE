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

    def forward(self, positions, boxvectors=None):        # (n_atoms, 3) → ()
        # boxvectors: TorchForce calls forward(positions, boxvectors) whenever
        # setUsesPeriodicBoundaryConditions(True) was set on it -- unused here (accepted
        # only to match that calling arity) because _PairDistFeaturizer has no PBC notion
        # to begin with (used for the vacuum/no-cutoff case only).
        positions = positions.to(dtype=pt.float32)        # OpenMM may call with double
                                                           # positions (e.g. Reference/CPU
                                                           # platforms are double-precision
                                                           # internally); the network's own
                                                           # weights are float32, so cast
                                                           # once at the traced entry point
                                                           # rather than assume the caller's
                                                           # precision matches training.
        flat  = positions.reshape(1, -1)                  # (1, n_atoms*3)
        feats = self.feat(flat)                           # (1, n_pairs)
        chi   = self.nu(feats)                            # (1,) or ()
        return chi.reshape(())                            # scalar


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------

def export_cv_torchscript(nu, pairs, example_positions_nm, path, periodic=False):
    """
    Trace the chi pipeline and save it as a TorchScript (.pt) file
    suitable for use with openmmtorch.TorchForce.

    The exported module has the signature::

        forward(positions: Tensor[n_atoms, 3]) -> Tensor[()]                     periodic=False
        forward(positions: Tensor[n_atoms, 3], boxvectors: Tensor[3, 3]) -> Tensor[()]  periodic=True

    where positions are in nm (float32).  ``periodic`` MUST match whatever
    ``torch_force.setUsesPeriodicBoundaryConditions(...)`` is set to on the
    ``TorchForce`` this model is loaded into (see ``build_constrained_system`` /
    ``build_cv_gradient_system``) -- when that flag is True, OpenMM calls the traced
    module with an extra box-vectors argument, and a schema traced with only one
    argument raises a TorchScript arity error at runtime (``boxvectors`` itself is
    accepted-and-ignored here since ``_PairDistFeaturizer`` has no PBC handling; the
    argument only needs to exist to match the calling convention).

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
    periodic : bool
        Trace a 2-argument (positions, boxvectors) schema instead of the default
        1-argument one.  Default False (matches the original NoCutoff/vacuum usage).
    """
    device = next(nu.parameters()).device
    nu.eval()

    pipeline = _CVPipeline(nu, pairs).to(device)
    pipeline.eval()

    pos_t = pt.from_numpy(
        example_positions_nm.astype(np.float32)
    ).to(device)                                          # (n_atoms, 3)

    with pt.no_grad():
        if periodic:
            box_t = pt.zeros(3, 3, dtype=pt.float32, device=device)  # unused, see docstring
            traced = pt.jit.trace(pipeline, (pos_t, box_t))
        else:
            traced = pt.jit.trace(pipeline, pos_t)

    traced.save(path)
    return path


class _TracedCVModule(pt.nn.Module):
    """
    Generalized CV pipeline for TorchForce: an arbitrary (already-built) featurizer
    nn.Module composed with an arbitrary CV nn.Module (e.g. ``amore.mep.simplex.FaceCV``
    wrapping a full k-membership network).  Unlike ``_CVPipeline``, this does not assume
    a raw-pairwise-distance featurizer -- it takes the featurizer as a module directly,
    so it works with e.g. ``comfeat.PocketFeaturizer`` (side-chain COM gather/scatter +
    all-atom contact distances, PBC minimum-image), which the simple pairwise pipeline
    cannot represent.

    Input : positions (n_atoms, 3) float32, nm
    Output: scalar CV value
    """

    def __init__(self, featurizer_module, cv_module):
        super().__init__()
        self.featurizer = featurizer_module
        self.cv = cv_module

    def forward(self, positions, boxvectors=None):          # (n_atoms, 3) -> ()
        # boxvectors: see _CVPipeline.forward -- accepted-and-ignored to match
        # TorchForce's calling arity when setUsesPeriodicBoundaryConditions(True).
        # comfeat's featurizers (ResidueCOMPairFeaturizer / PocketFeaturizer / ...) apply
        # their own minimum-image convention against a FIXED box captured as a registered
        # buffer at construction time (the training system's box, not a live one) -- they
        # were built that way for `data.build_features*`'s training pipeline, which has no
        # notion of a changing box either, so this is not a regression, just carried over.
        positions = positions.to(dtype=pt.float32)          # see _CVPipeline.forward
        flat = positions.reshape(1, -1)                    # (1, n_atoms*3)
        feats = self.featurizer(flat)                       # (1, n_feat)
        val = self.cv(feats)                                 # (1, 1) or (1,)
        return val.reshape(())


def export_cv_torchscript_module(featurizer_module, cv_module, example_positions_nm, path,
                                 periodic=False):
    """
    Trace an arbitrary (featurizer_module, cv_module) pair -- e.g.
    ``comfeat.PocketFeaturizer`` + ``amore.mep.simplex.FaceCV(model, i)`` -- and save it
    as a TorchScript (.pt) file for ``openmmtorch.TorchForce``.  Generalizes
    ``export_cv_torchscript`` (which hardcodes a raw-pairwise-distance featurizer) to any
    already-built featurizer nn.Module, so it works with real trained-model featurizers
    (residue-COM / pocket / ligand pipelines), not only the toy pairwise-distance case.

    The exported module has the same signature as ``export_cv_torchscript``'s output::

        forward(positions: Tensor[n_atoms, 3]) -> Tensor[()]                     periodic=False
        forward(positions: Tensor[n_atoms, 3], boxvectors: Tensor[3, 3]) -> Tensor[()]  periodic=True

    Parameters
    ----------
    featurizer_module : torch.nn.Module
        forward(flat_coords: (1, n_atoms*3)) -> (1, n_feat).  E.g. the ``.module``
        attribute of a ``comfeat.make_torch_featurizer_pocket(...)`` closure.
    cv_module : torch.nn.Module
        forward(feats: (1, n_feat)) -> (1, 1) or (1,) scalar CV.  E.g.
        ``amore.mep.simplex.FaceCV(model, i)`` or ``EdgeCV(model, i, j)``.
    example_positions_nm : np.ndarray, shape (n_atoms, 3)
        A representative configuration in nanometres.  Used only for tracing.
    path : str
        Output file path for the .pt model.
    periodic : bool
        Trace a 2-argument (positions, boxvectors) schema instead of the default
        1-argument one -- MUST match ``torch_force.setUsesPeriodicBoundaryConditions``
        on the destination TorchForce (see ``export_cv_torchscript``'s docstring for why).
        Any PBC-solvated system (e.g. 2cm2's ``system_flex.xml``, built with
        ``nonbondedMethod=PME``) needs ``periodic=True``.

    Returns
    -------
    path : str
    """
    device = next(cv_module.parameters()).device
    featurizer_module.eval()
    cv_module.eval()

    pipeline = _TracedCVModule(featurizer_module, cv_module).to(device)
    pipeline.eval()

    pos_t = pt.from_numpy(
        np.asarray(example_positions_nm, dtype=np.float32)
    ).to(device)                                            # (n_atoms, 3)

    with pt.no_grad():
        if periodic:
            box_t = pt.zeros(3, 3, dtype=pt.float32, device=device)  # unused, see docstring
            traced = pt.jit.trace(pipeline, (pos_t, box_t))
        else:
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


def build_cv_gradient_system(system, cv_model_path, grad_force_group=None):
    """
    Add a TorchForce-based CV to a copy of *system* that reports ``-grad(xi)`` as its
    own OpenMM force, in its own dedicated force group.

    ``CustomCVForce(expression)`` symbolically differentiates *expression* w.r.t. its
    collective variables and applies the chain rule through each CV's own analytic
    gradient (TorchForce supplies xi's gradient via its GPU-resident autograd backward).
    With ``expression = "xi"`` (coefficient 1, no restraint), that force IS exactly
    ``-grad(xi)`` -- the same trick ``build_constrained_system`` already relies on for
    its harmonic restraint's force, just with the restraint stripped down to unit
    coefficient.  Putting the CV force in its own force group means it is never summed
    into the physical force-field forces (which live in group 0 by default): a query
    with ``groups={grad_force_group}`` returns ONLY ``-grad(xi)``, and a separate query
    with ``groups={0}`` (or whichever group the physical forces use) returns only the
    physical force -- both computed by OpenMM's own force evaluation in the SAME
    context, with no separate Python/PyTorch autograd call and no extra host<->device
    round trip beyond the two ``getState`` calls themselves.

    This is the building block for ``sample_levelset_projected_torchforce``, which
    mirrors ``sample_levelset_projected``'s projected-Langevin/Newton-retraction
    algorithm exactly but sources chi's value/gradient this way instead of via
    ``_chi_and_grad``/``_chi_val`` (a Python-orchestrated forward+backward through the
    featurizer with explicit ``.item()``/``.cpu()`` sync points every step).

    Parameters
    ----------
    system : openmm.System
        Existing physical system.  A deep copy is made so the original is not modified.
    cv_model_path : str
        Path to the TorchScript CV model (.pt file, from ``export_cv_torchscript`` or
        ``export_cv_torchscript_module``).
    grad_force_group : int or None
        Force group for the CV-gradient force.  If None, the first group index (0-31)
        not already used by any force in *system* is picked automatically.

    Returns
    -------
    new_system : openmm.System
    cv_force_index : int
        Index of the CustomCVForce inside new_system (for ``getCollectiveVariableValues``).
    grad_force_group : int
        The force group actually used (echoed back when auto-picked).
    """
    if not HAS_OPENMM:
        raise ImportError("openmm is required")
    if not HAS_OPENMMTORCH:
        raise ImportError("openmmtorch (openmm-torch plugin) is required")

    new_system = copy.deepcopy(system)

    if grad_force_group is None:
        used = {f.getForceGroup() for f in new_system.getForces()}
        grad_force_group = next(g for g in range(32) if g not in used)

    torch_force = TorchForce(cv_model_path)
    torch_force.setUsesPeriodicBoundaryConditions(
        new_system.usesPeriodicBoundaryConditions()
    )

    cv_force = mm.CustomCVForce("xi")
    cv_force.addCollectiveVariable("xi", torch_force)
    cv_force.setForceGroup(grad_force_group)

    cv_force_index = new_system.addForce(cv_force)
    return new_system, cv_force_index, grad_force_group


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

def _pairwise_kabsch_rmsd(pts):
    """All-pairs Kabsch-optimal RMSD, vectorized (one batched SVD, no explicit rotation).

    pts : (n_frames, n_points, 3) -> (n_frames, n_frames) RMSD matrix.

    Aligning every frame onto one arbitrary reference and then taking plain Euclidean
    distance is NOT the same as the true pairwise-optimal RMSD between two OTHER frames
    (each is only optimally aligned to the reference, not to each other), and biases the
    result toward whichever frame happens to be picked as reference. The correct pairwise
    distance re-solves Kabsch for every pair. That has a closed form via the singular
    values of each pair's cross-covariance matrix (no need to ever build the actual
    rotation matrix), so all n^2 pairs batch into a single SVD call.
    """
    n, n_pts, _ = pts.shape
    c  = pts - pts.mean(axis=1, keepdims=True)                    # (n, n_pts, 3), centered
    ss = (c ** 2).sum(axis=(1, 2))                                 # (n,)
    # H[i,j,k,l] = sum_a c[i,a,k]*c[j,a,l] -- 9 proper BLAS matmuls (one per (k,l) pair)
    # instead of a naive np.einsum('iak,jal->ijkl', ...), which for this index pattern
    # numpy does NOT map onto a BLAS call and falls back to a slow generic evaluation
    # (confirmed ~14x slower at n=200: 27.7s vs 2.0s -- the dominant cost of an entire
    # MFEP level at large steps_per_levelset before this fix).
    H = np.empty((n, n, 3, 3))
    for k in range(3):
        for l in range(3):
            H[:, :, k, l] = c[:, :, k] @ c[:, :, l].T
    U, S, Vt = np.linalg.svd(H)                                    # batched over leading (n, n)
    d = np.sign(np.linalg.det(np.matmul(np.swapaxes(Vt, -1, -2), np.swapaxes(U, -1, -2))))
    signed_sum = S[..., 0] + S[..., 1] + d * S[..., 2]             # (n, n)
    E0 = ss[:, None] + ss[None, :]
    rmsd2 = np.clip(E0 - 2.0 * signed_sum, 0.0, None) / n_pts
    return np.sqrt(rmsd2)


def _featurizer_atom_idx(featurizer):
    """Union of atom indices the featurizer actually reads (e.g. PocketFeaturizer's
    side-chain COM atoms + all-atom contact atoms), or None if unavailable."""
    mod = getattr(featurizer, "module", None)
    if mod is None:
        return None
    parts = [getattr(mod, name, None)
             for name in ("sc_atom_idx", "prot_contact_idx", "lig_contact_idx")]
    parts = [p.detach().cpu().numpy() for p in parts if p is not None]
    if not parts:
        return None
    return np.unique(np.concatenate(parts))


def select_medoid(positions, featurizer=None):
    """
    Select the medoid (minimum mean pairwise distance) from sampled frames.

    Parameters
    ----------
    positions : np.ndarray, shape (n_frames, n_particles, 3)
    featurizer : callable or None
        If given (and it exposes the atom-index buffers used by e.g. PocketFeaturizer),
        distance is the all-pairs Kabsch-optimal RMSD (see `_pairwise_kabsch_rmsd`) on the
        SAME atoms that feed the features -- not the computed feature values themselves (a
        pairwise-distance feature vector mixes many length scales and is dominated by
        whichever few distances happen to have the largest raw magnitude/variance, not
        necessarily the functionally interesting ones), and not raw full-system Cartesian
        coordinates either (dominated by unaligned whole-complex rigid-body motion and, for
        an explicit-solvent system, tens of thousands of uncorrelated solvent atoms that
        vastly outnumber the ~hundreds of protein/ligand atoms the network actually
        attends to). Falls back to raw full-system coordinates (no alignment) if
        `featurizer` is None or exposes no atom-index buffers.

    Returns
    -------
    medoid_idx : int
    medoid_pos : np.ndarray, shape (n_particles, 3)
    """
    n_frames = positions.shape[0]
    if n_frames == 1:
        return 0, positions[0]

    atom_idx = _featurizer_atom_idx(featurizer) if featurizer is not None else None
    if atom_idx is not None:
        dist_matrix = _pairwise_kabsch_rmsd(positions[:, atom_idx, :])
        mean_dists  = dist_matrix.mean(axis=1)
        idx         = int(np.argmin(mean_dists))
        return idx, positions[idx]

    rep = positions.reshape(n_frames, -1)
    dist_matrix = squareform(pdist(rep))
    mean_dists  = dist_matrix.mean(axis=1)
    idx         = int(np.argmin(mean_dists))
    return idx, positions[idx]


# ---------------------------------------------------------------------------
# Euler step to next level set
# ---------------------------------------------------------------------------

def _capped_step(step, max_norm):
    """Rescale `step` down to `max_norm` if it exceeds it; direction unchanged."""
    n = np.linalg.norm(step)
    if n > max_norm:
        return step * (max_norm / n)
    return step


def euler_step_to_levelset(nu, featurizer, positions_nm, step_size,
                            max_newton_iters=10, tol=1e-3, max_step_nm=0.15):
    """
    From medoid at ξ = c, take an Euler step along ±∇χ to reach
    levelset ξ ≈ c + step_size.

    positions_nm : np.ndarray (n_atoms, 3)  –OR–  (n_atoms*3,)
        Particle coordinates in **nanometres**.
    step_size : float
        Signed CV increment.  Positive = toward state B (+∇χ).
    max_step_nm : float
        Both the raw Euler step and each Newton correction are Lagrange-multiplier-style
        hard-constraint steps (divide by ||grad chi||^2) -- if chi's gradient transiently
        collapses at some intermediate iterate (a near-critical-point/plateau of the
        trained network, not physically meaningful), the resulting step can be many nm
        even though the chi residual being corrected is tiny.  One such runaway iteration
        is enough to shove atoms into a genuine steric clash that later reads back as an
        astronomical OpenMM force -- confirmed on 2cm2 pose2's pocket-model MFEP work
        (g_norm2 collapsed to 6.7e-6 at Newton iteration 3, producing a 1.07 nm step vs.
        0.008-0.045 nm for every neighboring iteration).  Capping each step's magnitude
        bounds the damage without changing behavior in the well-conditioned case (every
        healthy iteration observed this session stayed under 0.05 nm).

    Returns
    -------
    new_pos : np.ndarray, shape (n_atoms, 3) nm
    new_c   : float  (actual CV value after Newton correction)
    """
    flat_nm   = np.asarray(positions_nm, dtype=np.float64).flatten()
    xi_val, g = _chi_and_grad(nu, featurizer, flat_nm)
    target_c  = xi_val + step_size

    g_norm2 = np.dot(g, g) + 1e-30
    x = flat_nm + _capped_step((step_size / g_norm2) * g, max_step_nm)   # Euler step (flat nm)

    # Newton correction to land on target_c
    for _ in range(max_newton_iters):
        xi_new, g_new = _chi_and_grad(nu, featurizer, x)
        residual = xi_new - target_c
        if abs(residual) < tol:
            break
        g_norm2_new = np.dot(g_new, g_new) + 1e-30
        x = x - _capped_step((residual / g_norm2_new) * g_new, max_step_nm)

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
                              time_breakdown=False, max_retract_nm=0.15):
    """
    Projected Langevin on the chi = chi_level surface.

    At every step:
      1. OpenMM forces  (GPU)
      2. Project F orthogonal to ∇χ; record mean-force lambda = −F·∇χ/‖∇χ‖²
      3. Langevin velocity/position update
      4. Newton retraction onto chi_level
      5. Record Fixman Z = Σ_i (∇χ_i)²/m_i  and per-atom sensitivity

    max_retract_nm : float
        Cap on the per-step Newton retraction magnitude -- same rationale as
        `euler_step_to_levelset`'s `max_step_nm`: retraction divides by ‖∇χ‖², which can
        transiently collapse mid-trajectory and otherwise produce a many-nm single-step
        jump (observed directly: 43-95 nm retraction steps once the level-set had already
        been knocked into a bad clash).  Bounding it stops runaway feedback within a
        level-set's own relaxation window; healthy retractions stay well under this.

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
        db_proj = np.dot(db, dchi) / dchi_sq
        db -= db_proj * dchi                     # project noise onto the tangent space too --
                                                  # otherwise every step's raw isotropic noise
                                                  # has a generic off-manifold component along
                                                  # dchi that the single retraction below has to
                                                  # absorb in full, not just the curvature drift
        v += (F / m - gamma * v) * dt + np.sqrt(2.0 * gamma * kBT * dt / m) * db
        x += v * dt

        # Newton retraction: reuse pre-step dchi for the correction direction
        # (saves one full backward pass per step); only the chi value is needed
        # to compute the residual.
        t0 = time.perf_counter()
        chi_new = _chi_val(nu, featurizer, x)
        x -= _capped_step(((chi_new - chi_level) / dchi_sq) * dchi, max_retract_nm)
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
    n = steps + burnin
    timing = {"openmm": t_openmm / n, "chi_grad": t_chi_grad / n, "retract": t_retract / n}
    total = t_openmm + t_chi_grad + t_retract + 1e-12
    print(f"  Levelset ξ={chi_level:.4f}: final ξ={chi_final:.4f}, "
          f"mean λ={lambdas.mean():.3f} kJ/mol, mean Z={Zs.mean():.4g}")
    if time_breakdown:
        print(f"    time/step: OpenMM {timing['openmm']*1e3:.2f} ms  "
              f"χ-grad {timing['chi_grad']*1e3:.2f} ms  "
              f"retract {timing['retract']*1e3:.2f} ms  "
              f"| {t_openmm/total*100:.0f}% / {t_chi_grad/total*100:.0f}% / {t_retract/total*100:.0f}%")
    return {
        "positions":   positions,
        "lambdas":     lambdas,
        "Zs":          Zs,
        "sensitivity": sensitivity,
        "timing":      timing,   # seconds/step: {"openmm", "chi_grad", "retract"}
    }


def sample_levelset_projected_torchforce(sim, cv_force_index, grad_force_group, x0, chi_level,
                                         steps, burnin=0, phys_force_groups=(0,),
                                         time_breakdown=False, max_retract_nm=0.15):
    """
    TorchForce-fused counterpart to ``sample_levelset_projected`` -- IDENTICAL algorithm
    (projected Langevin + Newton retraction on the chi = chi_level surface), but chi's
    value/gradient are sourced from an OpenMM CustomCVForce+TorchForce evaluation (see
    ``build_cv_gradient_system``) instead of a separate Python/PyTorch autograd call
    through ``nu``/``featurizer``.  This targets exactly the two most expensive pieces
    ``sample_levelset_projected``'s own ``time_breakdown`` instrumentation identified
    (the chi-network gradient and the Newton-retraction chi evaluation): both become
    OpenMM ``getState``/``getCollectiveVariableValues`` calls on the SAME context as the
    physical force query, fused into OpenMM's own (GPU-resident) force evaluation with
    no separate Python-dispatched forward/backward through the featurizer and no
    explicit ``.item()``/``.cpu()`` sync points.

    ``sim.system`` MUST already be one built by ``build_cv_gradient_system`` (or
    equivalent): a physical force field in ``phys_force_groups`` plus a
    ``CustomCVForce("xi")`` wrapping a traced chi model in its own ``grad_force_group``,
    so a force query restricted to that group returns exactly ``-grad(xi)``.

    Parameters
    ----------
    sim : OpenMMSimulator
        Its ``._sim.system`` must be the CV-gradient system (see above); ``._sim.context``
        is stepped in place, same as ``sample_levelset_projected``.
    cv_force_index : int
        Index of the CustomCVForce inside ``sim._sim.system`` (from
        ``build_cv_gradient_system``), used for ``getCollectiveVariableValues``.
    grad_force_group : int
        Force group the CV-gradient force was assigned to (from
        ``build_cv_gradient_system``).
    x0 : np.ndarray (n_atoms*3,)
    chi_level : float
    steps : int
    burnin : int
        Steps to run before recording begins (discarded; not counted in output).
    phys_force_groups : tuple of int
        Force group(s) holding the physical force field (default ``(0,)``, matching
        ``ForceField.createSystem``'s default group assignment).
    max_retract_nm : float
        Same rationale as ``sample_levelset_projected``'s own cap.

    Returns
    -------
    dict, same keys/shapes as ``sample_levelset_projected``.
    """
    ctx   = sim._sim.context
    integ = ctx.getIntegrator()
    dt    = sim._dt
    temp  = sim._temp
    gamma = integ.getFriction().value_in_unit(unit.picosecond**-1)
    kBT   = 0.008314463 * temp                          # kJ/mol

    system = sim._sim.system
    cv_force = system.getForce(cv_force_index)
    phys_groups = set(phys_force_groups)

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

    def _chi_val_now():
        return float(cv_force.getCollectiveVariableValues(ctx)[0])

    def _step(x, v):
        nonlocal t_openmm, t_chi_grad, t_retract

        t0 = time.perf_counter()
        ctx.setPositions(x.reshape(n_particles, 3))
        F = (ctx.getState(getForces=True, groups=phys_groups)
             .getForces(asNumpy=True)
             .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
             .flatten().astype(np.float64))
        t_openmm += time.perf_counter() - t0

        t0 = time.perf_counter()
        neg_grad = (ctx.getState(getForces=True, groups={grad_force_group})
                    .getForces(asNumpy=True)
                    .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
                    .flatten().astype(np.float64))
        dchi = -neg_grad
        dchi_sq = np.dot(dchi, dchi) + 1e-30
        F_proj  = np.dot(F, dchi) / dchi_sq
        F      -= F_proj * dchi
        t_chi_grad += time.perf_counter() - t0

        db = np.random.randn(len(x))
        db_proj = np.dot(db, dchi) / dchi_sq
        db -= db_proj * dchi                     # project noise onto the tangent space too,
                                                  # same rationale as sample_levelset_projected
        v += (F / m - gamma * v) * dt + np.sqrt(2.0 * gamma * kBT * dt / m) * db
        x += v * dt

        # Newton retraction: reuse pre-step dchi for the correction direction (same as
        # sample_levelset_projected); only the chi value at the trial x is needed.
        t0 = time.perf_counter()
        ctx.setPositions(x.reshape(n_particles, 3))
        chi_new = _chi_val_now()
        x -= _capped_step(((chi_new - chi_level) / dchi_sq) * dchi, max_retract_nm)
        t_retract += time.perf_counter() - t0

        return x, v, F_proj, dchi

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

    ctx.setPositions(x.reshape(n_particles, 3))
    chi_final = _chi_val_now()
    n = steps + burnin
    timing = {"openmm": t_openmm / n, "chi_grad": t_chi_grad / n, "retract": t_retract / n}
    total = t_openmm + t_chi_grad + t_retract + 1e-12
    print(f"  Levelset ξ={chi_level:.4f} [torchforce]: final ξ={chi_final:.4f}, "
          f"mean λ={lambdas.mean():.3f} kJ/mol, mean Z={Zs.mean():.4g}")
    if time_breakdown:
        print(f"    time/step: OpenMM {timing['openmm']*1e3:.2f} ms  "
              f"χ-grad {timing['chi_grad']*1e3:.2f} ms  "
              f"retract {timing['retract']*1e3:.2f} ms  "
              f"| {t_openmm/total*100:.0f}% / {t_chi_grad/total*100:.0f}% / {t_retract/total*100:.0f}%")
    return {
        "positions":   positions,
        "lambdas":     lambdas,
        "Zs":          Zs,
        "sensitivity": sensitivity,
        "timing":      timing,
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
                            store_positions=False, time_breakdown=False, mean_force_tol=1e5,
                            cv_step=None, steps_back=None, steps_fwd=None, minimize_seeds=True,
                            minimize_max_iterations=0, minimize_tolerance=10.0,
                            seed_burnin=None):
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
    mean_force_tol : float or None
        If a level's mean force magnitude (|lambdas.mean()|, kJ/mol) exceeds this, the
        branch is truncated there instead of continuing.  This is a direct, symptom-level
        safeguard -- react to the actual observed energy/force rather than trying to
        predict trouble upstream (an earlier ||grad||^2-based predictive safeguard was
        tried and removed: it didn't reliably catch every bad case even with a fine
        step size, e.g. one seed's forward branch still reached F_free ~2e15 kJ/mol at
        steps=400 on 2cm2 pose2's ligand-inclusive k=3 model, while two sibling seeds
        from the same state completed cleanly with the same settings).  A sensible
        `steps`/`steps_per_levelset` choice should make this rare, not the primary
        defense -- it's a backstop for the rare case, not a substitute for good
        hyperparameters.  Set to None to disable.
    store_positions : bool
        If False (default), strip the ``positions`` array from each per-levelset
        result dict after use.  Set True only when you need the full ensemble —
        storing all frames for 50+ level sets can exhaust RAM.
    minimize_seeds : bool
        Locally energy-minimize (see ``_minimize_seed``) x0 ONLY, before the initial
        level -- not every branch's Euler-stepped seed.  Default True.  x0 is
        typically a raw real trajectory frame from a differently-constrained
        production system (rigid water + HBonds vs. this work's fully-flexible
        bonds); sampled cold it carries a slow-decaying strain transient (observed
        directly: mean projected force starting near -1e6 kJ/mol, still swinging in
        the 1e5-1e6 range after 50 fs). Every subsequent branch seed, by contrast, is
        already a THERMALLY sampled medoid from the previous level's own Langevin
        dynamics -- re-minimizing it at every level would strip that thermal motion
        (most consequentially the solvent's) and repeatedly reset it to an
        unrepresentative zero-temperature local minimum instead. Those seeds should
        only be re-equilibrated (real Langevin dynamics, not minimization) before
        each new level's recording begins -- pass ``burnin`` > 0 for that; it already
        applies uniformly to every level's ``sample_levelset_projected`` call,
        including the minimized initial one (letting the solvent re-thermalize around
        the minimized geometry before anything is recorded).
    minimize_max_iterations, minimize_tolerance : passed to ``_minimize_seed``.

    Returns
    -------
    dict with keys
      "images"    : list of np.ndarray (n_atoms*3,) — medoid configs, A→B order
      "cv_values" : list of float
    """
    x0 = np.asarray(x0, dtype=np.float64).flatten()
    if minimize_seeds:
        x0 = _minimize_seed(sim, x0, max_iterations=minimize_max_iterations,
                            tolerance=minimize_tolerance)
    chi0 = _chi_val(nu, featurizer, x0)
    # cv_step/steps_back/steps_fwd default to the original [0,1]-bounded convention (chi0 as
    # a fraction, splitting `steps` proportionally); pass them explicitly for an unbounded CV
    # (e.g. LogitEdgeCV/LogitFaceCV logits, which aren't in [0,1] so "chi0 as a fraction of
    # steps" is meaningless -- there's no natural total range to split proportionally).
    if cv_step is None:
        cv_step = 1.0 / steps
    if steps_back is None:
        steps_back = max(int(steps * chi0), 1)
    if steps_fwd is None:
        steps_fwd = max(int(steps * (1.0 - chi0)), 1)
    print(f"Initial ξ = {chi0:.4f}  ({steps_back} backward + {steps_fwd} forward images)")

    kBT = 0.008314463 * sim._temp

    # Sample and replace x0 with the medoid of its own level set
    print(f"  Sampling initial level set ξ={chi0:.4f}")
    res0 = sample_levelset_projected(sim, nu, featurizer, x0, chi0, steps_per_levelset,
                                              burnin=burnin, time_breakdown=time_breakdown)
    _, x0_medoid = select_medoid(res0["positions"].reshape(len(res0["positions"]), -1, 3),
                                 featurizer=featurizer)
    x0_medoid = x0_medoid.flatten()

    all_results = [res0]   # ordered A→B; filled in below then sorted

    def _run_branch(direction, n_steps):
        results, images, cv_vals = [], [], []
        x = x0_medoid.copy()
        label = "Fwd" if direction > 0 else "Bwd"
        for i in range(n_steps):
            x, c = euler_step_to_levelset(nu, featurizer, x, direction * cv_step)
            res = sample_levelset_projected(sim, nu, featurizer, x, c, steps_per_levelset,
                                           burnin=burnin, time_breakdown=time_breakdown)
            mean_force = float(res["lambdas"].mean())
            if mean_force_tol is not None and abs(mean_force) > mean_force_tol:
                print(f"  {label} {i+1}/{n_steps}: |mean force|={abs(mean_force):.3g} kJ/mol "
                      f"> {mean_force_tol:.3g} -- truncating branch at {len(images)} "
                      f"image(s) instead of continuing")
                break
            print(f"  {label} {i+1}/{n_steps}  ξ → {c:.4f}")
            _, x = select_medoid(res["positions"].reshape(len(res["positions"]), -1, 3),
                                 featurizer=featurizer)
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


# ---------------------------------------------------------------------------
# Decoupled ("string") χ-MEP: independent per-level sampling, no chaining
# ---------------------------------------------------------------------------

def _minimize_seed(sim, x_flat, max_iterations=0, tolerance=10.0):
    """Local energy minimization of a raw seed configuration under sim's own force field.

    Real trajectory frames were produced under a DIFFERENT system (rigid water +
    HBonds-constrained protein, per the production run) than the fully-flexible-bond
    system used for MFEP work here -- dropped in cold, they carry real bond/angle strain
    from that mismatch (observed directly: a raw seed's mean projected force started at
    ~-1.2e6 kJ/mol and was still swinging in the 1e5-1e6 range after 50 fs of Langevin,
    a slow-decaying transient, not evidence of a genuine chi-landscape problem at that
    level). A short minimization removes this non-dynamically and is far cheaper than
    diluting it with many more (expensive) relaxation steps.

    Defaults are OpenMM's own (tolerance=10 kJ/mol/nm, max_iterations=0 i.e. run to
    convergence rather than an arbitrary cap) -- empirically confirmed sufficient and
    cheap here: the worst observed seed (raw RMS force 683 kJ/mol/nm, max|F| 4894
    kJ/mol/nm) converged to RMS force 5.6 / max|F| 121 kJ/mol/nm in 372 L-BFGS
    iterations, 10.7s wall time. A fixed low iteration cap (200, this function's
    original default) silently under-converges worse-than-typical seeds instead of
    erroring -- max_iterations=0 is correct, not a cap sized by guesswork.

    tolerance : float
        OpenMM's RMS-force convergence criterion (kJ/mol/nm).

    Returns
    -------
    new_x_flat : np.ndarray (n_atoms*3,) nm
    """
    ctx = sim._sim.context
    n_particles = sim._sim.system.getNumParticles()
    ctx.setPositions(x_flat.reshape(n_particles, 3))
    mm.LocalEnergyMinimizer.minimize(ctx, tolerance=tolerance, maxIterations=max_iterations)
    pos = (ctx.getState(getPositions=True).getPositions(asNumpy=True)
           .value_in_unit(unit.nanometers))
    return pos.flatten().astype(np.float64)


def build_chi_string_independent(sim, nu, featurizer, seeds, cv_levels, steps_per_levelset,
                                 burnin=0, store_positions=False, time_breakdown=False,
                                 minimize_seeds=True, minimize_max_iterations=0,
                                 minimize_tolerance=10.0):
    """
    Decoupled per-level chi sampling -- the "one-shot string" alternative to
    ``build_chi_mep_projected``'s sequential chain.

    ``build_chi_mep_projected`` seeds level i from the single relaxed medoid of level
    i-1: any pathology at one level (a bad relaxation, an unlucky retraction) becomes
    the ONLY seed for every level downstream. Here every level is instead seeded
    independently from its own configuration (ideally a real trajectory frame already
    close to the target level) and retracted onto it with a single Newton correction --
    no history dependence between levels.

    This is exactly one "evolve" pass of the finite-temperature string method (E &
    Vanden-Eijnden), with the simplification that no arc-length reparametrization step
    is needed: under the standing assumption that chi's level sets approximate the
    committor's isocommittor surfaces (the same assumption ``build_chi_mep_projected``
    already makes), the CV value itself is already the correct arc-length
    parametrization -- images are sorted by their own (retracted) CV value below, so
    redistribution by aligned RMSD would be redundant.

    Parameters
    ----------
    sim : OpenMMSimulator
    nu, featurizer : chi network / collective variable
    seeds : sequence of np.ndarray (n_atoms*3,) or (n_atoms, 3)
        One seed configuration per level (nm), ideally a real trajectory frame already
        near that level's target CV value.
    cv_levels : sequence of float
        Target CV values, one per seed. Need not be sorted -- output is sorted by the
        actual retracted CV value before integration.
    steps_per_levelset : int
        Langevin steps at each level set (same meaning as in
        ``build_chi_mep_projected``).
    burnin, store_positions, time_breakdown : as in ``build_chi_mep_projected``.
    minimize_seeds : bool
        Locally energy-minimize each seed under ``sim``'s own force field before
        retraction/sampling (see ``_minimize_seed``). Default True -- seeds are
        typically raw real trajectory frames from a differently-constrained production
        system, and sampling them cold carries a slow-decaying strain transient that
        contaminates the mean-force estimate. Set False to skip (e.g. if seeds are
        already relaxed under this force field).
    minimize_max_iterations, minimize_tolerance : passed to ``_minimize_seed``.

    Returns
    -------
    dict, same keys as ``build_chi_mep_projected``'s output.
    """
    seeds = [np.asarray(s, dtype=np.float64).flatten() for s in seeds]
    kBT = 0.008314463 * sim._temp

    all_results, all_images, all_cv = [], [], []
    n = len(seeds)
    for i, (seed, target_c) in enumerate(zip(seeds, cv_levels)):
        if minimize_seeds:
            seed = _minimize_seed(sim, seed, max_iterations=minimize_max_iterations,
                                  tolerance=minimize_tolerance)
        seed_c = _chi_val(nu, featurizer, seed)
        x, c_actual = euler_step_to_levelset(nu, featurizer, seed, target_c - seed_c)
        print(f"Level {i+1}/{n}  seed ξ={seed_c:.4f} -> retracted ξ={c_actual:.4f} "
              f"(target {target_c:.4f})")
        res = sample_levelset_projected(sim, nu, featurizer, x, c_actual, steps_per_levelset,
                                        burnin=burnin, time_breakdown=time_breakdown)
        _, medoid = select_medoid(res["positions"].reshape(len(res["positions"]), -1, 3),
                                  featurizer=featurizer)
        all_results.append(res)
        all_images.append(medoid.flatten())
        all_cv.append(c_actual)

    order = np.argsort(all_cv)
    all_cv      = [all_cv[k] for k in order]
    all_images  = [all_images[k] for k in order]
    all_results = [all_results[k] for k in order]

    chi_arr        = np.array(all_cv)
    mean_forces    = np.array([r["lambdas"].mean()             for r in all_results])
    mean_inv_sqrtZ = np.array([np.mean(1.0 / np.sqrt(r["Zs"])) for r in all_results])

    F_rigid = np.zeros(len(chi_arr))
    for i in range(1, len(chi_arr)):
        dchi = chi_arr[i] - chi_arr[i - 1]
        F_rigid[i] = F_rigid[i - 1] + 0.5 * (mean_forces[i] + mean_forces[i - 1]) * dchi
    F_free = F_rigid - kBT * np.log(mean_inv_sqrtZ + 1e-300)
    F_free -= F_free.min()

    if not store_positions:
        for r in all_results:
            del r["positions"]

    print(f"\nχ-string complete: {len(all_images)} images, "
          f"ξ ∈ [{chi_arr[0]:.4f}, {chi_arr[-1]:.4f}], ΔF = {F_free.max():.2f} kJ/mol")
    return {
        "images":      all_images,
        "cv_values":   list(chi_arr),
        "results":     all_results,
        "mean_forces": mean_forces,
        "F_rigid":     F_rigid,
        "F_free":      F_free,
        "sensitivity": np.array([r["sensitivity"] for r in all_results]),
    }
