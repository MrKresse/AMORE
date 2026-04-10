"""
OpenMM-based molecular dynamics simulator for ISOKANN.

This module is a Python port / adaptation of the Julia implementation in
ISOKANN.jl by axsk (https://github.com/axsk/ISOKANN.jl), specifically
  src/simulators/openmm.jl
  src/simulators/mopenmm.py
  src/models.jl

Positions are stored and returned as flat numpy arrays of shape
(n_atoms * 3,) in nanometres throughout.
"""

import copy
import math
import os

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    import openmm as mm
    from openmm import app, unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


# ---------------------------------------------------------------------------
# Default paths and force fields  (mirror openmm.jl constants)
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
)
DEFAULT_PDB             = os.path.join(_DATA_DIR, 'alanine-dipeptide-nowater.pdb')
FORCE_AMBER             = ['amber14-all.xml']
FORCE_AMBER_IMPLICIT    = ['amber14-all.xml', 'implicit/obc2.xml']
FORCE_AMBER_EXPLICIT    = ['amber14-all.xml', 'amber/tip3p_standard.xml']


# ---------------------------------------------------------------------------
# Low-level helpers  (mirror mopenmm.py from ISOKANN.jl)
# ---------------------------------------------------------------------------

def _get_positions(context):
    """Return flat (n_atoms*3,) float64 array of positions in nm."""
    state = context.getState(getPositions=True)
    return (state.getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
            .flatten()
            .astype(np.float64))


def _set_positions(context, x_flat, temp_kelvin):
    """Set positions from flat (n_atoms*3,) array and randomise velocities."""
    n_atoms = len(x_flat) // 3
    context.setPositions(x_flat.reshape(n_atoms, 3))
    context.setVelocitiesToTemperature(temp_kelvin * unit.kelvin)


def _new_context(context):
    """Clone a context (independent copy for parallel propagation)."""
    return mm.Context(
        context.getSystem(),
        copy.copy(context.getIntegrator()),
        context.getPlatform(),
    )


# ---------------------------------------------------------------------------
# OpenMMSimulator — the core class
# ---------------------------------------------------------------------------

class OpenMMSimulator:
    """
    Thin wrapper around an OpenMM Simulation for ISOKANN.

    Do not construct directly — use :func:`OpenMMSimulation` which provides
    the same interface as ``OpenMMSimulation`` in ISOKANN.jl.
    """

    def __init__(self, pysim, steps, temp, dt):
        self._sim   = pysim
        self._steps = steps
        self._temp  = temp
        self._dt    = dt
        self.n_atoms = sum(1 for _ in pysim.topology.atoms())
        self.dim     = self.n_atoms * 3

    @property
    def lagtime(self):
        """Lag time in ps (steps × dt)."""
        return self._steps * self._dt

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_positions(self):
        """Return flat (n_atoms*3,) float64 array in nm."""
        return _get_positions(self._sim.context)

    def set_positions(self, x):
        """Set positions from flat (n_atoms*3,) array (nm); randomise velocities."""
        _set_positions(self._sim.context,
                       np.asarray(x, dtype=np.float64),
                       self._temp)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def step(self, n_steps):
        """Advance the simulation by n_steps integration steps in-place."""
        self._sim.context.getIntegrator().step(n_steps)

    def trajectory(self, x0=None, T=None, save_every_steps=None):
        """
        Run a trajectory, optionally from x0.

        Parameters
        ----------
        x0 : np.ndarray of shape (n_atoms*3,) or None
            Starting positions in nm.  Uses current simulation state if None.
        T : float or None
            Total simulation time in ps.  Defaults to ``lagtime``.
        save_every_steps : int or None
            Save a frame every this many steps.  Defaults to ``self._steps``.

        Returns
        -------
        traj : np.ndarray of shape (n_frames, n_atoms*3)
        """
        if T is None:
            T = self.lagtime
        if save_every_steps is None:
            save_every_steps = self._steps

        total_steps = max(1, int(round(T / self._dt)))
        ctx = _new_context(self._sim.context)
        x_init = np.asarray(x0, dtype=np.float64) if x0 is not None else _get_positions(self._sim.context)
        _set_positions(ctx, x_init, self._temp)

        n_frames = total_steps // save_every_steps + 1
        traj = np.empty((n_frames, self.dim))
        traj[0] = _get_positions(ctx)
        sim_time_ps = save_every_steps * self._dt
        it = range(1, n_frames)
        if _HAS_TQDM:
            it = _tqdm(it, desc="trajectory", unit="frame",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]")
        for k in it:
            ctx.getIntegrator().step(save_every_steps)
            traj[k] = _get_positions(ctx)
            if _HAS_TQDM:
                it.set_postfix(sim_ps=f"{k * sim_time_ps:.1f}/{(n_frames-1)*sim_time_ps:.1f}")
        return traj

    def koopman_pairs(self, xs, steps=None):
        """
        Propagate each row of ``xs`` for ``steps`` integration steps.

        Parameters
        ----------
        xs : np.ndarray of shape (n_samples, n_atoms*3) or (n_atoms*3,)
            Starting positions in nm.  A single frame is used as-is (batch=1).
        steps : int or None
            Number of integration steps per sample.  Defaults to
            ``self._steps`` (i.e. one lagtime).

        Returns
        -------
        X0, Xtau : np.ndarray of shape (n_samples, n_atoms*3)
        """
        if steps is None:
            steps = self._steps

        xs = np.asarray(xs, dtype=np.float64)
        if xs.ndim == 1:
            xs = xs[None, :]

        n_samples = len(xs)
        X0   = xs.copy()
        Xtau = np.empty_like(X0)

        lagtime_ps = steps * self._dt
        it = range(n_samples)
        if _HAS_TQDM:
            it = _tqdm(it, desc=f"koopman pairs (τ={lagtime_ps:.3g} ps)", unit="pair")
        for i in it:
            ctx = _new_context(self._sim.context)
            _set_positions(ctx, xs[i], self._temp)
            ctx.getIntegrator().step(steps)
            Xtau[i] = _get_positions(ctx)

        return X0, Xtau

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"OpenMMSimulator(n_atoms={self.n_atoms}, "
                f"steps={self._steps}, lagtime={self.lagtime:.3g} ps, "
                f"temp={self._temp} K)")


# ---------------------------------------------------------------------------
# Lag-based Koopman pairs from an existing trajectory
# ---------------------------------------------------------------------------

def koopman_pairs_lag(traj, lag=1):
    """
    Extract Koopman pairs from a trajectory by pairing frames ``lag`` apart.

    Much cheaper than burst sampling: no new MD required.  Suitable when a
    long equilibrium trajectory is already available.

    Parameters
    ----------
    traj : np.ndarray of shape (n_frames, dim)
        Trajectory of flat position vectors.
    lag : int
        Frame lag (number of saved frames between X0 and X_tau).
        In physical time: ``lag × save_every_steps × dt`` ps.

    Returns
    -------
    X0, Xtau : np.ndarray of shape (n_frames - lag, dim)
    """
    traj = np.asarray(traj)
    return traj[:-lag].copy(), traj[lag:].copy()


# ---------------------------------------------------------------------------
# OpenMMSimulation — convenience constructor (mirrors ISOKANN.jl)
# ---------------------------------------------------------------------------

def OpenMMSimulation(pdb=None, py=None, steps=500,
                     forcefields=None,
                     temp=310.0, friction=1.0, dt=2e-3,
                     minimize=False,
                     platform=None,
                     n_threads=1,
                     constraints=None,
                     nonbonded_method='NoCutoff'):
    """
    Create an :class:`OpenMMSimulator`.  Mirrors ``OpenMMSimulation`` in ISOKANN.jl.

    Three calling modes
    -------------------
    ``OpenMMSimulation()``
        Default: alanine dipeptide, amber14-all.xml, 310 K.

    ``OpenMMSimulation(pdb="my.pdb", ...)``
        Build from a PDB file and keyword parameters.

    ``OpenMMSimulation(py="setup.py")``
        Execute a Python script that defines a variable named ``simulation``
        (an ``openmm.app.Simulation`` object) and wrap it.

    Parameters
    ----------
    pdb : str or None
        Path to PDB file.  Defaults to the bundled alanine-dipeptide PDB.
    py : str or None
        Path to a ``.py`` script that defines ``simulation``.
        When given, all other force-field parameters are ignored.
    steps : int
        Integration steps per Koopman lag (lagtime = steps × dt).
    forcefields : list of str or None
        Force-field XML files.  Defaults to ``['amber14-all.xml']``.
    temp : float
        Temperature in kelvin (default 310).
    friction : float
        Langevin friction in ps⁻¹.
    dt : float
        Integration time step in ps (default 2 fs).
    minimize : bool
        Run energy minimisation after setup.
    platform : str or None
        ``'CUDA'``, ``'OpenCL'``, ``'CPU'``, or ``'gpu'`` (alias for CUDA,
        matching Julia's ``mmthreads='gpu'`` convention).  If ``None`` (default),
        OpenMM automatically selects the fastest registered platform.
    n_threads : int
        CPU threads (only used when platform='CPU').
    constraints : str or None
        ``None``, ``'HBonds'``, ``'AllBonds'``, or ``'HAngles'``.
    nonbonded_method : str
        ``'NoCutoff'``, ``'CutoffNonPeriodic'``, ``'PME'``, or ``'auto'``.
    """
    if not HAS_OPENMM:
        raise ImportError("openmm is required: conda install -c conda-forge openmm")

    # --- dispatch on py= ---
    if py is not None:
        pysim = _load_from_script(py)
        _dt = pysim.context.getIntegrator().getStepSize().value_in_unit(unit.picoseconds)
        _temp = pysim.context.getIntegrator().getTemperature().value_in_unit(unit.kelvin)
        return OpenMMSimulator(pysim, steps, _temp, _dt)

    # --- default pdb ---
    if pdb is None:
        pdb = DEFAULT_PDB
    if forcefields is None:
        forcefields = FORCE_AMBER

    # --- build system ---
    pdb_obj    = app.PDBFile(pdb)
    forcefield = app.ForceField(*forcefields)
    modeller   = app.Modeller(pdb_obj.topology, pdb_obj.positions)

    nb_method = _parse_nonbonded(nonbonded_method, pdb_obj)
    system    = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=nb_method,
        removeCMMotion=False,
        constraints=_parse_constraints(constraints),
    )

    integrator = mm.LangevinMiddleIntegrator(
        temp     * unit.kelvin,
        friction / unit.picosecond,
        dt       * unit.picoseconds,
    )

    # --- platform selection (mirrors Julia: mmthreads='gpu' → CUDA, else CPU) ---
    if platform is None or platform == 'auto':
        # Let OpenMM pick the fastest registered platform automatically
        pysim = app.Simulation(modeller.topology, system, integrator)
    else:
        # 'gpu' is the Julia convention (mopenmm.py mmthreads='gpu')
        plat_name = 'CUDA' if platform.lower() == 'gpu' else platform
        plat_obj  = mm.Platform.getPlatformByName(plat_name)
        plat_kw   = {'Threads': str(n_threads)} if plat_name.upper() == 'CPU' else {}
        pysim     = app.Simulation(modeller.topology, system, integrator, plat_obj, plat_kw)
    pysim.context.setPositions(modeller.positions)
    pysim.context.setVelocitiesToTemperature(temp * unit.kelvin)

    if minimize:
        pysim.minimizeEnergy()

    return OpenMMSimulator(pysim, steps, temp, dt)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_from_script(py_path):
    """Execute a .py script and return its ``simulation`` variable."""
    ns = {}
    with open(py_path) as f:
        exec(compile(f.read(), py_path, 'exec'), ns)
    if 'simulation' not in ns:
        raise ValueError(
            f"Script {py_path!r} must define a variable named 'simulation'."
        )
    return ns['simulation']


def _parse_nonbonded(method, pdb_obj):
    mapping = {
        'NoCutoff':          app.NoCutoff,
        'CutoffNonPeriodic': app.CutoffNonPeriodic,
        'CutoffPeriodic':    app.CutoffPeriodic,
        'PME':               app.PME,
        'LJPME':             app.LJPME,
        'auto': (app.CutoffNonPeriodic
                 if pdb_obj.getTopology().getPeriodicBoxVectors() is None
                 else app.CutoffPeriodic),
    }
    if method not in mapping:
        raise ValueError(
            f"Unknown nonbonded_method '{method}'. Choose from: {', '.join(mapping)}"
        )
    return mapping[method]


def _parse_constraints(c):
    mapping = {
        None:       None,
        'None':     None,
        'HBonds':   app.HBonds,
        'AllBonds': app.AllBonds,
        'HAngles':  app.HAngles,
    }
    if c not in mapping:
        raise ValueError(f"Unknown constraints '{c}'.")
    return mapping[c]


# ---------------------------------------------------------------------------
# pairnet node widths  (mirrors ISOKANN.jl models.jl pairnet)
# ---------------------------------------------------------------------------

def pairnet_nodes(n_features, nout=1, layers=3):
    """
    Compute network layer widths matching ``pairnet`` in ISOKANN.jl.

    Uses a geometric progression from ``n_features`` down to
    ``round(n_features^(1/layers))``, then ``nout``.

    Parameters
    ----------
    n_features : int
        Input dimension (number of pairwise-distance features).
    nout : int
        Output dimension (default 1 for chi).
    layers : int
        Number of hidden + output layers (default 3, matching Julia default).

    Returns
    -------
    nodes : list of int
        Layer widths ``[n_features, h1, h2, ..., nout]``.

    Example
    -------
    >>> pairnet_nodes(231)
    [231, 37, 6, 1]
    """
    widths = [round(n_features ** (l / layers)) for l in range(layers, 0, -1)]
    return widths + [nout]


# ---------------------------------------------------------------------------
# Dihedral helpers for alanine dipeptide  (0-based atom indices)
# ---------------------------------------------------------------------------
# Atom order from alanine-dipeptide-nowater.pdb:
#  0  HH31  ACE     4  C    ACE  (phi C_prev)
#  1  CH3   ACE     6  N    ALA  (phi N  / psi N)
#  2  HH32  ACE     8  CA   ALA  (phi CA / psi CA)
#  3  HH33  ACE    14  C    ALA  (phi C' / psi C')
#                  16  N    NME  (psi N_next)

_PHI_IDX = (4, 6, 8, 14)    # C(ACE)–N(ALA)–CA(ALA)–C(ALA)
_PSI_IDX = (6, 8, 14, 16)   # N(ALA)–CA(ALA)–C(ALA)–N(NME)

"""
def _dihedral(p0, p1, p2, p3):
    b0 = p1 - p0          # standard IUPAC convention: vector along p0→p1 bond
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v  = b0 - np.dot(b0, b1) * b1
    w  = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))
"""

def _dihedral(p0, p1, p2, p3):
    b  = p2 - p1
    u  = np.cross(b, p1 - p0)   # = cross(b, -b0) 
    w  = np.cross(b, p2 - p3)   # = cross(b, -b2)
    cross_uw = np.cross(u, w)
    return np.arctan2(
        np.dot(cross_uw, b),
        np.dot(u, w) * np.linalg.norm(b)
    )

def phi(coords_flat):
    """
    Backbone φ dihedral for alanine dipeptide.

    Parameters
    ----------
    coords_flat : array of shape (n_atoms*3,) or (n_frames, n_atoms*3)

    Returns
    -------
    float or np.ndarray of shape (n_frames,)
    """
    coords_flat = np.asarray(coords_flat, dtype=np.float64)
    single = (coords_flat.ndim == 1)
    if single:
        coords_flat = coords_flat[None, :]
    xyz    = coords_flat.reshape(len(coords_flat), -1, 3)
    result = np.array([
        _dihedral(xyz[k, _PHI_IDX[0]], xyz[k, _PHI_IDX[1]],
                  xyz[k, _PHI_IDX[2]], xyz[k, _PHI_IDX[3]])
        for k in range(len(xyz))
    ])
    return float(result[0]) if single else result


def psi(coords_flat):
    """
    Backbone ψ dihedral for alanine dipeptide.

    Parameters
    ----------
    coords_flat : array of shape (n_atoms*3,) or (n_frames, n_atoms*3)

    Returns
    -------
    float or np.ndarray of shape (n_frames,)
    """
    coords_flat = np.asarray(coords_flat, dtype=np.float64)
    single = (coords_flat.ndim == 1)
    if single:
        coords_flat = coords_flat[None, :]
    xyz    = coords_flat.reshape(len(coords_flat), -1, 3)
    result = np.array([
        _dihedral(xyz[k, _PSI_IDX[0]], xyz[k, _PSI_IDX[1]],
                  xyz[k, _PSI_IDX[2]], xyz[k, _PSI_IDX[3]])
        for k in range(len(xyz))
    ])
    return float(result[0]) if single else result
