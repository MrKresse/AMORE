import os
import sys
import random

import numpy as np
import torch as pt
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

import openmm as mm
from openmm import app, unit

# ---------------------------------------------------------------------------
# MoKiTo
# ---------------------------------------------------------------------------
MOKITO_ROOT = "/home/numerik/jkresse/code/MoKiTo"
sys.path.insert(0, MOKITO_ROOT)
from src.isokann.modules3 import NeuralNetwork, power_method, scale_and_shift

# ---------------------------------------------------------------------------
# amore
# ---------------------------------------------------------------------------
import amore
from amore.sims import OpenMMSimulation, phi, psi, pairnet_nodes
from amore.sims.openmm_sim import (
    DEFAULT_PDB, FORCE_AMBER,
    _get_positions,
)
from amore.features import make_featurizer
from amore.mep import reaction_path_minimum, transition_state

# Atom indices for alanine dipeptide (0-based), from openmm_sim.py
_PHI_IDX = (4, 6, 8, 14)   # C(ACE)–N(ALA)–CA(ALA)–C(ALA)
_PSI_IDX = (6, 8, 14, 16)  # N(ALA)–CA(ALA)–C(ALA)–N(NME)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
pt.manual_seed(SEED)

OUT_DIR     = "alaninedipeptide_metad_out"
SCRATCH_DIR = "/scratch/htc/jkresse/AMORE/"
os.makedirs(OUT_DIR,     exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# Simulation
DT       = 2e-3   # ps (2 fs)
TEMP     = 450.0  # K
FRICTION = 1.0    # ps⁻¹

# MetaD  (openmm.app.Metadynamics parameters)
METAD_STEPS  = 50_000_000  # total MetaD integration steps (~50 ns)
METAD_PACE   = 1000        # hill deposited every PACE steps (1 ps)
HEIGHT       = 1.2        # kJ/mol initial Gaussian height
SIGMA_PHI    = 0.2       # rad
SIGMA_PSI    = 0.2       # rad
BIASFACTOR   = 15.0       # well-tempered gamma; standard MetaD
GRID_SIZE    = 100        # grid points per CV dimension

# How often to snapshot MetaD positions for use as burst seeds
SAVE_EVERY   = 1     # MetaD steps between saved frames (multiple of METAD_PACE)

# Burst (Koopman) sampling
BURST_STEPS  = 2500       # unbiased steps per burst (5 ps)

# Features: all pairwise distances for 22 atoms
N_ATOMS = 22
PAIRS   = [(i, j) for i in range(N_ATOMS) for j in range(i + 1, N_ATOMS)]
N_FEATS = len(PAIRS)   # 231
NODES   = pairnet_nodes(N_FEATS)   # [231, 37, 6, 1]

# Training
NITERS_INIT  = 2000
NITERS_AMORE = 1000
NEPOCHS      = 1
LR           = 1e-3
WD           = 1e-3
BS           = 1000
PATIENCE     = 50
TOL          = 1e-5

# AMORE-MD loop
N_AMORE         = 10
MEP_STEPS       = 100
MEP_STEPSIZE    = 1 / MEP_STEPS
MEP_ENERGY_TOL  = 1e-6
MEP_MAX_ITER    = 500
CHI_LO, CHI_HI = 0.49, 0.51

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"device : {device}")
print(f"pairnet: {NODES}")

# ---------------------------------------------------------------------------
# 1. Build OpenMM system
# ---------------------------------------------------------------------------
print("Building OpenMM system …")

pdb_obj    = app.PDBFile(DEFAULT_PDB)
forcefield = app.ForceField(*FORCE_AMBER)
modeller   = app.Modeller(pdb_obj.topology, pdb_obj.positions)
system     = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.NoCutoff,
    removeCMMotion=False,
)

# ---------------------------------------------------------------------------
# 2. Define collective variables as CustomTorsionForce("theta").
#
#    "energy" of each force = dihedral angle in radians — this is the standard
#    trick to expose a torsion angle as a CV to CustomCVForce / BiasVariable.
#
#    To use a deep learned CV (openmm-torch), replace these two blocks with:
#        phi_cv = TorchForce(torch.jit.script(your_phi_module))
#        psi_cv = TorchForce(torch.jit.script(your_psi_module))
#    and adjust minValue/maxValue/biasWidth below — everything else is identical.
# ---------------------------------------------------------------------------
phi_cv = mm.CustomTorsionForce("theta")
phi_cv.addTorsion(*_PHI_IDX, [])

psi_cv = mm.CustomTorsionForce("theta")
psi_cv.addTorsion(*_PSI_IDX, [])

phi_bias = app.BiasVariable(phi_cv, -np.pi, np.pi, SIGMA_PHI,
                            periodic=True, gridWidth=GRID_SIZE)
psi_bias = app.BiasVariable(psi_cv, -np.pi, np.pi, SIGMA_PSI,
                            periodic=True, gridWidth=GRID_SIZE)

# ---------------------------------------------------------------------------
# 3. Attach well-tempered MetaD to the system.
#
#    openmm.app.Metadynamics adds a CustomCVForce to the system whose hill
#    parameters are updated via context.setParameter() — no reinitialize(),
#    so the GPU kernels are never rebuilt during the run.
# ---------------------------------------------------------------------------
metad = app.Metadynamics(
    system,
    [phi_bias, psi_bias],
    TEMP   * unit.kelvin,
    BIASFACTOR,
    HEIGHT * unit.kilojoules_per_mole,
    METAD_PACE,
    saveFrequency = METAD_PACE * 100,   # checkpoint hills every 100 depositions
    biasDir       = SCRATCH_DIR,
)

integrator = mm.LangevinMiddleIntegrator(
    TEMP     * unit.kelvin,
    FRICTION / unit.picosecond,
    DT       * unit.picoseconds,
)
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.context.setVelocitiesToTemperature(TEMP * unit.kelvin)
simulation.minimizeEnergy()

n_atoms = sum(1 for _ in simulation.topology.atoms())
print(f"System: {n_atoms} atoms, {METAD_STEPS} MetaD steps "
      f"(~{METAD_STEPS*DT/1000:.1f} ns), pace={METAD_PACE} steps")

# ---------------------------------------------------------------------------
# 4. Run MetaD
#    Loop in METAD_PACE-sized chunks so we can log CV values at each hill
#    and snapshot positions every SAVE_EVERY steps.
# ---------------------------------------------------------------------------
print(f"Running well-tempered MetaD (gamma={BIASFACTOR}, h0={HEIGHT} kJ/mol) …")

n_hills      = METAD_STEPS // METAD_PACE
snap_every_h = max(1, SAVE_EVERY // METAD_PACE)  # hills between position snapshots

_cv1_rows  = []
_cv2_rows  = []
_traj_rows = []

it = range(n_hills)
if _HAS_TQDM:
    it = _tqdm(it, desc="MetaD hills", unit="hill",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

for k in it:
    metad.step(simulation, METAD_PACE)
    cvs = metad.getCollectiveVariables(simulation)
    _cv1_rows.append(float(cvs[0]))
    _cv2_rows.append(float(cvs[1]))

    if k % snap_every_h == 0:
        _traj_rows.append(_get_positions(simulation.context).copy())

# Collect and save — keep private names above so a stray re-paste of the
# initialisation block (metad_traj = []) cannot wipe the results.
cv1_hist   = np.array(_cv1_rows)
cv2_hist   = np.array(_cv2_rows)
metad_traj = np.array(_traj_rows)
print(f"MetaD done: {n_hills} hills, {len(metad_traj)} saved frames")

np.save(os.path.join(SCRATCH_DIR, "metad_traj.npy"), metad_traj)
np.save(os.path.join(SCRATCH_DIR, "metad_cv1.npy"),  cv1_hist)
np.save(os.path.join(SCRATCH_DIR, "metad_cv2.npy"),  cv2_hist)

# ---------------------------------------------------------------------------
# 5. Free-energy surface
#    getFreeEnergy() returns the accumulated well-tempered FES estimate as a
#    numpy array of shape (GRID_SIZE, GRID_SIZE) in kJ/mol.
# ---------------------------------------------------------------------------
fes = metad.getFreeEnergy()          # shape (GRID_SIZE, GRID_SIZE), kJ/mol
fes = fes - fes.min()                # shift minimum to 0

s1_ax = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)   # phi axis
s2_ax = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)   # psi axis

np.save(os.path.join(SCRATCH_DIR, "metad_fes.npy"), fes)

# ---------------------------------------------------------------------------
# 6. Select burst seeds from MetaD trajectory
#    Cap at MAX_PER_CELL frames per 40×40 (phi, psi) grid cell; take as many
#    as available when a cell has fewer than the cap.
# ---------------------------------------------------------------------------
#metad_traj = np.load(os.path.join(SCRATCH_DIR, "metad_traj.npy"))
#cv1_hist   = np.load(os.path.join(SCRATCH_DIR, "metad_cv1.npy"))
#cv2_hist   = np.load(os.path.join(SCRATCH_DIR, "metad_cv2.npy"))
#fes        = np.load(os.path.join(SCRATCH_DIR, "metad_fes.npy"))
#s1_ax      = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)
#s2_ax      = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)

MAX_PER_CELL = 100

# Bin each saved MetaD frame by its actual phi/psi coordinates.
# Using phi()/psi() on the coordinates (rather than cv1_hist) ensures
# the binning is consistent with _in_B and all downstream cell lookups.
_traj_phi = phi(metad_traj)
_traj_psi = psi(metad_traj)
_phi_bins = np.clip(np.digitize(_traj_phi, _grid_edges) - 1, 0, DISC_GRID - 1)
_psi_bins = np.clip(np.digitize(_traj_psi, _grid_edges) - 1, 0, DISC_GRID - 1)

_seed_idx = []
for gi in range(DISC_GRID):
    for gj in range(DISC_GRID):
        cell_idx = np.where((_phi_bins == gi) & (_psi_bins == gj))[0]
        if len(cell_idx) == 0:
            continue
        _seed_idx.append(np.random.choice(cell_idx, MAX_PER_CELL, replace=True))

seed_idx = np.concatenate(_seed_idx)
seeds    = metad_traj[seed_idx]
X0=seeds

n_occupied = len(_seed_idx)
n_total    = DISC_GRID * DISC_GRID
print(f"Grid occupancy: {n_occupied} / {n_total} cells occupied "
      f"({'all cells covered' if n_occupied == n_total else f'{n_total - n_occupied} cells empty'}).")
print(f"Selected {len(seeds)} burst seeds "
      f"({MAX_PER_CELL} per occupied {DISC_GRID}×{DISC_GRID} grid cell, with replacement).")

# ---------------------------------------------------------------------------
# 7. Burst Koopman pairs (unbiased plain MD)
# ---------------------------------------------------------------------------
sim_plain = OpenMMSimulation(steps=BURST_STEPS, dt=DT, temp=TEMP)

print(f"Generating {len(seeds)} Koopman pairs "
      f"(unbiased, {BURST_STEPS} steps = {BURST_STEPS*DT*1000:.0f} fs each) …")
X0, Xtau = sim_plain.koopman_pairs(seeds)
print(f"Koopman pairs: X0={X0.shape}, Xtau={Xtau.shape}")

np.save(os.path.join(SCRATCH_DIR, "X0_metad.npy"),   X0)
np.save(os.path.join(SCRATCH_DIR, "Xtau_metad.npy"), Xtau)

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------
ticks  = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

DISC_GRID = 40   # number of discrete bins per CV dimension
_grid_edges = np.linspace(-np.pi, np.pi, DISC_GRID + 1)

def _overlay_grid(ax):
    """Draw a 40×40 bin grid on a [-π, π]² Ramachandran axes."""
    for v in _grid_edges:
        ax.axvline(v, color="k", lw=0.3, alpha=0.4, zorder=4)
        ax.axhline(v, color="k", lw=0.3, alpha=0.4, zorder=4)

# MetaD Ramachandran coloured by simulation progress
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(cv1_hist, cv2_hist, s=1, alpha=0.3, rasterized=True,
                c=np.arange(len(cv1_hist)), cmap="plasma")
plt.colorbar(sc, ax=ax, label="Hill index")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title("MetaD trajectory (phi/psi at hill deposition)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
_overlay_grid(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metad_ramachandran.png"), dpi=200)
plt.close()

# Reconstructed FES
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.contourf(s1_ax, s2_ax, fes, levels=30, cmap="RdYlBu_r")
plt.colorbar(im, ax=ax, label=r"$F(\phi, \psi)$ [kJ/mol]")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title("Reconstructed FES (well-tempered MetaD)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
_overlay_grid(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metad_fes.png"), dpi=200)
plt.close()

# Ramachandran of burst seeds
phi_X0 = phi(X0)
psi_X0 = psi(X0)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(phi_X0, psi_X0, s=4, alpha=0.6, rasterized=True)
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title("Burst seeds (MetaD-enhanced X0)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
_overlay_grid(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "burst_seeds_ramachandran.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 8. Transfer operator on the DISC_GRID² Ramachandran grid
# ---------------------------------------------------------------------------
from scipy.linalg import eig as scipy_eig

N_CELLS = DISC_GRID * DISC_GRID   # 1600 for DISC_GRID=40
N_EIG   = 20

def _to_cell(coords):
    """(N, n_atoms, 3) coords → flat cell index array (N,) on DISC_GRID²."""
    ph = phi(coords)
    ps = psi(coords)
    bi = np.clip(np.digitize(ph, _grid_edges) - 1, 0, DISC_GRID - 1)
    bj = np.clip(np.digitize(ps, _grid_edges) - 1, 0, DISC_GRID - 1)
    return bi * DISC_GRID + bj

print(f"\nEstimating transfer operator ({N_CELLS}×{N_CELLS}) …")
ci = _to_cell(X0)
cj = _to_cell(Xtau)

# Count matrix C[i, j] = number of transitions i → j
C = np.zeros((N_CELLS, N_CELLS), dtype=np.float64)
np.add.at(C, (ci, cj), 1.0)

# Row-normalise to row-stochastic T; leave empty rows as zero
row_sums = C.sum(axis=1, keepdims=True)
occupied = row_sums[:, 0] > 0
T        = np.zeros_like(C)
T[occupied] = C[occupied] / row_sums[occupied]

EPS = 1e-3   # uniform mixing probability
T   = (1.0 - EPS) * T + EPS / N_CELLS

np.save(os.path.join(SCRATCH_DIR, "transfer_op.npy"), T)
print(f"  Occupied rows: {occupied.sum()} / {N_CELLS}  (eps={EPS})")

# Eigendecomposition — left eigenvectors: v T = λ v  (i.e. right evecs of T^T)
vals, lvecs = scipy_eig(T.T)
order = np.argsort(vals.real)[::-1]
vals  = vals[order]
lvecs = lvecs[:, order]

# Right eigenvectors: T v = λ v
_, rvecs = scipy_eig(T)
rvecs    = rvecs[:, order]


print(f"  Top {N_EIG} eigenvalues: {vals[:N_EIG].real}")
np.save(os.path.join(SCRATCH_DIR, "transfer_eigenvalues.npy"),  vals[:N_EIG])
np.save(os.path.join(SCRATCH_DIR, "transfer_eigenvectors.npy"), lvecs[:, :N_EIG])

# Eigenspectrum plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(vals.real, vals.imag, s=20, alpha=0.6, zorder=3)
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
ax.set_title("Transfer operator eigenvalues")
ax.set_aspect("equal")

ax = axes[1]
ax.plot(np.arange(1, N_EIG + 1), vals[:N_EIG].real, "o-", lw=1.5)
ax.axhline(1, color="k", lw=0.5, ls="--")
ax.set_xlabel("Index"); ax.set_ylabel(r"Re($\lambda$)")
ax.set_title(f"Top {N_EIG} eigenvalues")
ax.set_xticks(np.arange(1, N_EIG + 1))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transfer_eigenspectrum.png"), dpi=200)
plt.close(fig)

# Left eigenvectors on Ramachandran
for ev_idx in range(0, min(5, N_EIG)):
    # cell index = phi_bin * DISC_GRID + psi_bin  →  reshape[phi_bin, psi_bin]
    # pcolormesh(x_edges, y_edges, Z) expects Z[y_idx, x_idx] → transpose
    v = lvecs[:, ev_idx].real.reshape(DISC_GRID, DISC_GRID).T
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(v).max()
    im = ax.pcolormesh(_grid_edges, _grid_edges, v,
                       cmap="RdBu_r", shading="flat", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"left EV {ev_idx+1}")
    ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
    ax.set_title(fr"Left EV {ev_idx+1}  ($\lambda={vals[ev_idx].real:.4f}$)")
    ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"transfer_lev{ev_idx+1}.png"), dpi=200)
    plt.close(fig)

# Right eigenvectors on Ramachandran
for ev_idx in range(0, min(5, N_EIG)):
    v = rvecs[:, ev_idx].real.reshape(DISC_GRID, DISC_GRID).T
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(v).max()
    im = ax.pcolormesh(_grid_edges, _grid_edges, v,
                       cmap="RdBu_r", shading="flat", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"right EV {ev_idx+1}")
    ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
    ax.set_title(fr"Right EV {ev_idx+1}  ($\lambda={vals[ev_idx].real:.4f}$)")
    ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"transfer_rev{ev_idx+1}.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Finite-time transition probability to target set B
#    delta = 5 ps (one propagator step), N = T/delta = 500ps/5ps = 100 steps
# ---------------------------------------------------------------------------
from numpy.linalg import matrix_power

N_STEPS    = 100      # number of propagator applications
B_PHI_DEG  = -108.0  # center of B in degrees
B_PSI_DEG  =  157.5
B_PHI_BINS =  10     # box width in phi (number of grid cells)
B_PSI_BINS =  13     # box width in psi (number of grid cells)

_bw = 2 * np.pi / DISC_GRID   # bin width in radians
_phi_c = int(np.floor((np.deg2rad(B_PHI_DEG) + np.pi) / _bw))
_psi_c = int(np.floor((np.deg2rad(B_PSI_DEG) + np.pi) / _bw))

indicator_B = np.zeros(N_CELLS)
_cell_centers = 0.5 * (_grid_edges[:-1] + _grid_edges[1:])
for di in range(-(B_PHI_BINS // 2), B_PHI_BINS - B_PHI_BINS // 2):
    bi = _phi_c + di
    if bi < 0 or bi >= DISC_GRID:
        continue
    for dj in range(-(B_PSI_BINS // 2), B_PSI_BINS - B_PSI_BINS // 2):
        bj = (_psi_c + dj) % DISC_GRID   # psi is periodic
        indicator_B[bi * DISC_GRID + bj] = 1.0

print(f"\nTarget set B: {int(indicator_B.sum())} cells  "
      f"(phi_c=bin {_phi_c} [{_cell_centers[_phi_c]*180/np.pi:.1f}°], "
      f"psi_c=bin {_psi_c} [{_cell_centers[_psi_c]*180/np.pi:.1f}°])")

# p[i] = P(X_{N*delta} in B | X_0 in cell i)
TN      = matrix_power(T, N_STEPS)
p_reach = TN @ indicator_B

np.save(os.path.join(SCRATCH_DIR, "transition_prob_B.npy"), p_reach)
np.save(os.path.join(SCRATCH_DIR, "transfer_op_N.npy"),     TN)

# p_B(x0, T) for x0 = (phi=103.5°, psi=148.5°)
X0_PHI_DEG, X0_PSI_DEG = 103.5, 148.5
_x0_phi_bin = int(np.clip(np.digitize(np.deg2rad(X0_PHI_DEG), _grid_edges) - 1, 0, DISC_GRID - 1))
_x0_psi_bin = int(np.clip(np.digitize(np.deg2rad(X0_PSI_DEG), _grid_edges) - 1, 0, DISC_GRID - 1))
_x0_cell    = _x0_phi_bin * DISC_GRID + _x0_psi_bin
p_B_x0      = p_reach[_x0_cell]

print(f"\np_B(x0, T={N_STEPS * BURST_STEPS * DT:.0f} ps):"
      f"  x0=({X0_PHI_DEG}°, {X0_PSI_DEG}°)"
      f"  → cell ({_x0_phi_bin}, {_x0_psi_bin})  [bin centers: "
      f"{_cell_centers[_x0_phi_bin]*180/np.pi:.2f}°, "
      f"{_cell_centers[_x0_psi_bin]*180/np.pi:.2f}°]"
      f"\n  p_B = {p_B_x0:.6f}")

B_grid = indicator_B.reshape(DISC_GRID, DISC_GRID).T   # [psi, phi]
p_grid = p_reach.reshape(DISC_GRID, DISC_GRID).T

# Transition probability plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.pcolormesh(_grid_edges, _grid_edges, p_grid,
                   cmap="hot_r", shading="flat", vmin=0, vmax=p_grid.max())
plt.colorbar(im, ax=ax, label=r"$P(X_{N\delta} \in B \mid X_0)$")
ax.contour(_cell_centers, _cell_centers, B_grid, levels=[0.5],
           colors="cyan", linewidths=2)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(fr"Transition probability to $B$ in $N={N_STEPS}$ steps "
             fr"($T={N_STEPS * BURST_STEPS * DT:.0f}$ ps)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transition_prob_B.png"), dpi=200)
plt.close(fig)

# Stationary distribution with target set B overlay
pi_vec  = lvecs[:, 0].real
pi_vec  = np.abs(pi_vec) / np.abs(pi_vec).sum()
pi_grid = pi_vec.reshape(DISC_GRID, DISC_GRID).T

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.pcolormesh(_grid_edges, _grid_edges, pi_grid,
                   cmap="viridis", shading="flat")
plt.colorbar(im, ax=ax, label=r"$\pi$ (stationary distribution)")
ax.contour(_cell_centers, _cell_centers, B_grid, levels=[0.5],
           colors="red", linewidths=2, label="Target set $B$")
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(r"Stationary distribution $\pi$ with target set $B$")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "stationary_dist_with_B.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 10. Monte Carlo estimate of p_B(x0, T=500 ps) via unbiased trajectories
# ---------------------------------------------------------------------------
import time as _time

T_MC_PS  = 500.0             # total propagation time per trajectory [ps]
MC_STEPS = int(T_MC_PS / DT) # integration steps (250 000)
N_MC     = 500               # total MC trajectories
N_TIMING = 5                 # short pilot batch for time estimation

# Seeds: MetaD frames whose CV at snapshot time falls in the x0 cell
_x0_mask  = (_phi_bins == _x0_phi_bin) & (_psi_bins == _x0_psi_bin)
_x0_pool  = metad_traj[_x0_mask]
if len(_x0_pool) == 0:
    raise RuntimeError(f"No MetaD frames in x0 cell ({_x0_phi_bin}, {_x0_psi_bin})")
print(f"\nMC: {len(_x0_pool)} MetaD frames available in x0 cell, "
      f"running {N_MC} trajectories of {T_MC_PS:.0f} ps each.")
np.save(os.path.join(SCRATCH_DIR, "mc_initial_states.npy"), _x0_pool)

def _in_B(coords):
    """Return bool array: True if each frame ends in target set B."""
    ph = phi(coords)
    ps = psi(coords)
    bi = np.clip(np.digitize(ph, _grid_edges) - 1, 0, DISC_GRID - 1)
    bj = np.clip(np.digitize(ps, _grid_edges) - 1, 0, DISC_GRID - 1)
    return indicator_B[bi * DISC_GRID + bj].astype(bool)

# Plot MC seeds (all frames in x0 cell)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(phi(_x0_pool), psi(_x0_pool), s=6, alpha=0.6, rasterized=True)
ax.scatter([np.deg2rad(X0_PHI_DEG)], [np.deg2rad(X0_PSI_DEG)],
           c="red", s=80, zorder=5, label=r"$x_0$ target")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(f"MC initial seeds  (x0 cell: {len(_x0_pool)} frames)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
_overlay_grid(ax)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mc_seeds_ramachandran.png"), dpi=200)
plt.close(fig)

# --- Timing pilot ---
_pilot_seeds = _x0_pool[np.random.choice(len(_x0_pool), N_TIMING, replace=True)]
_t0 = _time.perf_counter()
_, _pilot_ends = sim_plain.koopman_pairs(_pilot_seeds, steps=MC_STEPS)
_t_pilot = _time.perf_counter() - _t0
_t_per   = _t_pilot / N_TIMING
print(f"  Timing: {N_TIMING} trajectories in {_t_pilot:.1f} s  "
      f"→ {_t_per:.2f} s/traj, estimated total: {_t_per * N_MC / 60:.1f} min")

# --- Full MC run ---
_hits     = int(_in_B(_pilot_ends).sum())
_done     = N_TIMING
_all_ends = [_pilot_ends]

it_mc = range(0, N_MC - N_TIMING, N_TIMING)
if _HAS_TQDM:
    it_mc = _tqdm(it_mc, desc="MC trajectories", unit=f"batch({N_TIMING})")

for _ in it_mc:
    _batch_seeds = _x0_pool[np.random.choice(len(_x0_pool), N_TIMING, replace=True)]
    _, _ends     = sim_plain.koopman_pairs(_batch_seeds, steps=MC_STEPS)
    _hits     += int(_in_B(_ends).sum())
    _done     += N_TIMING
    _all_ends.append(_ends)

_all_ends = np.concatenate(_all_ends, axis=0)
np.save(os.path.join(SCRATCH_DIR, "mc_endpoints.npy"), _all_ends)

# Wilson 95% confidence interval
_z    = 1.96
_n    = _done
_k    = _hits
_p_w  = (_k + _z**2 / 2) / (_n + _z**2)
_half = _z * np.sqrt(_p_w * (1 - _p_w) / (_n + _z**2))
_lo, _hi = _p_w - _half, _p_w + _half

print(f"\nMonte Carlo result  (N={_n} trajectories, T={T_MC_PS:.0f} ps):")
print(f"  hits          = {_k}")
print(f"  p_B(x0, T)    = {_k / _n:.6f}  (raw fraction)")
print(f"  Wilson 95% CI = [{_lo:.6f}, {_hi:.6f}]")
print(f"  Propagator (500 ps) = {p_B_x0:.6f}")

np.save(os.path.join(SCRATCH_DIR, "mc_result_p_B.npy"),
        np.array([_n, _k, _k / _n, _lo, _hi, p_B_x0]))

