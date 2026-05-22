"""
Generate Koopman pairs for alanine dipeptide at 450 K.

Protocol
--------
1. Run well-tempered MetaD on φ,ψ at 450 K for METAD_STEPS to fill the FES.
   This explores all three basins (C7eq, C7ax, C7eq').
2. Build a 40×40 grid on (φ,ψ) ∈ [-π,π]².
3. For each grid cell, pick the saved MetaD frame whose (φ,ψ) is closest
   to the cell centre.  Remove cells with no nearby frame (> FILL_TOL rad).
4. From each anchor, run N_BURSTS=20 unbiased burst trajectories of LAG_PS=5 ps.
   Save full Cartesian coordinates and the 231 pairwise-distance features.
5. Save patch-based 80/20 train/test splits (5 seeds, contiguous patches).

Features
--------
  All C(22,2) = 231 pairwise distances for the 22 atoms of alanine dipeptide.

Outputs
-------
  data/alanine_koopman.npz
    anchors_cart  (N_ANC, 22, 3)  — Cartesian nm coords of anchors
    anchors_feat  (N_ANC, 231)    — pairwise distances of anchors
    anchors_phi   (N_ANC,)        — φ angles of anchors
    anchors_psi   (N_ANC,)        — ψ angles of anchors
    bursts_feat   (N_ANC, N_K, 231)
    bursts_phi    (N_ANC, N_K)
    bursts_psi    (N_ANC, N_K)
    patch_splits  (5, N_ANC)      — 0=train 1=test per anchor
    grid_phi      (40,)           — φ grid centres
    grid_psi      (40,)           — ψ grid centres

Notes
-----
  Simulation settings: 450 K, vacuum, AMBER14, 2 fs timestep, Langevin (1 ps⁻¹).
  MetaD: σ=0.35 rad, height=1.2 kJ/mol, biasfactor=10, pace=1000 steps.
  Burst lag: 5 ps = 2500 steps.
"""

import os
import sys
import numpy as np

import openmm as mm
from openmm import app, unit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from amore.sims.openmm_sim import DEFAULT_PDB, FORCE_AMBER

OUTDIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTDIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
TEMP         = 450.0          # K  (benchmark spec)
DT           = 2e-3           # ps  (2 fs)
FRICTION     = 1.0            # ps⁻¹
METAD_STEPS  = 5_000_000      # 10 ns MetaD to fill FES
METAD_PACE   = 1_000          # hill every 1 ps
HEIGHT       = 1.2            # kJ/mol
SIGMA_PHI    = 0.35           # rad
SIGMA_PSI    = 0.35           # rad
BIASFACTOR   = 10.0
GRID_CV      = 100            # MetaD grid per CV axis

LAG_PS       = 5.0            # ps lag time
BURST_STEPS  = int(LAG_PS / DT)   # 2500 steps
N_BURSTS     = 20

PHI_GRID_N   = 40             # φ grid points
PSI_GRID_N   = 40             # ψ grid points
FILL_TOL     = 0.25           # rad — discard cell if nearest MetaD frame > this

PATCH_N      = 4              # 4×4 = 16 patches
N_HOLD       = 4              # patches held out
N_SEEDS      = 5

SEED = 42
rng  = np.random.default_rng(SEED)

# Atom indices (0-based)
_PHI_IDX = (4, 6, 8, 14)
_PSI_IDX = (6, 8, 14, 16)
N_ATOMS  = 22
PAIRS    = [(i, j) for i in range(N_ATOMS) for j in range(i + 1, N_ATOMS)]


# ── Helper functions ───────────────────────────────────────────────────────────

def compute_phi_psi(pos_nm: np.ndarray) -> tuple[float, float]:
    """Compute φ,ψ dihedrals from positions in nm."""
    def dihedral(p0, p1, p2, p3):
        b1 = p1 - p0; b2 = p2 - p1; b3 = p3 - p2
        n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
        n1 /= np.linalg.norm(n1) + 1e-12
        n2 /= np.linalg.norm(n2) + 1e-12
        m  = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
        return np.arctan2(m @ n2, n1 @ n2)
    p = pos_nm
    phi = dihedral(p[_PHI_IDX[0]], p[_PHI_IDX[1]], p[_PHI_IDX[2]], p[_PHI_IDX[3]])
    psi = dihedral(p[_PSI_IDX[0]], p[_PSI_IDX[1]], p[_PSI_IDX[2]], p[_PSI_IDX[3]])
    return float(phi), float(psi)

def pairwise_distances(pos_nm: np.ndarray) -> np.ndarray:
    """231 pairwise distances (nm) from (22,3) positions."""
    return np.array([np.linalg.norm(pos_nm[i] - pos_nm[j]) for i, j in PAIRS],
                    dtype=np.float32)


# ── 1. Build OpenMM system ────────────────────────────────────────────────────
print("Building OpenMM system (450 K, vacuum, AMBER14) …")
pdb_obj    = app.PDBFile(DEFAULT_PDB)
forcefield = app.ForceField(*FORCE_AMBER)
modeller   = app.Modeller(pdb_obj.topology, pdb_obj.positions)
system     = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.NoCutoff,
                                     removeCMMotion=False)

phi_cv = mm.CustomTorsionForce("theta")
phi_cv.addTorsion(*_PHI_IDX, [])
psi_cv = mm.CustomTorsionForce("theta")
psi_cv.addTorsion(*_PSI_IDX, [])

phi_bias = app.BiasVariable(phi_cv, -np.pi, np.pi, SIGMA_PHI,
                            periodic=True, gridWidth=GRID_CV)
psi_bias = app.BiasVariable(psi_cv, -np.pi, np.pi, SIGMA_PSI,
                            periodic=True, gridWidth=GRID_CV)

bias_dir = os.path.join(OUTDIR, "alanine_metad_bias")
os.makedirs(bias_dir, exist_ok=True)

metad = app.Metadynamics(
    system, [phi_bias, psi_bias], TEMP * unit.kelvin,
    BIASFACTOR, HEIGHT * unit.kilojoules_per_mole,
    METAD_PACE, saveFrequency=METAD_PACE,
    biasDir=bias_dir,
)

integrator = mm.LangevinMiddleIntegrator(
    TEMP * unit.kelvin, FRICTION / unit.picosecond, DT * unit.picoseconds
)

simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(TEMP * unit.kelvin)


# ── 2. Run MetaD, collect frames ───────────────────────────────────────────────
print(f"Running {METAD_STEPS:,} steps of MetaD at {TEMP} K …")
SAVE_EVERY   = METAD_PACE    # save one frame per hill
n_save       = METAD_STEPS // SAVE_EVERY

metad_cart   = []   # list of (22,3) arrays
metad_phi    = []
metad_psi    = []

for step_i in range(n_save):
    metad.step(simulation, SAVE_EVERY)
    state = simulation.context.getState(getPositions=True)
    pos   = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))  # (22,3)
    phi_, psi_ = compute_phi_psi(pos)
    metad_cart.append(pos.astype(np.float32))
    metad_phi.append(phi_)
    metad_psi.append(psi_)
    if (step_i + 1) % 1000 == 0:
        print(f"  MetaD step {(step_i+1)*SAVE_EVERY:,} / {METAD_STEPS:,}")

metad_cart = np.array(metad_cart)    # (n_save, 22, 3)
metad_phi  = np.array(metad_phi)
metad_psi  = np.array(metad_psi)
print(f"  Collected {len(metad_cart):,} MetaD frames")


# ── 3. Build 40×40 anchor grid ────────────────────────────────────────────────
phi_centres = np.linspace(-np.pi, np.pi, PHI_GRID_N, endpoint=False) + np.pi / PHI_GRID_N
psi_centres = np.linspace(-np.pi, np.pi, PSI_GRID_N, endpoint=False) + np.pi / PSI_GRID_N

anchor_cart = []
anchor_phi  = []
anchor_psi  = []
anchor_gphi = []   # grid index
anchor_gpsi = []

for gp, phi_c in enumerate(phi_centres):
    for gq, psi_c in enumerate(psi_centres):
        # Angular distance (wrap-around)
        dphi = np.abs(np.arctan2(np.sin(metad_phi - phi_c), np.cos(metad_phi - phi_c)))
        dpsi = np.abs(np.arctan2(np.sin(metad_psi - psi_c), np.cos(metad_psi - psi_c)))
        dist = np.sqrt(dphi**2 + dpsi**2)
        best = int(np.argmin(dist))
        if dist[best] < FILL_TOL:
            anchor_cart.append(metad_cart[best])
            anchor_phi.append(metad_phi[best])
            anchor_psi.append(metad_psi[best])
            anchor_gphi.append(gp)
            anchor_gpsi.append(gq)

anchor_cart  = np.array(anchor_cart,  dtype=np.float32)   # (N_ANC, 22, 3)
anchor_phi   = np.array(anchor_phi,   dtype=np.float32)
anchor_psi   = np.array(anchor_psi,   dtype=np.float32)
anchor_gphi  = np.array(anchor_gphi,  dtype=np.int32)
anchor_gpsi  = np.array(anchor_gpsi,  dtype=np.int32)
N_ANC        = len(anchor_cart)
anchor_feat  = np.array([pairwise_distances(c) for c in anchor_cart])  # (N_ANC, 231)
print(f"  Grid cells filled: {N_ANC} / {PHI_GRID_N*PSI_GRID_N}")


# ── 4. Burst propagation ──────────────────────────────────────────────────────
print(f"Propagating {N_ANC} × {N_BURSTS} bursts ({BURST_STEPS} steps each) …")
bursts_feat = np.empty((N_ANC, N_BURSTS, len(PAIRS)), dtype=np.float32)
bursts_phi  = np.empty((N_ANC, N_BURSTS),             dtype=np.float32)
bursts_psi  = np.empty((N_ANC, N_BURSTS),             dtype=np.float32)

# Build an unbiased context for burst propagation (no MetaD bias)
unbias_sys  = forcefield.createSystem(modeller.topology,
                                      nonbondedMethod=app.NoCutoff,
                                      removeCMMotion=False)
unbias_int  = mm.LangevinMiddleIntegrator(
    TEMP * unit.kelvin, FRICTION / unit.picosecond, DT * unit.picoseconds
)
unbias_sim  = app.Simulation(modeller.topology, unbias_sys, unbias_int)

for i, pos0 in enumerate(anchor_cart):
    pos0_q = (pos0 * unit.nanometer).tolist()
    for k in range(N_BURSTS):
        unbias_sim.context.setPositions(pos0_q)
        unbias_sim.context.setVelocitiesToTemperature(TEMP * unit.kelvin)
        unbias_sim.step(BURST_STEPS)
        state = unbias_sim.context.getState(getPositions=True)
        pos1  = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                         dtype=np.float32)
        bursts_feat[i, k] = pairwise_distances(pos1)
        bursts_phi[i, k], bursts_psi[i, k] = compute_phi_psi(pos1)
    if (i + 1) % 100 == 0:
        print(f"  anchor {i+1}/{N_ANC}")

print("Done propagating.")


# ── 5. Patch-based train/test splits ─────────────────────────────────────────
patch_id    = anchor_gphi * PATCH_N + (anchor_gpsi * PATCH_N // PSI_GRID_N)
all_patches = list(range(PATCH_N * PATCH_N))
split_rng   = np.random.default_rng(SEED + 200)

patch_splits = np.zeros((N_SEEDS, N_ANC), dtype=np.int8)
for s in range(N_SEEDS):
    hold = split_rng.choice(all_patches, size=N_HOLD, replace=False)
    for h in hold:
        patch_splits[s, patch_id == h] = 1
    print(f"  seed {s}: {(patch_splits[s]==0).sum()} train / {(patch_splits[s]==1).sum()} test")


# ── 6. Save ───────────────────────────────────────────────────────────────────
np.savez(
    os.path.join(OUTDIR, "alanine_koopman.npz"),
    anchors_cart  = anchor_cart,     # (N_ANC, 22, 3)
    anchors_feat  = anchor_feat,     # (N_ANC, 231)
    anchors_phi   = anchor_phi,      # (N_ANC,)
    anchors_psi   = anchor_psi,
    bursts_feat   = bursts_feat,     # (N_ANC, N_K, 231)
    bursts_phi    = bursts_phi,
    bursts_psi    = bursts_psi,
    patch_splits  = patch_splits,    # (5, N_ANC)
    grid_phi      = phi_centres.astype(np.float32),
    grid_psi      = psi_centres.astype(np.float32),
    n_bursts      = np.array([N_BURSTS]),
    lag_ps        = np.array([LAG_PS]),
    temp_K        = np.array([TEMP]),
)
print(f"\nSaved: data/alanine_koopman.npz")
print(f"  anchors_feat={anchor_feat.shape}  bursts_feat={bursts_feat.shape}")
