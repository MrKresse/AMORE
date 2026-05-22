"""
Well-tempered metadynamics (MetaD) on alanine dipeptide (phi/psi),
followed by Koopman burst-pair sampling for ISOKANN.

Workflow
--------
1. Build an OpenMM system with a 2D well-tempered MetaD bias on phi and psi.
   CVs are defined as CustomTorsionForce objects whose "energy" = the dihedral
   angle (in radians).  This interface is designed to be drop-in replaceable
   with openmm-torch TorchForce CVs for deep learned collective variables.
2. Run MetaD via openmm.app.Metadynamics (built-in, GPU-resident bias — updates
   hill parameters without reinitializing the context).
3. Reconstruct the free-energy surface (FES) from the accumulated bias.
4. Collect thinned MetaD frames as seeds.
5. Generate (X0, X_tau) Koopman pairs via short unbiased burst propagation.
6. Save all arrays and produce diagnostic plots.
"""

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
# Set MOKITO_ROOT to your local MoKiTo checkout, or export MOKITO_ROOT=/path/to/MoKiTo
MOKITO_ROOT = os.environ.get("MOKITO_ROOT", "")
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
SCRATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR,     exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# Simulation
DT       = 2e-3   # ps (2 fs)
TEMP     = 310.0  # K
FRICTION = 1.0    # ps⁻¹

# MetaD  (openmm.app.Metadynamics parameters)
METAD_STEPS  = 10_000_000  # total MetaD integration steps (~10 ns)
METAD_PACE   = 1000        # hill deposited every PACE steps (1 ps)
HEIGHT       = 1.2        # kJ/mol initial Gaussian height
SIGMA_PHI    = 0.35       # rad
SIGMA_PSI    = 0.35       # rad
BIASFACTOR   = 15.0       # well-tempered gamma; gamma→∞ → standard MetaD
GRID_SIZE    = 100        # grid points per CV dimension

# How often to snapshot MetaD positions for use as burst seeds
SAVE_EVERY   = 1     # MetaD steps between saved frames (multiple of METAD_PACE)

# Burst (Koopman) sampling
BURST_STEPS  = 100        # unbiased steps per burst (~1 ps)
N_SEEDS      = 10000       # max MetaD frames to use as burst seeds

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
# ---------------------------------------------------------------------------
#metad_traj = np.load(os.path.join(SCRATCH_DIR, "metad_traj.npy"))
#cv1_hist   = np.load(os.path.join(SCRATCH_DIR, "metad_cv1.npy"))
#cv2_hist   = np.load(os.path.join(SCRATCH_DIR, "metad_cv2.npy"))
#fes        = np.load(os.path.join(SCRATCH_DIR, "metad_fes.npy"))
#s1_ax      = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)
#s2_ax      = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)

if len(metad_traj) > N_SEEDS:
    idx   = np.random.choice(len(metad_traj), N_SEEDS, replace=False)
    seeds = metad_traj[idx]
else:
    seeds = metad_traj

print(f"Selected {len(seeds)} / {len(metad_traj)} MetaD frames as burst seeds.")

# ---------------------------------------------------------------------------
# 7. Burst Koopman pairs (unbiased plain MD)
# ---------------------------------------------------------------------------
sim_plain = OpenMMSimulation(steps=BURST_STEPS, dt=DT)

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

# MetaD Ramachandran coloured by simulation progress
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(cv1_hist, cv2_hist, s=1, alpha=0.3, rasterized=True,
                c=np.arange(len(cv1_hist)), cmap="plasma")
plt.colorbar(sc, ax=ax, label="Hill index")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title("MetaD trajectory (phi/psi at hill deposition)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
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
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "burst_seeds_ramachandran.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 8. Load Koopman pairs and featurise
# ---------------------------------------------------------------------------
X0   = np.load(os.path.join(SCRATCH_DIR, "X0_metad.npy"))
Xtau = np.load(os.path.join(SCRATCH_DIR, "Xtau_metad.npy"))

featurizer = make_featurizer(PAIRS)

def featurize(coords_np):
    """(N, 66) numpy → (N, 231) float32 torch tensor on device."""
    return featurizer(pt.from_numpy(coords_np.astype(np.float32)).to(device))

X0_t   = featurize(X0)
Xtau_t = featurize(Xtau)
print(f"Feature tensors: X0={X0_t.shape}, Xtau={Xtau_t.shape}")

# ---------------------------------------------------------------------------
# 9. Network definition
# ---------------------------------------------------------------------------
class NeuralNetworkLN(NeuralNetwork):
    """pairnet with LayerNorm on the input."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden, act = self.hidden_layers, self.activation
        del self.hidden_layers, self.activation, self.activation2
        self.input_norm    = pt.nn.LayerNorm(self.input_size)
        self.hidden_layers = hidden
        self.activation    = act

    def forward(self, X):
        X = self.input_norm(X)
        for layer in self.hidden_layers[:-1]:
            X = self.activation(layer(X))
        return self.hidden_layers[-1](X).squeeze(-1)


f_NN = NeuralNetworkLN(Nodes=np.array(NODES), activation_function="sigmoid").to(device)

# ---------------------------------------------------------------------------
# 10. Initial training
# ---------------------------------------------------------------------------
print(f"\nInitial training: Niters={NITERS_INIT}, BS={BS}, lr={LR}, wd={WD} …")
train_loss, val_loss, _, _ = power_method(
    X0_t, Xtau_t, f_NN, scale_and_shift,
    Niters     = NITERS_INIT,
    Nepochs    = NEPOCHS,
    tolerance  = TOL,
    lr         = LR,
    wd         = WD,
    batch_size = BS,
    patience   = PATIENCE,
    print_eta  = True,
    loss       = "full",
)
pt.save(f_NN.state_dict(), os.path.join(SCRATCH_DIR, "f_NN_metad.pt"))

# Loss curves
fig, ax = plt.subplots()
ax.plot(np.asarray(train_loss, dtype=float), label="train")
ax.plot(np.asarray(val_loss,   dtype=float), label="val")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE loss")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_metad_init.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Evaluate chi and plot initial Ramachandran
# ---------------------------------------------------------------------------
f_NN.eval()
with pt.no_grad():
    chi_X0 = f_NN(X0_t).cpu().numpy().squeeze()

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(phi_X0, psi_X0, c=chi_X0, cmap="inferno",
                s=4, alpha=0.6, rasterized=True)
plt.colorbar(sc, ax=ax, label=r"$\chi$")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(r"Ramachandran — $\chi$ after initial training (MetaD seeds)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramachandran_metad_init.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 12. AMORE-MD loop
# ---------------------------------------------------------------------------
sim_amore = OpenMMSimulation(steps=1, dt=DT)

# OpenMM potential and gradient for constrained energy minimisation on chi
# level sets — mirrors Julia's energyminimization_chilevel which is the key
# step that makes the MEP traverse the physical conformational landscape.
# Without this, the Euler step only moves ~stepsize/||∇χ|| nm per step
# (~1e-4 nm) and all path frames cluster near the starting geometry.
_mep_ctx = sim_amore._sim.context
_n_atoms = sim_amore.n_atoms

def _openmm_potential(x):
    _mep_ctx.setPositions(x.reshape(_n_atoms, 3))
    return (_mep_ctx.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole))

def _openmm_grad(x):
    _mep_ctx.setPositions(x.reshape(_n_atoms, 3))
    f = (_mep_ctx.getState(getForces=True)
         .getForces(asNumpy=True)
         .value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
    return -f.flatten().astype(np.float64)   # gradient = -force

def feat_torch(x_t):
    return featurizer(x_t)

X0_all   = X0.copy()
Xtau_all = Xtau.copy()
traj_bg  = X0_all
phi_bg   = phi(X0_all)
psi_bg   = psi(X0_all)

xss = []   # MEP paths across iterations

for amore_iter in range(1, N_AMORE + 1):
    print(f"\n=== AMORE iteration {amore_iter}/{N_AMORE} ===")

    # Transition-state frames
    ts_frames, ts_chi = transition_state(
        f_NN, feat_torch, traj_bg, chi_lo=CHI_LO, chi_hi=CHI_HI
    )
    print(f"  TS frames: {len(ts_frames)}  (chi ∈ [{CHI_LO:.2f}, {CHI_HI:.2f}])")

    if len(ts_frames) == 0:
        print("  No TS frames — widening window to [0.4, 0.6]")
        ts_frames, ts_chi = transition_state(
            f_NN, feat_torch, traj_bg, chi_lo=0.0, chi_hi=1.0
        )
        if len(ts_frames) == 0:
            print("  Still none — skipping.")
            continue

    # chi-MEP from one random TS frame
    j     = random.randrange(len(ts_frames))
    x0_ts = ts_frames[j]
    print(f"  Starting MEP from chi = {ts_chi[j]:.4f}")

    xs_mep = reaction_path_minimum(
        f_NN, feat_torch, x0_ts,
        steps        = MEP_STEPS,
        stepsize     = MEP_STEPSIZE,
        potential_fn = _openmm_potential,
        grad_fn      = _openmm_grad,
        energy_tol   = MEP_ENERGY_TOL,
        energy_max_iter = MEP_MAX_ITER,
    )
    xss.append(xs_mep)
    np.save(os.path.join(SCRATCH_DIR, f"mep_metad_{amore_iter}.npy"), xs_mep)

    # Propagate MEP frames → new Koopman pairs
    print(f"  Propagating {len(xs_mep)} MEP frames …")
    X0_mep, Xtau_mep = sim_amore.koopman_pairs(xs_mep,steps=100)

    X0_all   = np.concatenate([X0_all,   X0_mep],   axis=0)
    Xtau_all = np.concatenate([Xtau_all, Xtau_mep], axis=0)

    X0_t   = featurize(X0_all)
    Xtau_t = featurize(Xtau_all)
    print(f"  Dataset: {X0_all.shape[0]} pairs total")

    traj_bg = X0_all
    phi_bg  = phi(X0_all)
    psi_bg  = psi(X0_all)

    # Re-train
    print(f"  Re-training: Niters={NITERS_AMORE} …")
    f_NN.train()
    power_method(
        X0_t, Xtau_t, f_NN, scale_and_shift,
        Niters     = NITERS_AMORE,
        Nepochs    = NEPOCHS,
        tolerance  = TOL,
        lr         = LR,
        wd         = WD,
        batch_size = BS,
        patience   = PATIENCE,
        print_eta  = True,
    )
    f_NN.eval()

    # Per-iteration Ramachandran
    phi_ts  = float(phi(x0_ts))
    psi_ts  = float(psi(x0_ts))
    phi_mep = phi(xs_mep)
    psi_mep = psi(xs_mep)

    with pt.no_grad():
        chi_mep = f_NN(featurize(xs_mep)).cpu().numpy().squeeze()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(s1_ax, s2_ax, fes, levels=30, cmap="RdYlBu_r", alpha=0.85)
    sc = ax.scatter(phi_mep, psi_mep, c=chi_mep, cmap="inferno",
                    s=6, alpha=0.95, vmin=0, vmax=1,
                    rasterized=True, label=r"$\chi$-MEP states", zorder=3)
    plt.colorbar(sc, ax=ax, label=r"$\chi$")
    ax.scatter([phi_ts], [psi_ts], c="cyan", zorder=5, s=60, label="Initial state")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
    ax.set_title(f"AMORE iteration {amore_iter}")
    ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
    ax.legend(fontsize=7, loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"ramachandran_amore_{amore_iter}.png"), dpi=200)
    plt.close(fig)

pt.save(f_NN.state_dict(), os.path.join(SCRATCH_DIR, "f_NN_metad_enhanced.pt"))
np.save(os.path.join(SCRATCH_DIR, "X0_metad_enhanced.npy"),   X0_all)
np.save(os.path.join(SCRATCH_DIR, "Xtau_metad_enhanced.npy"), Xtau_all)

# ---------------------------------------------------------------------------
# 13. Final plot: all MEP iterations overlaid on FES, coloured by chi
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
ax.contourf(s1_ax, s2_ax, fes, levels=30, cmap="RdYlBu_r", alpha=0.85)
sc = None
for it_idx, xs_it in enumerate(xss):
    with pt.no_grad():
        chi_it = f_NN(featurize(xs_it)).cpu().numpy().squeeze()
    sc = ax.scatter(phi(xs_it), psi(xs_it), c=chi_it, cmap="inferno",
                    s=5, alpha=0.85, vmin=0, vmax=1,
                    rasterized=True, label=f"MEP iter {it_idx+1}", zorder=3)
if sc is not None:
    plt.colorbar(sc, ax=ax, label=r"$\chi$")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]", fontsize=14); ax.set_ylabel(r"$\psi$ [rad]", fontsize=14)
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.set_title(r"FES + $\chi$-MEP iterations (MetaD seeded)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramachandran_metad_iters.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 14. Convergence: RMS angular displacement between consecutive MEPs
# ---------------------------------------------------------------------------
def _wrap_delta(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

if len(xss) >= 2:
    rms_shifts, max_jumps = [], []
    for a, b in zip(xss[:-1], xss[1:]):
        n    = min(len(a), len(b))
        dist = np.sqrt(_wrap_delta(phi(b[:n]) - phi(a[:n])) ** 2
                       + _wrap_delta(psi(b[:n]) - psi(a[:n])) ** 2)
        rms_shifts.append(float(np.sqrt(np.mean(dist ** 2))))
        max_jumps.append(float(np.max(dist)))

    iters = np.arange(2, len(rms_shifts) + 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, rms_shifts, marker="o",  lw=2, label="RMS path shift")
    ax.plot(iters, max_jumps,  marker="D", lw=2, ls="--", label="Max displacement")
    ax.set_xlabel("AMORE-MD iteration", fontsize=13)
    ax.set_ylabel(r"Displacement in $(\phi, \psi)$ [rad]", fontsize=13)
    ax.legend(fontsize=11); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence_metad.png"), dpi=200)
    plt.close(fig)

print(f"\nDone.  Outputs in {os.path.abspath(OUT_DIR)}/")
print(f"Arrays in {os.path.abspath(SCRATCH_DIR)}")
