"""
ISOKANN on alanine dipeptide (no water, amber14-all.xml).

Workflow
--------
1. Create OpenMM simulation (default: alanine dipeptide, amber14-all.xml, 310 K).
2. Run a short equilibration trajectory to seed Koopman pairs.
3. Generate (X0, X_tau) Koopman pairs by burst propagation.
4. Featurise with all 231 pairwise distances.
5. Train chi network via the power method (MoKiTo).
6. Plot Ramachandran diagram coloured by chi.
7. Compute chi-MEP from the transition-state region.

Ported from ISOKANN.jl by axsk (https://github.com/axsk/ISOKANN.jl),
"""

import sys
import os

import numpy as np
import torch as pt
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# MoKiTo: local import only
# ---------------------------------------------------------------------------
MOKITO_ROOT = "/home/numerik/jkresse/code/MoKiTo"
sys.path.insert(0, MOKITO_ROOT)
from src.isokann.modules3 import NeuralNetwork, power_method, scale_and_shift

# ---------------------------------------------------------------------------
# amore
# ---------------------------------------------------------------------------
from openmm import unit

import amore
from amore.sims import OpenMMSimulation, pairnet_nodes, phi, psi
from amore.sims.openmm_sim import koopman_pairs_lag
from amore.features import make_featurizer
from amore.mep import reaction_path_minimum, transition_state

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
pt.manual_seed(SEED)

OUT_DIR     = "alaninedipeptide_out"
SCRATCH_DIR = "/scratch/htc/jkresse/AMORE/"
os.makedirs(OUT_DIR,     exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# Simulation  (defaults match ISOKANN.jl OpenMMSimulation)
DT          = 2e-3      # integration time step in ps (2 fs)
TRAJ_STEPS  = 100000     # long trajectory length (× DT = 20 ps); all frames become X0


# Koopman sampling  (CGEM: trajectorydata_bursts(sim, 10000, 1) ≡ all TRAJ_STEPS frames × 1 burst)
N_KOOPMAN   = TRAJ_STEPS  # one burst per trajectory frame

# Features: all pairwise distances for 22 atoms
N_ATOMS = 22
PAIRS   = [(i, j) for i in range(N_ATOMS) for j in range(i + 1, N_ATOMS)]
N_FEATS = len(PAIRS)   # 231

# Network: pairnet architecture from ISOKANN.jl models.jl
#   pairnet(n=231, layers=3) → [231, 37, 6, 1]
NODES    = pairnet_nodes(N_FEATS)   # [231, 37, 6, 1]

# Training  (hyperparams from CGEM/alaninedipeptide.jl)
NITERS   = 5000
NEPOCHS  = 1
LR       = 1e-3
WD       = 1e-4
BS       = 100        
PATIENCE = 5000 #no early stopping
TOL      = 1e-6

# chi-MEP
MEP_STEPS       = 100
MEP_STEPSIZE    = 1 / MEP_STEPS
MEP_ENERGY_TOL  = 1e-5
MEP_MAX_ITER    = 250
CHI_LO, CHI_HI = 0.45, 0.55

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"pairnet nodes: {NODES}")

# ---------------------------------------------------------------------------
# 1. Create simulation  (OpenMMSimulation() mirrors Julia's OpenMMSimulation())
#    No pdb= defaults to alanine dipeptide, amber14-all.xml, 310 K
# ---------------------------------------------------------------------------
print("Setting up OpenMM simulation …")
sim = OpenMMSimulation(steps=1, dt=DT)
print(sim)

# ---------------------------------------------------------------------------
# 1. Long trajectory  (save every step → TRAJ_STEPS frames = X0)
#    Mirrors: trajectory(sim, TRAJ_STEPS; saveevery=1) in ISOKANN.jl
# ---------------------------------------------------------------------------
print(f"Running {TRAJ_STEPS}-step trajectory (saving every step) …")
traj = sim.trajectory(T=TRAJ_STEPS * DT, save_every_steps=100)
print(f"trajectory frames: {traj.shape}")   # (TRAJ_STEPS+1, 66)

# ---------------------------------------------------------------------------
# 2. Koopman pairs  (burst each frame for BURST_STEPS steps)
# ---------------------------------------------------------------------------
print(f"Generating {len(traj)} Koopman pairs ({BURST_STEPS} steps = {BURST_STEPS*DT*1000:.0f} fs each) …")
X0, Xtau = koopman_pairs_lag(traj, lag=10)
print(f"Koopman pairs: X0={X0.shape}, Xtau={Xtau.shape}")

np.save(os.path.join(SCRATCH_DIR, "X0_ad.npy"),   X0)
np.save(os.path.join(SCRATCH_DIR, "Xtau_ad.npy"), Xtau)

X0 = np.load(os.path.join(SCRATCH_DIR, "X0_ad.npy"))
Xtau = np.load(os.path.join(SCRATCH_DIR, "Xtau_ad.npy"))


# ---------------------------------------------------------------------------
# 4. Featurise with pairwise distances
# ---------------------------------------------------------------------------
featurizer = make_featurizer(PAIRS)   # (batch, 66) → (batch, 231)

X0_t   = featurizer(pt.from_numpy(X0).float()).to(device)
Xtau_t = featurizer(pt.from_numpy(Xtau).float()).to(device)
print(f"feature tensors: X0={X0_t.shape}, Xtau={Xtau_t.shape}")

# ---------------------------------------------------------------------------
# 5. Train chi network
# ---------------------------------------------------------------------------
# Julia's pairnet prepends Flux.LayerNorm(n) before the dense layers;
# mirror that with nn.LayerNorm on the feature dimension.
def make_network(activation_function="sigmoid"):
    return pt.nn.Sequential(
        pt.nn.LayerNorm(N_FEATS),
        NeuralNetwork(Nodes=np.array(NODES), activation_function=activation_function),
    ).to(device)

f_NN = make_network("sigmoid")

train_loss, val_loss, best_loss, convergence = power_method(
    X0_t, Xtau_t, f_NN, scale_and_shift,
    Niters     = NITERS,
    Nepochs    = NEPOCHS,
    tolerance  = TOL,
    lr         = LR,
    wd         = WD,
    batch_size = BS,
    patience   = PATIENCE,
    print_eta  = True,
    test_size  = 0.1,
    loss       = "full",
)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
pt.save(f_NN.state_dict(), os.path.join(SCRATCH_DIR, "f_NN_ad.pt"))
np.save(os.path.join(SCRATCH_DIR, "train_loss_ad.npy"), np.asarray(train_loss, dtype=float))
np.save(os.path.join(SCRATCH_DIR, "val_loss_ad.npy"),   np.asarray(val_loss,   dtype=float))

# ---------------------------------------------------------------------------
# 7. Evaluate chi on X0
# ---------------------------------------------------------------------------
f_NN.eval()
with pt.no_grad():
    chi = f_NN(X0_t).cpu().numpy().squeeze()   # (N_KOOPMAN,)

# ---------------------------------------------------------------------------
# 8. Loss curves
# ---------------------------------------------------------------------------
plt.plot(np.asarray(train_loss, dtype=float), label="train")
plt.plot(np.asarray(val_loss,   dtype=float), label="validation")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves_ad.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# 9. Ramachandran diagram coloured by chi
# ---------------------------------------------------------------------------
phi_vals = phi(X0)
psi_vals = psi(X0)

ticks  = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(phi_vals, psi_vals, c=chi, cmap="inferno",
                s=2, alpha=0.6, rasterized=True)
plt.colorbar(sc, ax=ax, label=r"$\chi$")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]")
ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(r"Ramachandran — $X_0$ coloured by $\chi$")
ax.set_xticks(ticks, labels)
ax.set_yticks(ticks, labels)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ramachandran_chi.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 10. chi-MEP from the transition-state region
# ---------------------------------------------------------------------------
f_NN = make_network("sigmoid")
f_NN.load_state_dict(pt.load(os.path.join(SCRATCH_DIR, "f_NN_ad.pt"), map_location=device))
f_NN.eval()

X0 = np.load(os.path.join(SCRATCH_DIR, "X0_ad.npy"))

def feat_torch(x_t):
    return featurizer(x_t)

_mep_ctx = sim._sim.context
_n_atoms  = sim.n_atoms

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
    return -f.flatten().astype(np.float64)

ts_frames, ts_chi = transition_state(f_NN, feat_torch, X0,
                                     chi_lo=CHI_LO, chi_hi=CHI_HI)
print(f"Transition-state frames: {len(ts_frames)}  (chi ∈ [{CHI_LO}, {CHI_HI}])")

if len(ts_frames) == 0:
    print("No transition-state frames — widen CHI_LO/CHI_HI or increase N_KOOPMAN.")
else:
    best_idx = np.argmin(np.abs(ts_chi - 0.5))
    x0_ts    = ts_frames[best_idx]
    print(f"Starting MEP from chi = {ts_chi[best_idx]:.4f}")

    path = reaction_path_minimum(
        f_NN, feat_torch, x0_ts,
        steps           = MEP_STEPS,
        stepsize        = MEP_STEPSIZE,
        potential_fn    = _openmm_potential,
        grad_fn         = _openmm_grad,
        energy_tol      = MEP_ENERGY_TOL,
        energy_max_iter = MEP_MAX_ITER,
    )
    np.save(os.path.join(OUT_DIR, "mep_ad.npy"), path)
    print(f"MEP shape: {path.shape}")

    phi_path = phi(path)
    psi_path = psi(path)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(phi_vals, psi_vals, c=chi, cmap="inferno",
                    s=2, alpha=0.3, rasterized=True)
    plt.colorbar(sc, ax=ax, label=r"$\chi$")
    ax.plot(phi_path, psi_path, color="teal", lw=2, label="chi-MEP")
    ax.scatter([phi(x0_ts)], [psi(x0_ts)], color="cyan", zorder=5,
               s=60, label="transition state")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.set_ylabel(r"$\psi$ [rad]")
    ax.set_title("Ramachandran + chi-MEP")
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ramachandran_mep.png"), dpi=200)
    plt.close()

print(f"\nDone. Outputs in: {os.path.abspath(OUT_DIR)}/")
