"""
Train a chi (committor-like) function on the Koopman pairs produced by
alaninedipeptide_discrete.py, then compute and plot the chi-MEP.
"""

import os
import sys
import random

import numpy as np
import torch as pt
import matplotlib.pyplot as plt

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
from amore.sims.openmm_sim import DEFAULT_PDB, FORCE_AMBER
from amore.features import make_featurizer
from amore.mep import (
    reaction_path_minimum, transition_state,
    export_cv_torchscript, build_chi_mep_constrained,
    build_chi_mep_projected, sample_levelset_projected,
    energy_min_on_levelset, select_medoid
)
from amore.mep.core import _chi_val
from openmm import app, unit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
pt.manual_seed(SEED)

SCRATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_DIR     = "alaninedipeptide_mfep_out"
os.makedirs(OUT_DIR, exist_ok=True)

DT   = 2e-3   # ps
TEMP = 450.0  # K  (same as discrete script)

# Features
N_ATOMS = 22
PAIRS   = [(i, j) for i in range(N_ATOMS) for j in range(i + 1, N_ATOMS)]
N_FEATS = len(PAIRS)   # 231
NODES   = pairnet_nodes(N_FEATS)   # [231, 37, 6, 1]

# Training
NITERS   = 2000
NEPOCHS  = 1
LR       = 1e-3
WD       = 1e-3
BS       = 1000
PATIENCE = 50
TOL      = 1e-5

# MEP
MEP_STEPS       = 100
MEP_STEPSIZE    = 1 / MEP_STEPS
MEP_ENERGY_TOL  = 1e-6
MEP_MAX_ITER    = 500
CHI_LO, CHI_HI = 0.49, 0.51
STEPS_PER_LS  = 100   
BURNIN_PER_LS = 0    

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"device : {device}")
print(f"pairnet: {NODES}")

# ---------------------------------------------------------------------------
# 1. Load Koopman pairs
# ---------------------------------------------------------------------------
X0   = np.load(os.path.join(SCRATCH_DIR, "X0_metad.npy"))
Xtau = np.load(os.path.join(SCRATCH_DIR, "Xtau_metad.npy"))
print(f"Loaded Koopman pairs: X0={X0.shape}, Xtau={Xtau.shape}")

featurizer = make_featurizer(PAIRS)

def featurize(coords_np):
    """(N, n_atoms*3) numpy → (N, 231) float32 torch tensor on device."""
    return featurizer(pt.from_numpy(coords_np.astype(np.float32)).to(device))

X0_t   = featurize(X0)
Xtau_t = featurize(Xtau)
print(f"Feature tensors: X0={X0_t.shape}, Xtau={Xtau_t.shape}")

# ---------------------------------------------------------------------------
# 2. Network
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
# 3. Train
# ---------------------------------------------------------------------------
ticks  = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

phi_X0 = phi(X0)
psi_X0 = psi(X0)

print(f"\nTraining: Niters={NITERS}, BS={BS}, lr={LR}, wd={WD} …")
train_loss, val_loss, _, _ = power_method(
    X0_t, Xtau_t, f_NN, scale_and_shift,
    Niters     = NITERS,
    Nepochs    = NEPOCHS,
    tolerance  = TOL,
    lr         = LR,
    wd         = WD,
    batch_size = BS,
    patience   = PATIENCE,
    print_eta  = True,
    loss       = "full",
)
pt.save(f_NN.state_dict(), os.path.join(SCRATCH_DIR, "f_NN_discrete.pt"))

# Load the state dict
state_dict = pt.load(os.path.join(SCRATCH_DIR, "f_NN_discrete.pt"))
f_NN.load_state_dict(state_dict)
f_NN.eval()  # if you're doing inference
# Loss curves

fig, ax = plt.subplots()
ax.plot(np.asarray(train_loss, dtype=float), label="train")
ax.plot(np.asarray(val_loss,   dtype=float), label="val")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE loss")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_discrete.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Evaluate chi and plot Ramachandran
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
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramachandran_chi.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. MEP via chi-level-set minimisation
# ---------------------------------------------------------------------------
sim_mep = OpenMMSimulation(steps=1, dt=DT, temp=450)
_mep_ctx = sim_mep._sim.context
_n_atoms = sim_mep.n_atoms

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

def feat_torch(x_t):
    return featurizer(x_t)

# Pick a transition-state frame
ts_frames, ts_chi = transition_state(
    f_NN, feat_torch, X0, chi_lo=CHI_LO, chi_hi=CHI_HI
)
print(f"\nTS frames: {len(ts_frames)}  (chi ∈ [{CHI_LO}, {CHI_HI}])")
if len(ts_frames) == 0:
    raise RuntimeError("No TS frames found — widen CHI_LO/CHI_HI.")

j     = random.randrange(len(ts_frames))
x0_ts = ts_frames[j]
print(f"Starting MEP from chi = {ts_chi[j]:.4f}")

xs_mep = reaction_path_minimum(
    f_NN, feat_torch, x0_ts,
    steps           = MEP_STEPS,
    stepsize        = MEP_STEPSIZE,
    potential_fn    = _openmm_potential,
    grad_fn         = _openmm_grad,
    energy_tol      = MEP_ENERGY_TOL,
    energy_max_iter = MEP_MAX_ITER,
)
np.save(os.path.join(SCRATCH_DIR, "mep_discrete.npy"), xs_mep)

# Evaluate chi along MEP
with pt.no_grad():
    chi_mep = f_NN(featurize(xs_mep)).cpu().numpy().squeeze()

phi_mep = phi(xs_mep)
psi_mep = psi(xs_mep)

# MEP on Ramachandran coloured by chi
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(phi_X0, psi_X0, c="grey", s=2, alpha=0.2, rasterized=True)
sc = ax.scatter(phi_mep, psi_mep, c=chi_mep, cmap="inferno",
                s=8, alpha=0.95, vmin=0, vmax=1, zorder=3, rasterized=True)
plt.colorbar(sc, ax=ax, label=r"$\chi$")
ax.scatter([float(phi(x0_ts))], [float(psi(x0_ts))],
           c="cyan", s=60, zorder=5, label="TS seed")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(r"$\chi$-MEP (discrete Koopman seeds)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mep_ramachandran.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. χ-MFEP at finite temperature
# ---------------------------------------------------------------------------
print("\n--- Building constrained χ-MFEP ---")

# Prepare TS seed: equilibrate → minimize on levelset → equilibrate again.
TS_EQUIL_STEPS = int(5.0 / DT)   # 5 ps each phase
chi_ts = ts_chi[j]
"""
print(f"TS prep phase 1/3: 5 ps equilibration on χ={chi_ts:.4f} …")
res_eq1 = sample_levelset_projected(sim_mep, f_NN, feat_torch, x0_ts, chi_ts, TS_EQUIL_STEPS)
_, x_eq1 = select_medoid(res_eq1["positions"].reshape(TS_EQUIL_STEPS, -1, 3))
x_eq1 = x_eq1.flatten()
"""
print(f"TS prep phase 2/3: energy minimization on levelset …")
x_min = energy_min_on_levelset(
    f_NN, feat_torch, _openmm_potential, x0_ts, chi_ts, grad_fn=_openmm_grad
)
"""
print(f"TS prep phase 3/3: 5 ps equilibration from minimized structure …")
res_eq2 = sample_levelset_projected(sim_mep, f_NN, feat_torch, x_min, chi_ts, TS_EQUIL_STEPS)
_, x0_ts_relaxed = select_medoid(res_eq2["positions"].reshape(TS_EQUIL_STEPS, -1, 3))
x0_ts_relaxed = x0_ts_relaxed.flatten()
print(f"TS seed ready: χ = {_chi_val(f_NN, feat_torch, x0_ts_relaxed):.4f}")
"""

mep_projected = build_chi_mep_projected(
    sim_mep, f_NN, feat_torch, x0_ts_relaxed,
    steps              = MEP_STEPS,
    steps_per_levelset = STEPS_PER_LS,
    burnin             = BURNIN_PER_LS,
    time_breakdown     =True
)


np.save(os.path.join(SCRATCH_DIR, "mep_projected_images.npy"),
        np.array(mep_projected["images"]))
np.save(os.path.join(SCRATCH_DIR, "mep_projected_cv.npy"),
        np.array(mep_projected["cv_values"]))
print(f"Saved projected MEP: {len(mep_projected['images'])} images")

imgs_proj  = np.array(mep_projected["images"])
phi_pmep   = phi(imgs_proj)
psi_pmep   = psi(imgs_proj)
cv_pmep    = np.array(mep_projected["cv_values"])
F_free     = mep_projected["F_free"]
F_rigid    = mep_projected["F_rigid"]


GRID_SIZE=100
fes        = np.load(os.path.join(SCRATCH_DIR, "metad_fes.npy"))
s1_ax      = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)
s2_ax      = np.linspace(-np.pi, np.pi, GRID_SIZE, endpoint=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cv_pmep, F_rigid, "k--", lw=1.2, label=r"$F_\mathrm{rigid}$")
ax.plot(cv_pmep, F_free,  "C0",  lw=2,   label=r"$F_\mathrm{free}$ (+ Fixman)")
ax.set_xlabel(r"$\chi$")
ax.set_ylabel(r"$F(\chi)$ [kJ/mol]")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fes_chi.png"), dpi=200)
plt.close(fig)





fig, ax = plt.subplots(figsize=(7, 6))
im = ax.contourf(s1_ax, s2_ax, fes, levels=30, cmap="RdYlBu_r")
plt.colorbar(im, ax=ax, label=r"$F(\phi, \psi)$ [kJ/mol]")
ax.scatter(phi_pmep, psi_pmep, color="#2d2d2d", linewidth=1.5, zorder=5, label="χ-MFEP")
ax.scatter([float(phi(x_min))], [float(psi(x_min))],
           c="cyan", s=60, zorder=5, label="TS seed")
ax.legend(loc="upper right")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title("Reconstructed FES with χ-MFEP (well-tempered MetaD)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metad_fes_mfep.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 8. Transition tube ensemble — 50 TS seeds, minimize only, no equilibration
# ---------------------------------------------------------------------------
N_TS_SEEDS = 50
rng_ts = np.random.default_rng(0)
seed_idx = rng_ts.choice(len(ts_frames), size=min(N_TS_SEEDS, len(ts_frames)), replace=False)

ensemble_meps = []
for k, idx in enumerate(seed_idx):
    x_ts_k   = ts_frames[idx]
    chi_ts_k = ts_chi[idx]
    print(f"\n[Ensemble {k+1}/{len(seed_idx)}]  χ={chi_ts_k:.4f}")
    x_min_k = energy_min_on_levelset(
        f_NN, feat_torch, _openmm_potential, x_ts_k, chi_ts_k, grad_fn=_openmm_grad
    )
    mep_k = build_chi_mep_projected(
        sim_mep, f_NN, feat_torch, x_min_k,
        steps              = MEP_STEPS,
        steps_per_levelset = STEPS_PER_LS,
        burnin             = BURNIN_PER_LS,
    )
    ensemble_meps.append(mep_k)

np.save(os.path.join(SCRATCH_DIR, "ensemble_meps.npy"), ensemble_meps, allow_pickle=True)

# 8a. All paths on Ramachandran
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(phi_X0, psi_X0, c="grey", s=2, alpha=0.15, rasterized=True)
for mep_k in ensemble_meps:
    imgs_k = np.array(mep_k["images"])
    cv_k   = np.array(mep_k["cv_values"])
    order  = np.argsort(cv_k)
    ax.plot(_phi(imgs_k[order]), _psi(imgs_k[order]),
            lw=0.6, alpha=0.4, color="C0", zorder=3)
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(rf"Transition tube ensemble ($N={len(ensemble_meps)}$)")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ensemble_ramachandran.png"), dpi=200)
plt.close(fig)

# 8b. All FES profiles
fig, ax = plt.subplots(figsize=(6, 4))
for mep_k in ensemble_meps:
    ax.plot(mep_k["cv_values"], mep_k["F_free"], lw=0.8, alpha=0.4, color="C0")
# mean profile
cv_common = np.linspace(0, 1, 200)
F_interp = np.array([
    np.interp(cv_common, np.array(m["cv_values"]), m["F_free"])
    for m in ensemble_meps
])
ax.plot(cv_common, F_interp.mean(axis=0), "k", lw=2, label="mean")
ax.fill_between(cv_common,
                F_interp.mean(0) - F_interp.std(0),
                F_interp.mean(0) + F_interp.std(0),
                alpha=0.15, color="k", label=r"$\pm 1\sigma$")
ax.set_xlabel(r"$\chi$")
ax.set_ylabel(r"$F(\chi)$ [kJ/mol]")
ax.set_title("Free energy ensemble")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ensemble_fes.png"), dpi=200)
plt.close(fig)

print(f"\nDone. Outputs in {os.path.abspath(OUT_DIR)}/")