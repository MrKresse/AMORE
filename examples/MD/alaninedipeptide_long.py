"""
ISOKANN on alanine dipeptide — long unbiased trajectories + AMORE-MD.

Python port of ISOKANN.jl/scripts/AMORE/alaninedipeptide_long.jl

Loads pre-computed JLD2 files from /scratch/htc/jkresse/ad/:
    500ns_{i}.jld2       → subtraj  (10000, 66)  sub-sampled coords (every 100th frame)
    500ns_phi_{i}.jld2   → phi      (1000000,)   phi of full saved trajectory
    500ns_psi_{i}.jld2   → psi      (1000000,)   psi of full saved trajectory

Workflow (mirroring the Julia script)
--------------------------------------
1. Load 10 replica trajectories + phi/psi from JLD2.
2. Build Koopman pairs per replica (lag=10 saved frames ≈ 5 ps).
3. Merge all replica data; featurise with 231 pairwise distances.
4. Train chi via the power method (mirrors run!(iso, 2000)).
5. Plot loss + Ramachandran coloured by chi.
6. AMORE-MD loop (10 iterations):
   a. Find transition-state frames (chi ≈ 0.5).
   b. Compute chi-MEP from one random TS frame.
   c. Propagate MEP frames one lag → new Koopman pairs; merge dataset.
   d. Re-train (mirrors run!(iso, 10000)).
7. Final Ramachandran + all MEP iterations; convergence plot.
"""

import os
import sys
import random

import h5py
import numpy as np
import torch as pt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# MoKiTo power method
# ---------------------------------------------------------------------------
# Set MOKITO_ROOT to your local MoKiTo checkout, or export MOKITO_ROOT=/path/to/MoKiTo
MOKITO_ROOT = os.environ.get("MOKITO_ROOT", "")
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
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pt.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Config  (mirrors alaninedipeptide_long.jl)
# ---------------------------------------------------------------------------
DATA_DIR    = os.environ.get("DATA_DIR", ".")  # set to directory with JLD2 input files
SCRATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_DIR     = "alaninedipeptide_long_out"
os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(OUT_DIR,     exist_ok=True)

N_REPLICAS = 10
LAG        = 10      # Koopman lag in saved frames  (10 × 250 steps × 2e-3 ps = 5 ps)

# Features: 231 pairwise distances for 22 heavy atoms
N_ATOMS = 22
PAIRS   = [(i, j) for i in range(N_ATOMS) for j in range(i + 1, N_ATOMS)]
N_FEATS = len(PAIRS)   # 231

# Network: pairnet  [231, 37, 6, 1]
NODES = pairnet_nodes(N_FEATS)

# Training  (Iso(data, opt=NesterovRegularized(), minibatch=10000); run!(iso, 2000))
NITERS_INIT = 2000
NEPOCHS     = 1
LR          = 1e-3
WD          = 1e-3     # NesterovRegularized default reg=1e-4
BS          = 10000     # minibatch=10000
PATIENCE    = 2000 #no early stopping
TOL         = 1e-10

# AMORE-MD loop
N_AMORE         = 10
MEP_STEPS       = 100
MEP_STEPSIZE    = 1 / MEP_STEPS
MEP_ENERGY_TOL  = 1e-6
MEP_MAX_ITER    = 500
CHI_LO, CHI_HI = 0.49, 0.51
NITERS_AMORE    = 1_000   # run!(iso, 10000) per AMORE step

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"device : {device}")
print(f"pairnet: {NODES}")

# ---------------------------------------------------------------------------
# Helper: load a single variable from a JLD2 (HDF5) file
# h5py reads Julia (column-major) arrays; for 2-D matrices the axes are
# already (n_frames, dim) because Julia saved (dim, n_frames) in col-major,
# which HDF5.jl writes as shape (n_frames, dim) in the HDF5 file.
# ---------------------------------------------------------------------------
def load_jld2(path, key):
    with h5py.File(path, "r") as f:
        return f[key][:]


# ---------------------------------------------------------------------------
# 1. Load all replicas from JLD2, build merged Koopman pairs
#    (mirrors the second for-loop in the Julia script)
# ---------------------------------------------------------------------------
phi_all  = None   # concatenated phi from all full saved trajectories
psi_all  = None
X0_all   = None   # merged Koopman X0  (n_pairs_total, dim)
Xtau_all = None

print("Loading replica trajectories from JLD2 …")
for i in range(1, N_REPLICAS + 1):
    subtraj = load_jld2(os.path.join(DATA_DIR, f"500ns_{i}.jld2"),     "subtraj")  # (10000, 66)
    phi_rep = load_jld2(os.path.join(DATA_DIR, f"500ns_phi_{i}.jld2"), "phi")      # (1000000,)
    psi_rep = load_jld2(os.path.join(DATA_DIR, f"500ns_psi_{i}.jld2"), "psi")      # (1000000,)

    print(f"  replica {i}: subtraj={subtraj.shape}, phi={phi_rep.shape}")

    # Koopman pairs from sub-sampled trajectory (lag=10 frames)
    X0_rep, Xtau_rep = koopman_pairs_lag(subtraj, lag=LAG)

    phi_all  = phi_rep  if phi_all  is None else np.concatenate([phi_all,  phi_rep])
    psi_all  = psi_rep  if psi_all  is None else np.concatenate([psi_all,  psi_rep])
    X0_all   = X0_rep   if X0_all   is None else np.concatenate([X0_all,   X0_rep],   axis=0)
    Xtau_all = Xtau_rep if Xtau_all is None else np.concatenate([Xtau_all, Xtau_rep], axis=0)

print(f"Total Koopman pairs : X0={X0_all.shape}, Xtau={Xtau_all.shape}")
print(f"Total phi/psi frames: {phi_all.shape[0]}")

# ---------------------------------------------------------------------------
# 2. Featurise  (pairwise distances, 231 features)
# ---------------------------------------------------------------------------
featurizer = make_featurizer(PAIRS)

def featurize(coords_np):
    """(N, 66) numpy → (N, 231) float32 torch tensor on device."""
    return featurizer(pt.from_numpy(coords_np.astype(np.float32)).to(device))

X0_t   = featurize(X0_all)
Xtau_t = featurize(Xtau_all)
print(f"Feature tensors: X0={X0_t.shape}, Xtau={Xtau_t.shape}")

# ---------------------------------------------------------------------------
# 3. Initial training  (run!(iso, 2000))
# ---------------------------------------------------------------------------
class NeuralNetworkLN(NeuralNetwork):
    """NeuralNetwork with a single LayerNorm applied to the input."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-register submodules in forward-pass order; drop unused activation2
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

pt.save(f_NN.state_dict(), os.path.join(SCRATCH_DIR, "f_NN_long.pt"))

# ---------------------------------------------------------------------------
# 4. Plots: loss + Ramachandran  (mirrors plot_training + scatter_ramachandran)
# ---------------------------------------------------------------------------
ticks  = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

fig, ax = plt.subplots()
ax.plot(np.asarray(train_loss, dtype=float), label="train")
ax.plot(np.asarray(val_loss,   dtype=float), label="val")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE loss")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_long.pdf"))
plt.close(fig)

# evaluate chi on training X0
f_NN.eval()
with pt.no_grad():
    chi_X0 = f_NN(X0_t).cpu().numpy().squeeze()   # (N_pairs,)

# for Ramachandran, sub-sample X0 and the corresponding phi/psi
# phi/psi of X0: X0 comes from subtraj[:-lag], so index into subtraj
# simplest: just compute phi/psi directly from X0_all
phi_X0 = phi(X0_all)
psi_X0 = psi(X0_all)

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(phi_X0[::10], psi_X0[::10], c=chi_X0[::10],
                cmap="inferno", s=2, alpha=0.6, rasterized=True)
plt.colorbar(sc, ax=ax, label=r"$\chi$")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
ax.set_title(r"Ramachandran — $\chi$ after initial training")
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramachandran_long_init.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. AMORE-MD loop  (mirrors the for i in 1:10 ... end block)
# ---------------------------------------------------------------------------
# For new MD propagation (MEP frame → Xtau) we need the simulator
sim = OpenMMSimulation(steps=1, dt=2e-3)   # steps=1 placeholder; we call koopman_pairs directly

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

# featurizer wrapper expected by mep module (torch input)
def feat_torch(x_t):
    return featurizer(x_t)

# background training traj for plotting  (mirrors: traj = iso.data.coords[1])
traj_bg  = X0_all
phi_bg   = phi_X0
psi_bg   = psi_X0

xss = []   # collect MEP paths across iterations

for amore_iter in range(1, N_AMORE + 1):
    print(f"\n=== AMORE iteration {amore_iter}/{N_AMORE} ===")

    # a) Transition-state frames  (transition_state(cpu(iso), 0.49, 0.51))
    ts_frames, ts_chi = transition_state(
        f_NN, feat_torch, traj_bg, chi_lo=CHI_LO, chi_hi=CHI_HI
    )
    print(f"  TS frames: {len(ts_frames)}  (chi ∈ [{CHI_LO:.2f}, {CHI_HI:.2f}])")

    if len(ts_frames) == 0:
        print("  No TS frames — widening window to [0.4, 0.6]")
        ts_frames, ts_chi = transition_state(
            f_NN, feat_torch, traj_bg, chi_lo=0.4, chi_hi=0.6
        )
        if len(ts_frames) == 0:
            print("  Still none — skipping.")
            continue

    # b) Pick one random TS frame; compute chi-MEP
    #    (reactionpath_minimum(iso, x; steps=100, f_reltol=1e-5, alphaguess=1e-5,
    #                          iterations=250, algorithm=ConjugateGradient))
    j     = random.randrange(len(ts_frames))
    x0_ts = ts_frames[j]
    print(f"  Starting MEP from chi = {ts_chi[j]:.4f}")

    xs_mep = reaction_path_minimum(
        f_NN, feat_torch, x0_ts,
        steps           = MEP_STEPS,
        stepsize        = MEP_STEPSIZE,
        potential_fn    = _openmm_potential,
        grad_fn         = _openmm_grad,
        energy_tol      = MEP_ENERGY_TOL,
        energy_max_iter = MEP_MAX_ITER,
    )
    xss.append(xs_mep)
    np.save(os.path.join(SCRATCH_DIR, f"mep_{amore_iter}.npy"), xs_mep)

    # c) Propagate each MEP frame one lag → new Koopman pairs
    #    (SimulationData(sim, xs, 1))
    print(f"  Propagating {len(xs_mep)} MEP frames …")
    X0_mep, Xtau_mep = sim.koopman_pairs(xs_mep)

    X0_all   = np.concatenate([X0_all,   X0_mep],   axis=0)
    Xtau_all = np.concatenate([Xtau_all, Xtau_mep], axis=0)

    X0_t   = featurize(X0_all)
    Xtau_t = featurize(Xtau_all)
    print(f"  Dataset: {X0_all.shape[0]} pairs total")

    # update background traj for transition-state search
    traj_bg = X0_all
    phi_bg  = phi(X0_all)
    psi_bg  = psi(X0_all)

    # d) Re-train  (run!(iso, 10000))
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

    # Per-iteration Ramachandran  (mirrors the scatter(phi,psi) inside the loop)
    phi_ts  = float(phi(x0_ts))
    psi_ts  = float(psi(x0_ts))
    phi_mep = phi(xs_mep)
    psi_mep = psi(xs_mep)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(phi_bg[::10], psi_bg[::10], c="grey", s=2, alpha=0.25, rasterized=True,
               label="Stationary distribution")
    ax.scatter(phi_mep, psi_mep, c="blue",  s=3, alpha=0.9, rasterized=True,
               label=r"$\chi$-MEP states")
    ax.scatter([phi_ts], [psi_ts], c="cyan", zorder=5, s=60, label="Initial state")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\phi$ [rad]"); ax.set_ylabel(r"$\psi$ [rad]")
    ax.set_title(f"AMORE iteration {amore_iter}")
    ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
    ax.legend(fontsize=7, loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"ramachandran_amore_{amore_iter}.png"), dpi=200)
    plt.close(fig)

pt.save(f_NN.state_dict(), os.path.join(SCRATCH_DIR, "f_NN_long_enhanced.pt"))
np.save(os.path.join(SCRATCH_DIR, "X0_long_enhanced.npy"),   X0_all)
np.save(os.path.join(SCRATCH_DIR, "Xtau_long_enhanced.npy"), Xtau_all)

# ---------------------------------------------------------------------------
# 6. Final plot: all MEP iterations overlaid  (mirrors ad_long_iters.png)
# ---------------------------------------------------------------------------
highlight_iters = [1, 5, 10, 15]
highlight_cols  = ["#D55E00", "#E69F00", "#009E73", "#0000FF"]   # match Julia colours
highlight_ms    = [3, 3, 4, 5]
highlight_alpha = [0.55, 0.70, 0.85, 1.0]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(phi_bg[::10], psi_bg[::10], c="grey", s=2, alpha=0.25, rasterized=True,
           label="Unbiased MD samples")

for it, col, ms, ma in zip(highlight_iters, highlight_cols, highlight_ms, highlight_alpha):
    if it - 1 >= len(xss):
        continue
    xs_it  = xss[it - 1]
    ax.scatter(phi(xs_it), psi(xs_it), color=col, s=ms, alpha=ma, rasterized=True,
               label=rf"$\chi$-MEP (iter {it})")

ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel(r"$\phi$ [rad]", fontsize=14); ax.set_ylabel(r"$\psi$ [rad]", fontsize=14)
ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
ax.legend(fontsize=10, loc="upper right")
ax.set_title(r"Ramachandran + $\chi$-MEP iterations")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramachandran_long_iters.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Convergence plot  (mirrors ad_long_convergence.png)
#    RMS and max angular displacement in (phi, psi) between consecutive MEPs.
# ---------------------------------------------------------------------------

def _wrap_delta(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def amore_convergence_phi_psi(paths):
    rms_shifts, max_jumps = [], []
    for a, b in zip(paths[:-1], paths[1:]):
        n     = min(len(a), len(b))
        dphi  = _wrap_delta(phi(b[:n]) - phi(a[:n]))
        dpsi  = _wrap_delta(psi(b[:n]) - psi(a[:n]))
        dist  = np.sqrt(dphi**2 + dpsi**2)
        rms_shifts.append(float(np.sqrt(np.mean(dist**2))))
        max_jumps.append(float(np.max(dist)))
    return np.array(rms_shifts), np.array(max_jumps)

if len(xss) >= 2:
    rms_shift, max_jump = amore_convergence_phi_psi(xss)
    iters = np.arange(2, len(rms_shift) + 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, rms_shift, marker="o",  lw=2, label="RMS path shift")
    ax.plot(iters, max_jump,  marker="D", lw=2, ls="--", label="Maximum point displacement")
    ax.set_xlabel("AMORE-MD iteration", fontsize=13)
    ax.set_ylabel(r"Displacement in $(\phi, \psi)$ [rad]", fontsize=13)
    ax.legend(fontsize=11); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence_long.png"), dpi=200)
    plt.close(fig)

print(f"\nDone. Outputs in: {os.path.abspath(OUT_DIR)}/")
