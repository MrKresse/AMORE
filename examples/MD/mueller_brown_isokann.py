"""
ISOKANN on the Mueller-Brown potential.

Workflow
--------
1. Generate (X0, X_tau) pairs by uniform Koopman sampling.
2. Train the chi network via the power method (MoKiTo).
3. Plot chi coloured on the training points and on a PES contour.
4. Compute the chi-MEP from the transition-state region.
"""

import sys
import os

import numpy as np
import torch as pt
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# MoKiTo: local import only (not a package dependency)
# ---------------------------------------------------------------------------
# Set MOKITO_ROOT to your local MoKiTo checkout, or export MOKITO_ROOT=/path/to/MoKiTo
MOKITO_ROOT = os.environ.get("MOKITO_ROOT", "")
sys.path.insert(0, MOKITO_ROOT)

from src.isokann.modules3 import NeuralNetwork, power_method, scale_and_shift

# ---------------------------------------------------------------------------
# amore
# ---------------------------------------------------------------------------
import amore
from amore.sims import MuellerBrown, mueller_brown_potential
from amore.sims.mueller_brown import gradient as mueller_brown_gradient
from amore.mep import reaction_path_minimum, transition_state

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
pt.manual_seed(SEED)

OUT_DIR = "mueller_brown_out"
SCRATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# Simulation / Koopman sampling
LAGTIME    = 1e-3     # lag time per Koopman pair  (matches CGEM/mueller_brown.jl)
SIGMA      = 5.0      # noise amplitude

# Network + training  (architecture matches ISOKANN.jl smallnet; hyperparams from CGEM/mueller_brown.jl)
NODES    = [2, 5, 10, 5, 1]  # smallnet: nin → 5 → 10 → 5 → nout
NITERS   = 250
NEPOCHS  = 200
LR       = 1e-3
WD       = 1e-3               # NesterovRegularized(1e-3, 1e-3)
BS       = 1000               # minibatch=1000
PATIENCE = 50
TOL      = 1e-6

# Uniform Koopman sampling
N_KOOPMAN = 1000   # extra (x0, x_tau) pairs drawn uniformly from PES support

# chi-MEP  (hyperparameters from CGEM/mueller_brown.jl)
MEP_STEPS      = 100    # total path images (split by chi(x0))
MEP_STEPSIZE   = 1 / MEP_STEPS   # = 0.01; Julia uses stepsize=1/steps
MEP_ENERGY_TOL = 1e-5   # SLSQP convergence tolerance
MEP_MAX_ITER   = 100    # SLSQP max iterations per projection
CHI_LO, CHI_HI = 0.45, 0.55  # transition-state chi window

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"device: {device}")

sim = MuellerBrown(dt=1e-4, lagtime=LAGTIME, sigma=SIGMA)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Lagged pairs  (uniform Koopman sampling for full PES coverage)
# ---------------------------------------------------------------------------
print(f"Generating {N_KOOPMAN} uniform Koopman pairs …")
X0, Xtau = sim.koopman_pairs(N_KOOPMAN, rng=rng)
print(f"Koopman pairs: X0={X0.shape}, Xtau={Xtau.shape}")

X0_t   = pt.from_numpy(X0).float().to(device)
Xtau_t = pt.from_numpy(Xtau).float().to(device)

# ---------------------------------------------------------------------------
# 3. Train chi network
# ---------------------------------------------------------------------------
f_NN = NeuralNetwork(
    Nodes=np.array(NODES),
    activation_function="sigmoid",
).to(device)

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
    test_size  = 0.2,
    loss       = "full",
)

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
pt.save(f_NN.state_dict(), os.path.join(SCRATCH_DIR, "f_NN.pt"))
np.save(os.path.join(SCRATCH_DIR, "train_loss.npy"),  np.asarray(train_loss, dtype=float))
np.save(os.path.join(SCRATCH_DIR, "val_loss.npy"),    np.asarray(val_loss,   dtype=float))
np.save(os.path.join(SCRATCH_DIR, "X0.npy"),          X0)

# ---------------------------------------------------------------------------
# 5. Evaluate chi on training points
# ---------------------------------------------------------------------------
f_NN.eval()
with pt.no_grad():
    chi = f_NN(X0_t).cpu().numpy().squeeze()   # (N_KOOPMAN,)

# ---------------------------------------------------------------------------
# 6. Plot: PES  |  chi-coloured X0 samples
# ---------------------------------------------------------------------------
stride_raw = max(1, len(X0) // 10000)

xg = np.linspace(-1.5, 1.0, 300)
yg = np.linspace(-0.5, 2.0, 300)
XX, YY = np.meshgrid(xg, yg)
ZZ = np.clip(
    np.vectorize(lambda x, y: mueller_brown_potential([x, y]))(XX, YY),
    None, 200,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cf = axes[0].contourf(XX, YY, ZZ, levels=30, cmap="RdYlBu_r")
plt.colorbar(cf, ax=axes[0], label="Energy")
axes[0].set_xlim(-1.5, 1.0)
axes[0].set_ylim(-0.5, 2.0)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Mueller-Brown PES")

sc = axes[1].scatter(X0[::stride_raw, 0], X0[::stride_raw, 1],
                     c=chi[::stride_raw], cmap="inferno",
                     s=4, alpha=0.6, rasterized=True)
plt.colorbar(sc, ax=axes[1], label=r"$\chi$")
axes[1].set_xlim(-1.5, 1.0)
axes[1].set_ylim(-0.5, 2.0)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title(r"Koopman $X_0$ coloured by $\chi$")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pes_chi.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# 7. Plot: loss curves
# ---------------------------------------------------------------------------
plt.plot(np.asarray(train_loss, dtype=float), label="train")
plt.plot(np.asarray(val_loss,   dtype=float), label="validation")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves.pdf"))
plt.close()



# ---------------------------------------------------------------------------
# 8. chi-MEP from the transition-state region
# ---------------------------------------------------------------------------
f_NN = NeuralNetwork(Nodes=np.array(NODES), activation_function="sigmoid").to(device)
f_NN.load_state_dict(pt.load(os.path.join(SCRATCH_DIR, "f_NN.pt"), map_location=device))
f_NN.eval()

X0 = np.load(os.path.join(SCRATCH_DIR, "X0.npy"))

# For Mueller-Brown, coords ARE features — identity featurizer.
featurizer = lambda x: x   # noqa: E731

ts_frames, ts_chi = transition_state(f_NN, featurizer, X0,
                                     chi_lo=CHI_LO, chi_hi=CHI_HI)

print(f"Transition-state frames found: {len(ts_frames)}  "
      f"(chi ∈ [{CHI_LO}, {CHI_HI}])")

if len(ts_frames) == 0:
    print("No transition-state frames — skipping MEP.  "
          "Try widening CHI_LO/CHI_HI or running a longer trajectory.")
else:
    # Pick the frame closest to chi = 0.5
    best_idx = np.argmin(np.abs(ts_chi - 0.5))
    x0_ts = ts_frames[best_idx]
    print(f"Starting MEP from chi = {ts_chi[best_idx]:.4f}")

    path = reaction_path_minimum(
        f_NN, featurizer, x0_ts,
        steps        = MEP_STEPS,
        stepsize     = MEP_STEPSIZE,
        potential_fn = mueller_brown_potential,
        grad_fn      = mueller_brown_gradient,
        energy_tol   = MEP_ENERGY_TOL,
        energy_max_iter = MEP_MAX_ITER,
    )
    np.save(os.path.join(OUT_DIR, "mep.npy"), path)
    print(f"MEP shape: {path.shape}")

    # Plot: PES with MEP overlaid
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(XX, YY, ZZ, levels=30, cmap="RdYlBu_r")
    plt.colorbar(cf, ax=ax, label="Energy")
    ax.scatter(X0[::stride_raw, 0], X0[::stride_raw, 1],
               c=chi[::stride_raw], cmap="inferno",
               s=4, alpha=0.3, rasterized=True)
    ax.plot(path[:, 0], path[:, 1], color="white", lw=2, label="chi-MEP")
    ax.scatter([x0_ts[0]], [x0_ts[1]], color="cyan", zorder=5,
               s=60, label="transition state")
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mueller-Brown PES + chi-MEP")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mep.png"), dpi=200)
    plt.close()

print(f"\nDone. Outputs in: {os.path.abspath(OUT_DIR)}/")
