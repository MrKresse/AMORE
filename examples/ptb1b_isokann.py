"""
ISOKANN run for PTB1B replica 0.

Workflow
--------
1. Load trajectory (DCD + PDB) and compute pairwise-distance features.
2. Normalise features and align lag pairs (X0, Xtau).
3. Train the chi network via the power method.
4. Save the model, losses, and chi values.
5. Compute chi sensitivity (gradient norms per atom).
6. Pick representative structures and save PDB files.
"""

import sys
import os
import pickle

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import MDAnalysis as mda

# ---------------------------------------------------------------------------
# MoKiTo: local import only (not a package dependency)
# ---------------------------------------------------------------------------
MOKITO_ROOT = "/home/numerik/jkresse/code/MoKiTo"
sys.path.insert(0, MOKITO_ROOT)

from src.useful_functions import read_dirs_paths
from src.isokann.modules3 import NeuralNetwork, power_method, scale_and_shift

# ---------------------------------------------------------------------------
# amore package
# ---------------------------------------------------------------------------
import amore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
pt.manual_seed(SEED)

MOKITO_EXAMPLE = "/home/numerik/jkresse/code/MoKiTo/examples/ptb1b"
OUT_DIR        = "/scratch/htc/jkresse/out_isokann/"
DCD_FILE       = "/scratch/htc/fsafarov/2cm2_simulation/md2/output/trajectories/openmm_files/trajectory_water_combined6.dcd"
PDB_FILE       = "/scratch/htc/fsafarov/2cm2_simulation/md2/output/trajectories/openmm_files/frame_3203.pdb"
HYPERPARAMS    = os.path.join(MOKITO_EXAMPLE, "hyperparameter_files/hyperparams_7.pkl")
DIST_INDICES   = os.path.join(OUT_DIR, "distance_indices.npy")
DIR_PATHS_TXT  = os.path.join(MOKITO_EXAMPLE, "dir_paths-checkpoint.txt")

LAG    = 9    # lag in frames
STRIDE = 1
NITERS = 400
TOL    = 1e-6

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"device: {device}")

# ---------------------------------------------------------------------------
# Load hyperparameters
# ---------------------------------------------------------------------------
with open(HYPERPARAMS, "rb") as f:
    hp = pickle.load(f)

# ---------------------------------------------------------------------------
# Build feature pairs from distance index file
# ---------------------------------------------------------------------------
distance_indices = np.load(DIST_INDICES, allow_pickle=True)
pairs = np.array([(i, j) for (i, j, _) in distance_indices[0]], dtype=np.int64)
del distance_indices

# ---------------------------------------------------------------------------
# Load trajectory and compute features
# ---------------------------------------------------------------------------
u = mda.Universe(PDB_FILE, DCD_FILE)
print(f"frames: {len(u.trajectory)},  atoms: {u.atoms.n_atoms}")

n_frames = len(u.trajectory)
n_atoms  = u.atoms.n_atoms
traj_flat = np.empty((n_frames, 3 * n_atoms), dtype=np.float32)
for fi, _ in enumerate(u.trajectory):
    traj_flat[fi] = u.atoms.positions.reshape(-1)
del u

x_torch = pt.from_numpy(traj_flat).to(device)
del traj_flat

D = amore.features_pairs(pairs, x_torch)   # (n_frames, n_feat)

# ---------------------------------------------------------------------------
# Lag-pair alignment and per-sample normalisation
# ---------------------------------------------------------------------------
D0 = D[:-LAG:STRIDE]
Dt = D[LAG::STRIDE]

def _normalise(X):
    mu  = X.mean(dim=-1, keepdim=True)
    std = X.std(dim=-1, keepdim=True)
    return (X - mu).abs() / (std + 1e-8)

D0_n = _normalise(D0)
Dt_n = _normalise(Dt)

# ---------------------------------------------------------------------------
# Build and train the chi network
# ---------------------------------------------------------------------------
f_NN = NeuralNetwork(
    Nodes=np.asarray(hp["nodes"]),
    activation_function=hp["act_fun"],
).to(device)

train_loss, val_loss, best_loss, convergence = power_method(
    D0_n, Dt_n, f_NN, scale_and_shift,
    Niters     = NITERS,
    Nepochs    = hp["Nepochs"],
    tolerance  = TOL,
    lr         = hp["learning_rate"],
    wd         = hp["weight_decay"],
    batch_size = hp["batch_size"],
    patience   = hp["patience"],
    print_eta  = True,
    test_size  = 0.2,
    loss       = "full",
)

# ---------------------------------------------------------------------------
# Save model and training results
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
pt.save(f_NN.state_dict(), os.path.join(OUT_DIR, "f_NN.pt"))
np.save(os.path.join(OUT_DIR, "train_loss.npy"),  train_loss)
np.save(os.path.join(OUT_DIR, "val_loss.npy"),    val_loss)
np.save(os.path.join(OUT_DIR, "convergence.npy"), convergence)
np.save(os.path.join(OUT_DIR, "chi.npy"),         f_NN(D0).cpu().detach().numpy())

# ---------------------------------------------------------------------------
# Plot loss curves
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
# Analysis on CPU (gradient computation)
# ---------------------------------------------------------------------------
f_NN = f_NN.to("cpu")
x_torch = x_torch.to("cpu")

featurizer = amore.make_featurizer(pairs)

amore.save_chi_histogram(f_NN, featurizer, x_torch,
                         out=os.path.join(OUT_DIR, "chi_histogram.png"))

_, avg_grad = amore.chi_sensitivity(f_NN, featurizer, x_torch)
amore.save_gradient(os.path.join(OUT_DIR, "mean_gradients.jld2"), avg_grad)

xs_bins,  _  = amore.pick_representative_xs(f_NN, featurizer, x_torch, nbins=20)
xs_bound, _  = amore.pick_representative_xs(f_NN, featurizer, x_torch,
                                             nbins=20, chi_range=(20.0, 25.0))

amore.save_pdb_as_frames(xs_bins,  PDB_FILE, os.path.join(OUT_DIR, "states_from_bins.pdb"))
amore.save_pdb_as_frames(xs_bound, PDB_FILE, os.path.join(OUT_DIR, "bound_states_from_bins.pdb"))
