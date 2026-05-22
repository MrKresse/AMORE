"""
Generate Koopman pairs for the 2D triple-well benchmark.

System
------
Same triple-well as examples/MD/triple_well_isokann.py:
  V(x,y) = -5 Σ exp(-|x-c_i|²) + 0.3|x|²
  Wells at A=(-1.2,0), B=(1.2,0), C=(0,1.5)
  σ=1.2, dt=5e-4, lagtime=0.3

Data layout
-----------
  anchors  : (N_A × N_B, 2)         — regular grid on x,y
  bursts   : (N_A × N_B, N_K, 2)    — N_K Langevin endpoints per anchor
  x0       : (N_A × N_B × N_K, 2)   — flattened pairs (anchor repeated)
  x1       : (N_A × N_B × N_K, 2)   — flattened burst endpoints
  kchi_avg : computed on the fly by averaging chi over the burst dimension

Train/test split — contiguous patches
  Grid split into PATCH_N×PATCH_N = 16 patches.
  4 random patches held out per seed.  5 seeds.
  patch_splits.npy  →  shape (5, N_anchors) with 0=train, 1=test
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from amore.sims import LangevinSimulator

OUTDIR   = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTDIR, exist_ok=True)

# ── Potential ──────────────────────────────────────────────────────────────────
WELLS       = np.array([[-1.2, 0.0], [1.2, 0.0], [0.0, 1.5]])
WELL_DEPTH  = 5.0
WELL_WIDTH  = 0.5
WALL_K      = 0.3

def potential(x: np.ndarray) -> float:
    V = sum(-WELL_DEPTH * np.exp(-np.sum((x - c)**2) / (2 * WELL_WIDTH))
            for c in WELLS)
    V += WALL_K * np.sum(x**2)
    return float(V)

def gradient(x: np.ndarray) -> np.ndarray:
    g = np.zeros(2)
    for c in WELLS:
        d = x - c
        g += WELL_DEPTH / WELL_WIDTH * d * np.exp(-np.sum(d**2) / (2 * WELL_WIDTH))
    g += 2 * WALL_K * x
    return g

# ── Config ─────────────────────────────────────────────────────────────────────
GRID_NX   = 40        # grid points per axis → 1600 anchors
GRID_NY   = 40
GRID_LO   = np.array([-2.0, -0.8])
GRID_HI   = np.array([ 2.0,  2.3])
E_MAX     = 10.0      # discard anchors above this energy

DT        = 5e-4
SIGMA     = 1.2
LAGTIME   = 0.30
N_BURSTS  = 20        # Koopman bursts per anchor

PATCH_N   = 4         # 4×4 = 16 patches
N_HOLD    = 4         # patches held out per seed
N_SEEDS   = 5

SEED = 42
rng  = np.random.default_rng(SEED)

# ── Build Langevin simulator ───────────────────────────────────────────────────
sim = LangevinSimulator(
    potential_fn = potential,
    grad_fn      = gradient,
    dim          = 2,
    sigma        = SIGMA,
    dt           = DT,
    lagtime      = LAGTIME,
    support      = np.stack([GRID_LO, GRID_HI]),
)

n_steps = max(1, int(round(LAGTIME / DT)))   # 600 steps per burst

# ── Anchor grid ────────────────────────────────────────────────────────────────
xs = np.linspace(GRID_LO[0], GRID_HI[0], GRID_NX)
ys = np.linspace(GRID_LO[1], GRID_HI[1], GRID_NY)
XX, YY = np.meshgrid(xs, ys)
grid_all = np.column_stack([XX.ravel(), YY.ravel()])   # (1600, 2)

# Filter high-energy anchors
E_grid = np.array([potential(x) for x in grid_all])
mask   = E_grid < E_MAX
grid   = grid_all[mask]
N_ANC  = len(grid)
print(f"Anchors below E={E_MAX}: {N_ANC} / {len(grid_all)}")

# ── Burst propagation ─────────────────────────────────────────────────────────
print(f"Propagating {N_ANC} × {N_BURSTS} bursts ({n_steps} steps each, σ={SIGMA}) …")
bursts = np.empty((N_ANC, N_BURSTS, 2), dtype=np.float32)
for i, x0 in enumerate(grid):
    for k in range(N_BURSTS):
        xi = x0.copy()
        for _ in range(n_steps):
            xi = sim.step(xi, rng)
        bursts[i, k] = xi
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_ANC}")

print("Done propagating.")

# Flatten to (x0, x1) pairs
x0_flat = np.repeat(grid, N_BURSTS, axis=0)                  # (N_ANC*N_K, 2)
x1_flat = bursts.reshape(-1, 2)                               # (N_ANC*N_K, 2)

# ── Patch-based train/test splits ─────────────────────────────────────────────
# Assign each anchor to a (px, py) patch index
px = np.floor((grid[:, 0] - GRID_LO[0]) / (GRID_HI[0] - GRID_LO[0]) * PATCH_N).astype(int)
py = np.floor((grid[:, 1] - GRID_LO[1]) / (GRID_HI[1] - GRID_LO[1]) * PATCH_N).astype(int)
px = np.clip(px, 0, PATCH_N - 1)
py = np.clip(py, 0, PATCH_N - 1)
patch_id = px * PATCH_N + py   # 0 … PATCH_N²-1

all_patches = list(range(PATCH_N * PATCH_N))
split_rng   = np.random.default_rng(SEED + 100)

patch_splits = np.zeros((N_SEEDS, N_ANC), dtype=np.int8)   # 0=train, 1=test
for s in range(N_SEEDS):
    hold = split_rng.choice(all_patches, size=N_HOLD, replace=False)
    for h in hold:
        patch_splits[s, patch_id == h] = 1
    n_test  = (patch_splits[s] == 1).sum()
    n_train = (patch_splits[s] == 0).sum()
    print(f"  seed {s}: {n_train} train / {n_test} test anchors")

# ── Save ───────────────────────────────────────────────────────────────────────
np.savez(
    os.path.join(OUTDIR, "triple_well_koopman.npz"),
    anchors      = grid.astype(np.float32),   # (N_ANC, 2)
    bursts       = bursts,                     # (N_ANC, N_K, 2)
    x0           = x0_flat.astype(np.float32),
    x1           = x1_flat.astype(np.float32),
    patch_splits = patch_splits,               # (5, N_ANC)
    wells        = WELLS.astype(np.float32),
    grid_lo      = GRID_LO.astype(np.float32),
    grid_hi      = GRID_HI.astype(np.float32),
    n_bursts     = np.array([N_BURSTS]),
    lagtime      = np.array([LAGTIME]),
    sigma        = np.array([SIGMA]),
)
print(f"\nSaved: data/triple_well_koopman.npz")
print(f"  anchors={grid.shape}  bursts={bursts.shape}  pairs={x0_flat.shape}")
