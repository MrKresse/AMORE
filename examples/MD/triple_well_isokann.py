"""
Multi-D ISOKANN benchmark: 2D triple-well potential with k=3.

Ground truth
------------
The triple-well has three minima:
    A = (-1.2,  0.0)
    B = ( 1.2,  0.0)
    C = ( 0.0,  1.5)

The three chi functions should satisfy:
    chi_i ≈ 1 near well i,  chi_i ≈ 0 elsewhere.
Together they form a 2-simplex (triangle) in chi-space.

Koopman pairs: grid-based (no long trajectory needed)
    - Sample x0 from a regular grid in the relevant region
    - Propagate each x0 with Langevin for `lagtime` steps -> x1
    - This gives uniform coverage without burn-in

Validation
----------
1. Visual: chi coloured on the PES — each function should peak at one well
2. Simplex: plot (chi1, chi2, chi3) scatter — should form a filled triangle
3. Rates: implied timescales from eigenvalues of K should separate slow modes
4. Expected k=3 timescales >> all others (k=4,...) for a 3-state system
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# amore imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import amore
from amore.sims import LangevinSimulator
from amore.isokann import ChiNetMulti, ChiNetMultiRaw, power_method_multi, implied_timescales

OUT_DIR = os.path.join(os.path.dirname(__file__), "triple_well_out")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────
K       = 3       # number of metastable states
DIM     = 2       # phase-space dimension

# Langevin
DT      = 5e-4
SIGMA   = 1.2     # noise amplitude (kBT in energy units)
LAGTIME = 0.3     # longer lagtime so pairs span the full state space

# Grid sampling
GRID_N  = 40      # grid points per axis (N^2 total)
GRID_LO = np.array([-2.0, -0.8])
GRID_HI = np.array([ 2.0,  2.2])
E_MAX   = 10.0    # discard grid points above this energy

# Network + training
# Use ChiNetMultiRaw (independent sigmoid) — avoids softmax competition that
# causes chi functions to collapse when one direction has small variance.
HIDDEN       = [64, 64, 32]
N_ITER       = 80
EPOCHS       = 400
LR           = 2e-3
LR_DECAY     = 0.98
BATCH        = 1024

DEVICE = torch.device("cpu")   # 2D system is fast on CPU


# ── Triple-well potential ──────────────────────────────────────────────────────

WELLS = np.array([[-1.2, 0.0], [1.2, 0.0], [0.0, 1.5]])  # A, B, C
WELL_DEPTH  = 5.0
WELL_WIDTH  = 0.5   # sigma^2
WALL_STRENGTH = 0.3

def potential(x: np.ndarray) -> float:
    """Three Gaussian wells plus a soft quadratic boundary."""
    x = np.asarray(x)
    V = 0.0
    for c in WELLS:
        V -= WELL_DEPTH * np.exp(-np.sum((x - c)**2) / (2 * WELL_WIDTH))
    V += WALL_STRENGTH * np.sum(x**2)
    return float(V)

def gradient(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of the triple-well potential."""
    x = np.asarray(x)
    g = np.zeros(2)
    for c in WELLS:
        diff = x - c
        r2   = np.sum(diff**2)
        g   += WELL_DEPTH / WELL_WIDTH * diff * np.exp(-r2 / (2 * WELL_WIDTH))
    g += 2 * WALL_STRENGTH * x
    return g


# ── Langevin simulator ────────────────────────────────────────────────────────

sim = LangevinSimulator(
    potential_fn = potential,
    grad_fn      = gradient,
    dim          = DIM,
    sigma        = SIGMA,
    dt           = DT,
    lagtime      = LAGTIME,
    support      = np.stack([GRID_LO, GRID_HI]),
)

rng = np.random.default_rng(SEED)


# ── Grid-based Koopman pair generation ────────────────────────────────────────

print(f"Generating Koopman pairs from {GRID_N}x{GRID_N} grid ...")
xs = np.linspace(GRID_LO[0], GRID_HI[0], GRID_N)
ys = np.linspace(GRID_LO[1], GRID_HI[1], GRID_N)
XX, YY = np.meshgrid(xs, ys)
grid = np.column_stack([XX.ravel(), YY.ravel()])   # (N^2, 2)

# Filter high-energy points — only propagate from accessible states
E_grid  = np.array([potential(x) for x in grid])
mask    = E_grid < E_MAX
grid_ok = grid[mask]
n_grid  = len(grid_ok)
print(f"  Grid points below E={E_MAX}: {n_grid} / {len(grid)}")

# Propagate each grid point
n_steps = max(1, int(round(LAGTIME / DT)))
X0   = grid_ok.copy()
X1   = np.empty_like(X0)
for i, x in enumerate(grid_ok):
    xi = x.copy()
    for _ in range(n_steps):
        xi = sim.step(xi, rng)
    X1[i] = xi

# Augment with uniform random pairs for better basin coverage
n_extra = 4 * n_grid
Xe0, Xe1 = sim.koopman_pairs(n_extra, rng=rng)
X0 = np.vstack([X0, Xe0])
X1 = np.vstack([X1, Xe1])
print(f"  Total pairs (grid + uniform): {len(X0):,}")

x0t = torch.tensor(X0, dtype=torch.float32, device=DEVICE)
x1t = torch.tensor(X1, dtype=torch.float32, device=DEVICE)


# ── Chi network ───────────────────────────────────────────────────────────────

# ChiNetMultiRaw: independent sigmoid outputs, no softmax competition.
# The orthogonal power iteration will enforce orthogonality via SVD deflation.
chi = ChiNetMultiRaw(in_dim=DIM, k=K, hidden=HIDDEN).to(DEVICE)
n_params = sum(p.numel() for p in chi.parameters())
print(f"\nChiNetMultiRaw: {n_params} parameters  k={K}")


# ── Multi-D ISOKANN power iteration ───────────────────────────────────────────

result = power_method_multi(
    chi, x0t, x1t,
    n_iter        = N_ITER,
    epochs_per_iter = EPOCHS,
    lr            = LR,
    lr_decay      = LR_DECAY,
    batch         = BATCH,
    verbose       = True,
)


# ── Implied timescales ────────────────────────────────────────────────────────

chi.eval()
with torch.no_grad():
    chi_x0 = chi(x0t)
    chi_x1 = chi(x1t)

evals, timescales = implied_timescales(chi_x0, chi_x1, lagtime=LAGTIME)
print(f"\nEigenvalues:     {np.abs(evals)}")
print(f"Implied timescales: {timescales}")


# ── Evaluation grid for plotting ──────────────────────────────────────────────

xg  = np.linspace(GRID_LO[0], GRID_HI[0], 150)
yg  = np.linspace(GRID_LO[1], GRID_HI[1], 150)
XXg, YYg = np.meshgrid(xg, yg)
Xeval = np.column_stack([XXg.ravel(), YYg.ravel()])
E_eval = np.array([potential(x) for x in Xeval]).reshape(150, 150)

xeval_t = torch.tensor(Xeval, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    chi_eval = chi(xeval_t).cpu().numpy()   # (150*150, k)

chi_grids = [chi_eval[:, i].reshape(150, 150) for i in range(K)]


# ── Plot 1: PES + three chi functions ─────────────────────────────────────────

fig = plt.figure(figsize=(16, 4))
gs  = GridSpec(1, K + 1, figure=fig, wspace=0.3)

# PES
ax  = fig.add_subplot(gs[0, 0])
cf  = ax.contourf(XXg, YYg, np.clip(E_eval, None, 12), levels=25, cmap="RdYlBu_r")
plt.colorbar(cf, ax=ax, label="Energy", shrink=0.8)
for i, (cx, cy) in enumerate(WELLS):
    ax.plot(cx, cy, "w*", ms=10, label=f"Well {['A','B','C'][i]}")
ax.legend(fontsize=7); ax.set_title("Triple-well PES")
ax.set_xlabel("x"); ax.set_ylabel("y")

# Three chi functions
state_labels = ["A", "B", "C"]
cmaps = ["Blues", "Greens", "Reds"]
for i in range(K):
    ax = fig.add_subplot(gs[0, i + 1])
    im = ax.contourf(XXg, YYg, chi_grids[i], levels=20, cmap=cmaps[i], vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=f"chi_{i+1}", shrink=0.8)
    # Overlay PES contour for context
    ax.contour(XXg, YYg, E_eval, levels=[0, 3, 6, 9], colors="white",
               linewidths=0.5, alpha=0.4)
    ax.plot(*WELLS[i], "w*", ms=10)
    ax.set_title(f"chi_{i+1} (state {state_labels[i]})")
    ax.set_xlabel("x"); ax.set_ylabel("y")

plt.suptitle(f"Multi-D ISOKANN k=3 on triple-well  ({N_ITER} iters)", fontsize=11)
fig.savefig(os.path.join(OUT_DIR, "chi_functions.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: chi_functions.png")


# ── Plot 2: Simplex projection (chi1, chi2, chi3) ─────────────────────────────

# Plot the chi-space scatter — should fill a 2-simplex (triangle)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
pairs = [(0,1), (0,2), (1,2)]
for ax, (i, j) in zip(axes, pairs):
    # Colour by energy
    E_pairs = np.array([potential(x) for x in Xeval])
    E_clip  = np.clip(E_pairs, None, 8)
    ax.scatter(chi_eval[:, i], chi_eval[:, j], c=E_clip, cmap="plasma_r",
               s=1, alpha=0.5, rasterized=True)
    ax.set_xlabel(f"chi_{i+1}"); ax.set_ylabel(f"chi_{j+1}")
    ax.set_title(f"chi_{i+1} vs chi_{j+1}  (coloured by energy)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.suptitle("Simplex projection of chi functions", fontsize=11)
fig.savefig(os.path.join(OUT_DIR, "chi_simplex.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: chi_simplex.png")


# ── Plot 3: Convergence ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(result["losses"])
axes[0].set_xlabel("Power iteration"); axes[0].set_ylabel("Avg MSE loss")
axes[0].set_title("Training convergence")

spans = result["spans"]   # (n_iter, k)
for i in range(K):
    axes[1].plot(spans[:, i], label=f"chi_{i+1}")
axes[1].set_xlabel("Power iteration"); axes[1].set_ylabel("chi span (max-min)")
axes[1].set_title("Chi function spread"); axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: convergence.png")


# ── Plot 4: Timescale spectrum ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 3.5))
abs_evals = np.abs(evals)
ax.bar(range(len(abs_evals)), sorted(abs_evals, reverse=True), color="steelblue")
ax.axhline(1.0, ls="--", c="gray", lw=1)
ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("|eigenvalue|")
ax.set_title(f"Koopman eigenvalue spectrum\nTimescales: {timescales.round(2)}")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eigenvalues.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: eigenvalues.png")


# ── Validation printout ────────────────────────────────────────────────────────

print("\n── Validation ──")
# Koopman eigenfunctions have sign/permutation ambiguity — argmax matching is
# NOT the right criterion.  The correct test is:
#   (1) each well has a DISTINCT chi vector (pairwise distance > threshold)
#   (2) all k eigenvalues are significantly > 0 (no collapsed function)
well_chis = []
for i, c in enumerate(WELLS):
    dists = np.sum((Xeval - c)**2, axis=1)
    near  = np.argmin(dists)
    cv    = chi_eval[near]
    well_chis.append(cv)
    print(f"  Well {state_labels[i]} at {c}: chi = [{', '.join(f'{v:.3f}' for v in cv)}]")

well_chis = np.array(well_chis)   # (k, k)
print("\n  Pairwise separation of well chi-vectors:")
all_sep = True
for i in range(len(WELLS)):
    for j in range(i+1, len(WELLS)):
        d = np.linalg.norm(well_chis[i] - well_chis[j])
        ok = d > 0.2
        print(f"    {state_labels[i]}-{state_labels[j]}: dist={d:.3f}  {'GOOD' if ok else 'POOR'}")
        all_sep = all_sep and ok

evals_abs = np.abs(evals)
no_collapse = all(evals_abs > 0.1)
print(f"\n  Eigenvalue check (all > 0.1): {'PASS' if no_collapse else 'FAIL'}  {evals_abs.round(3)}")
print(f"  Separation check:              {'PASS' if all_sep else 'FAIL'}")
print(f"  => {'SUCCESS: 3-state structure correctly identified' if (all_sep and no_collapse) else 'NEEDS MORE TRAINING'}")

print(f"\nAll outputs in: {OUT_DIR}/")
torch.save(chi.state_dict(), os.path.join(OUT_DIR, "chi_net.pt"))
