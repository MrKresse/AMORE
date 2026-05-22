# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
Benchmark v2 - Panel 0: Reference Quality Check

Triple-well:
  - Empirical basin-committor (fraction of bursts landing in each well basin)
  - Burst-convergence check: recompute at 5, 10, 20 bursts
  - Symmetry check: A/B wells are left-right mirrors; quantify asymmetry
  - Separatrix: cells where max-basin probability < 0.6

Alanine dipeptide (450 K):
  - Transfer-operator eigenspectrum on 40×40 (phi,psi) grid
  - Implied timescales (ITS) via matrix powers of T at tau=5ps
  - Eigenvalue gap ratio lambda_2/lambda_3
  - PCCA+ metastable state assignment (deeptime if available, else simplex)
  - Reference eigenvectors stored for Panel B correlation

Outputs (in panel0/):
  panel0_reference.pdf
  tw_committor.npz   — p_A, p_B, p_C per anchor
  adp_its.npz        — eigenvalues, ITS at multiple lags
  adp_eigvecs.npz    — eigenvectors on grid for Panel B
  adp_pcca.npz       — hard metastable state assignment per occupied cell

Stop criterion: inspect figure before running any method training.
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
from scipy.linalg import eig as scipy_eig

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE    = os.path.dirname(__file__)
DATA_V1 = os.path.join(HERE, "..", "benchmark", "data")
OUT     = os.path.join(HERE, "panel0")
os.makedirs(OUT, exist_ok=True)

# ── colour helpers ─────────────────────────────────────────────────────────────

def _scatter_phi_psi(ax, phi, psi, vals, cmap, vmin=None, vmax=None, s=8, alpha=0.9, label=""):
    sc = ax.scatter(phi, psi, c=vals, cmap=cmap, s=s, alpha=alpha,
                    vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
    ticks  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    tlabs  = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    ax.set_xticks(ticks, tlabs, fontsize=7)
    ax.set_yticks(ticks, tlabs, fontsize=7)
    return sc


# ══════════════════════════════════════════════════════════════════════════════
# TRIPLE WELL
# ══════════════════════════════════════════════════════════════════════════════

print("Loading triple-well data …")
tw = np.load(os.path.join(DATA_V1, "triple_well_koopman.npz"))

anchors = tw["anchors"]          # (N_ANC, 2)
bursts  = tw["bursts"]           # (N_ANC, N_K, 2)
wells   = tw["wells"]            # (3, 2)  A=(-1.2,0), B=(1.2,0), C=(0,1.5)
N_ANC, N_K, _ = bursts.shape

WELL_NAMES = ["A (left)", "B (right)", "C (top)"]
WELL_COLORS = ["steelblue", "tomato", "forestgreen"]

def basin_assignment(pts, w=wells):
    """Return basin index 0/1/2 (closest well) for each point in pts (N,2)."""
    dists = np.linalg.norm(pts[:, None, :] - w[None, :, :], axis=-1)  # (N, 3)
    return np.argmin(dists, axis=1)


def committor(n_bursts=None):
    """Empirical committor: fraction of bursts in each basin. (N_ANC, 3)"""
    nb = N_K if n_bursts is None else n_bursts
    ends  = bursts[:, :nb, :].reshape(-1, 2)     # (N_ANC*nb, 2)
    basis = basin_assignment(ends)               # (N_ANC*nb,)
    basis = basis.reshape(N_ANC, nb)             # (N_ANC, nb)
    p = np.stack([(basis == k).mean(axis=1) for k in range(3)], axis=1)
    return p                                     # (N_ANC, 3)


p20 = committor(20)   # full committor
np.savez(os.path.join(OUT, "tw_committor.npz"),
         anchors=anchors, wells=wells,
         p_A=p20[:, 0], p_B=p20[:, 1], p_C=p20[:, 2])


# ── Burst convergence ─────────────────────────────────────────────────────────
p5  = committor(5)
p10 = committor(10)
# max-abs difference between p5 and p20 (per anchor, max over basins)
conv5  = np.abs(p5  - p20).max(axis=1)
conv10 = np.abs(p10 - p20).max(axis=1)
print(f"  Burst convergence (5->20 bursts): median delta = {np.median(conv5):.4f}, "
      f"95th pct = {np.percentile(conv5, 95):.4f}")
print(f"  Burst convergence (10->20 bursts): median delta = {np.median(conv10):.4f}, "
      f"95th pct = {np.percentile(conv10, 95):.4f}")


# ── Symmetry check ────────────────────────────────────────────────────────────
# A=(-1.2,0), B=(1.2,0): p_A(x,y) should equal p_B(-x,y)
# For each anchor at (xi,yi) with xi>0, find nearest anchor at (-xi,yi)
pos_mask = anchors[:, 0] > 0
mirror_err = []
for i in np.where(pos_mask)[0]:
    xi, yi = anchors[i]
    cands  = np.abs(anchors[:, 1] - yi) + np.abs(anchors[:, 0] + xi)
    j      = int(np.argmin(cands))
    mirror_err.append(abs(p20[i, 0] - p20[j, 1]))   # p_A(xi,yi) vs p_B(-xi,yi)
mirror_err = np.array(mirror_err)
print(f"  Symmetry (A↔B mirror): mean err = {mirror_err.mean():.4f}, "
      f"max = {mirror_err.max():.4f}")


# ── Separatrix cells ──────────────────────────────────────────────────────────
max_p     = p20.max(axis=1)
sep_mask  = max_p < 0.6
n_sep     = sep_mask.sum()
print(f"  Separatrix cells (max-p < 0.6): {n_sep} / {N_ANC}")


# ══════════════════════════════════════════════════════════════════════════════
# ALANINE DIPEPTIDE
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading alanine dipeptide data …")
adp = np.load(os.path.join(DATA_V1, "alanine_koopman.npz"))

a_phi    = adp["anchors_phi"]    # (N_ANC_ADP,)
a_psi    = adp["anchors_psi"]
b_phi    = adp["bursts_phi"]     # (N_ANC_ADP, N_K_ADP)
b_psi    = adp["bursts_psi"]
grid_phi = adp["grid_phi"]       # (40,)
grid_psi = adp["grid_psi"]

N_ANC_ADP, N_K_ADP = b_phi.shape
DISC_GRID = 40
_edges    = np.linspace(-np.pi, np.pi, DISC_GRID + 1)
N_CELLS   = DISC_GRID * DISC_GRID
TAU       = 5.0   # ps (current assumption)
N_EIG     = 15

print(f"  ADP anchors: {N_ANC_ADP},  bursts per anchor: {N_K_ADP}")


def to_cell(phis, psis):
    bi = np.clip(np.digitize(phis, _edges) - 1, 0, DISC_GRID - 1)
    bj = np.clip(np.digitize(psis, _edges) - 1, 0, DISC_GRID - 1)
    return bi * DISC_GRID + bj


# ── Transfer operator at tau=5ps ──────────────────────────────────────────────
ci = to_cell(a_phi, a_psi)      # (N_ANC_ADP,) — source cells

C = np.zeros((N_CELLS, N_CELLS), dtype=np.float64)
for k in range(N_K_ADP):
    cj = to_cell(b_phi[:, k], b_psi[:, k])
    np.add.at(C, (ci, cj), 1.0)

row_sums = C.sum(axis=1, keepdims=True)
occupied = row_sums[:, 0] > 0
T1       = np.zeros_like(C)
T1[occupied] = C[occupied] / row_sums[occupied]

# Light regularisation
EPS = 1e-3
T1  = (1.0 - EPS) * T1 + EPS / N_CELLS

print(f"  Occupied cells: {occupied.sum()} / {N_CELLS}")
np.save(os.path.join(OUT, "adp_transfer_op.npy"), T1)


# ── Eigendecomposition ────────────────────────────────────────────────────────
vals1, rvecs1 = scipy_eig(T1)
order         = np.argsort(vals1.real)[::-1]
vals1         = vals1[order].real
rvecs1        = rvecs1[:, order].real

# Implied timescales at tau=5ps
its1 = -TAU / np.log(np.clip(vals1[1:N_EIG+1], 1e-10, 1 - 1e-10))

print(f"  Top eigenvalues (tau=5ps): {vals1[:6].round(4)}")
print(f"  Eigenvalue gap lambda_2/lambda_3 = {vals1[1]/vals1[2]:.4f}")
print(f"  Top ITS (ps): {its1[:5].round(1)}")


# ── ITS from multi-lag simulation data ────────────────────────────────────────
MULTILAG_FILE = os.path.join(DATA_V1, "alanine_multilag.npz")
its_real = None          # filled below if data exists
its_lags_ps = None
its_vals_per_lag = None

if os.path.exists(MULTILAG_FILE):
    ml = np.load(MULTILAG_FILE)
    ml_phi_src = ml["anchors_phi"]    # (N_ML,)
    ml_psi_src = ml["anchors_psi"]
    ml_bphi    = ml["bursts_phi"]     # (N_ML, N_BURSTS, N_LAGS)
    ml_bpsi    = ml["bursts_psi"]
    ml_lags    = ml["lags_ps"]        # (N_LAGS,)
    N_ML, N_BURSTS_ML, N_LAGS_ML = ml_bphi.shape
    print(f"  Multi-lag data: {N_ML} anchors, {N_BURSTS_ML} bursts, {N_LAGS_ML} lags")

    its_real       = np.zeros((N_LAGS_ML, N_EIG))
    its_lags_ps    = ml_lags
    its_vals_per_lag = np.zeros((N_LAGS_ML, N_EIG + 1))

    # Use a coarser grid for ITS estimation — with only ~1200 transitions,
    # the 40x40=1600 cell grid is too sparse. 10x10=100 cells gives ~12
    # transitions per cell on average, enough for reliable eigenvalues.
    ITS_GRID  = 10
    N_ITS_C   = ITS_GRID * ITS_GRID
    _its_edges = np.linspace(-np.pi, np.pi, ITS_GRID + 1)

    def to_cell_coarse(phis, psis):
        bi = np.clip(np.digitize(phis, _its_edges) - 1, 0, ITS_GRID - 1)
        bj = np.clip(np.digitize(psis, _its_edges) - 1, 0, ITS_GRID - 1)
        return bi * ITS_GRID + bj

    ci_src_c = to_cell_coarse(ml_phi_src, ml_psi_src)

    for li in range(N_LAGS_ML):
        C_li = np.zeros((N_ITS_C, N_ITS_C), dtype=np.float64)
        for k in range(N_BURSTS_ML):
            cj_li = to_cell_coarse(ml_bphi[:, k, li], ml_bpsi[:, k, li])
            np.add.at(C_li, (ci_src_c, cj_li), 1.0)
        rs_li = C_li.sum(axis=1, keepdims=True)
        occ_li = rs_li[:, 0] > 0
        T_li   = np.zeros_like(C_li)
        T_li[occ_li] = C_li[occ_li] / rs_li[occ_li]
        EPS_ITS = 1e-3
        T_li = (1.0 - EPS_ITS) * T_li + EPS_ITS / N_ITS_C

        v_li, _ = scipy_eig(T_li)
        v_li = np.sort(v_li.real)[::-1]
        its_vals_per_lag[li] = v_li[:N_EIG + 1]
        tau_li = float(ml_lags[li])
        its_real[li] = -tau_li / np.log(np.clip(v_li[1:N_EIG+1], 1e-10, 1 - 1e-10))
        print(f"    lag={tau_li:.1f}ps: lambda_2={v_li[1]:.4f}, "
              f"ITS_2={its_real[li, 0]:.1f}ps, ITS_3={its_real[li, 1]:.1f}ps")

    np.savez(os.path.join(OUT, "adp_its.npz"),
             lags_ps=ml_lags,
             its=its_real,
             eigenvalues_per_lag=its_vals_per_lag)
    print(f"  ITS data saved.")
else:
    print("  Multi-lag data not found — ITS plot will be skipped.")
    # Fallback: CK test via matrix powers of T(5ps)
    its_lags_ps = np.array([5.0, 10.0, 20.0, 40.0])
    its_vals_per_lag = np.array([vals1[1:N_EIG+1] ** n for n in [1, 2, 4, 8]])
    its_real = np.array([
        -(n * TAU) / np.log(np.clip(vals1[1:N_EIG+1] ** n, 1e-10, 1-1e-10))
        for n in [1, 2, 4, 8]
    ])
    np.savez(os.path.join(OUT, "adp_its.npz"),
             tau=np.array([TAU]),
             lags=np.array([1, 2, 4, 8]),
             eigenvalues=vals1[:N_EIG+1],
             its_1tau=its_real[0])

np.savez(os.path.join(OUT, "adp_eigvecs.npz"),
         eigenvalues=vals1[:N_EIG+1],
         eigvecs=rvecs1[:, :N_EIG+1],
         occupied=occupied,
         edges=_edges)


# ── PCCA+ metastable assignment (k=3) ─────────────────────────────────────────
# Try deeptime first; fall back to simplex-corner assignment from top eigenvectors.
K_PCCA = 3

def pcca_simplex(evecs, k):
    """
    Minimal PCCA+ via index map (inner simplex algorithm).
    evecs: (n_cells, k) columns are dominant eigenvectors (incl. trivial ev1=const)
    Returns hard assignment (n_cells,) and soft membership (n_cells, k).
    """
    n = evecs.shape[0]
    X = evecs[:, :k].copy()
    # Normalise rows to unit norm (for non-trivial evecs only)
    norms = np.linalg.norm(X[:, 1:], axis=1, keepdims=True) + 1e-12
    # Find k extreme points (ISA-style: iterative column selection)
    ind = [int(np.argmax(np.abs(X).sum(axis=1)))]
    for _ in range(k - 1):
        coeff = (X @ X[ind[-1]]) / (X[ind[-1]] @ X[ind[-1]] + 1e-15)  # (n,)
        X_proj = X - coeff[:, np.newaxis] * X[ind[-1]][np.newaxis, :]
        ind.append(int(np.argmax(np.linalg.norm(X_proj, axis=1))))
    # Affine transformation via the simplex corners
    try:
        A_inv = np.linalg.inv(X[ind, :])
        chi   = (X @ A_inv.T)
        chi   = np.clip(chi, 0, None)
        chi  /= chi.sum(axis=1, keepdims=True) + 1e-12
    except np.linalg.LinAlgError:
        chi = np.ones((n, k)) / k
    return chi.argmax(axis=1), chi


pcca_used = "kmeans"
occ_idx = np.where(occupied)[0]

# Use k-means on the top (K_PCCA-1) non-trivial eigenvectors — robust regardless of eigenvalue gap.
# Note: if only k_eff=2 modes are live, the 3rd cluster will slice a gradient, not a basin. Flag this.
from sklearn.cluster import KMeans
evecs_occ_nontrivial = rvecs1[np.ix_(occ_idx, np.arange(1, K_PCCA))]  # skip trivial ev1
km = KMeans(n_clusters=K_PCCA, n_init=20, random_state=42).fit(evecs_occ_nontrivial)
hard_occ = km.labels_
hard_state_full = np.zeros(N_CELLS, dtype=int)
hard_state_full[occ_idx] = hard_occ
pcca_full = np.zeros((N_CELLS, K_PCCA))
pcca_full[occ_idx, hard_occ] = 1.0

print(f"  PCCA+ backend: {pcca_used}")
state_counts = [(hard_state_full[occ_idx] == s).sum() for s in range(K_PCCA)]
print(f"  State sizes (occupied cells): {state_counts}")

np.savez(os.path.join(OUT, "adp_pcca.npz"),
         hard_state=hard_state_full,
         chi_soft=pcca_full,
         occ_idx=occ_idx,
         k=np.array([K_PCCA]),
         method=np.array([pcca_used]))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Panel 0 — Reference Quality Check", fontsize=13, y=0.98)

gs = GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.35,
              left=0.05, right=0.97, top=0.92, bottom=0.06)

# ── TW row 1: three committors ─────────────────────────────────────────────────
for col, (label, vals) in enumerate(zip(
        ["$p_A$ (left well)", "$p_B$ (right well)", "$p_C$ (top well)"],
        [p20[:, 0], p20[:, 1], p20[:, 2]])):
    ax = fig.add_subplot(gs[0, col])
    sc = ax.scatter(anchors[:, 0], anchors[:, 1], c=vals,
                    cmap="RdYlBu_r", vmin=0, vmax=1, s=8, rasterized=True)
    ax.scatter(*wells.T, c=WELL_COLORS, s=80, zorder=5, marker="*",
               edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title(f"TW committor {label}", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")

# ── TW row 1, col 3: symmetry residual ────────────────────────────────────────
ax_sym = fig.add_subplot(gs[0, 3])
sc_sym = ax_sym.scatter(anchors[pos_mask, 0], anchors[pos_mask, 1],
                        c=mirror_err, cmap="hot_r", vmin=0, vmax=0.3, s=8)
plt.colorbar(sc_sym, ax=ax_sym, fraction=0.046, label="|p_A(x,y)-p_B(-x,y)|")
ax_sym.set_title(f"A↔B symmetry residual\nmean={mirror_err.mean():.3f}, max={mirror_err.max():.3f}",
                 fontsize=9)
ax_sym.set_xlabel("x"); ax_sym.set_ylabel("y")

# ── TW row 1, col 4: burst convergence ────────────────────────────────────────
ax_cv = fig.add_subplot(gs[0, 4])
ax_cv.scatter(range(N_ANC), np.sort(conv5),  s=2, alpha=0.4, label="5→20 bursts")
ax_cv.scatter(range(N_ANC), np.sort(conv10), s=2, alpha=0.4, label="10→20 bursts")
ax_cv.axhline(0.05, color="k", ls="--", lw=0.8, label="0.05 threshold")
ax_cv.set_xlabel("Anchor (sorted)"); ax_cv.set_ylabel("max |Δp|")
ax_cv.set_title("TW burst convergence", fontsize=9)
ax_cv.legend(fontsize=7)

# ── ADP row 2: stationary distribution, EV2, EV3 ──────────────────────────────
pi_vec  = np.abs(rvecs1[:, 0])
pi_vec /= pi_vec.sum()

cell_centers = 0.5 * (_edges[:-1] + _edges[1:])
phi_centers  = cell_centers
psi_centers  = cell_centers

def cell_phi_psi():
    ci_idx, cj_idx = np.divmod(occ_idx, DISC_GRID)
    return phi_centers[ci_idx], psi_centers[cj_idx]

occ_phi, occ_psi = cell_phi_psi()

ax_pi = fig.add_subplot(gs[1, 0])
sc    = ax_pi.scatter(occ_phi, occ_psi, c=np.log(pi_vec[occ_idx]+1e-15),
                      cmap="viridis", s=6, rasterized=True)
plt.colorbar(sc, ax=ax_pi, fraction=0.046, label="log π")
ax_pi.set_title("ADP stationary dist. log π", fontsize=9)
ax_pi.set_xlabel(r"$\phi$"); ax_pi.set_ylabel(r"$\psi$")

for col, ev_i in enumerate([1, 2]):
    ax = fig.add_subplot(gs[1, col + 1])
    v  = rvecs1[occ_idx, ev_i]
    vm = np.abs(v).max()
    sc = ax.scatter(occ_phi, occ_psi, c=v, cmap="RdBu_r", vmin=-vm, vmax=vm,
                    s=6, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.046,
                 label=f"λ={vals1[ev_i]:.4f}")
    ax.set_title(f"ADP EV {ev_i+1} (λ={vals1[ev_i]:.4f})", fontsize=9)
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\psi$")

# ── ADP row 2, col 3: PCCA+ state map ─────────────────────────────────────────
ax_pc = fig.add_subplot(gs[1, 3])
state_cmap = plt.cm.get_cmap("Set1", K_PCCA)
sc_pc = ax_pc.scatter(occ_phi, occ_psi,
                      c=hard_state_full[occ_idx],
                      cmap=state_cmap, vmin=-0.5, vmax=K_PCCA - 0.5,
                      s=6, rasterized=True)
plt.colorbar(sc_pc, ax=ax_pc, fraction=0.046, ticks=range(K_PCCA),
             label=f"PCCA+ state ({pcca_used})")
ax_pc.set_title(f"ADP PCCA+ k={K_PCCA}", fontsize=9)
ax_pc.set_xlabel(r"$\phi$"); ax_pc.set_ylabel(r"$\psi$")

# ── ADP row 2, col 4: eigenvalue spectrum ─────────────────────────────────────
ax_ev = fig.add_subplot(gs[1, 4])
ax_ev.plot(np.arange(1, N_EIG + 2), vals1[:N_EIG + 1], "o-", ms=5, lw=1.5)
ax_ev.axvline(3.5, color="red", lw=0.8, ls="--", label="gap k=3")
ax_ev.set_xlabel("Eigenvalue index")
ax_ev.set_ylabel(r"Re($\lambda$)")
ax_ev.set_title(f"ADP eigenspectrum (τ={TAU}ps)\nλ₂/λ₃={vals1[1]/vals1[2]:.3f}", fontsize=9)
ax_ev.legend(fontsize=7)

# ── ADP row 3: ITS plot ────────────────────────────────────────────────────────
ax_its = fig.add_subplot(gs[2, :3])
colors_its = plt.cm.tab10(np.linspace(0, 1, min(6, N_EIG)))
if its_real is not None:
    for ki in range(min(2, N_EIG)):   # only ITS2 and ITS3 -- higher modes all noisy
        ax_its.plot(its_lags_ps, its_real[:, ki], "o-",
                    color=colors_its[ki], ms=5, lw=1.5, alpha=0.6,
                    label=f"ITS {ki+2} (1 burst/anchor, noisy)")
    # Expected ITS from T(5ps) as dashed reference
    ax_its.axhline(its1[0], color="steelblue", ls="--", lw=1.5, label=f"ITS_2={its1[0]:.0f}ps from T(5ps) 20-burst")
    ax_its.axhline(its1[1], color="tomato", ls="--", lw=1.5, label=f"ITS_3={its1[1]:.1f}ps from T(5ps) 20-burst")
    ax_its.set_title("ADP ITS from multi-lag data (10x10 grid, 1 burst/anchor)\n"
                     "Dashed = ITS from reliable T(5ps) 20-burst estimate.\n"
                     "Short-lag ITS noisy: ~5000 transitions needed to resolve 500ps mode",
                     fontsize=8)
else:
    ax_its.set_title("ITS data missing — run 01b_adp_its_data.py", fontsize=9)
ax_its.set_xlabel("Lag time tau (ps)")
ax_its.set_ylabel("Implied timescale (ps)")
ax_its.set_yscale("log")
ax_its.legend(fontsize=7, ncol=1)

# ── ADP row 3, col 3-4: ITS bar at tau=5ps ───────────────────────────────────
ax_bar = fig.add_subplot(gs[2, 3:])
ax_bar.bar(np.arange(1, N_EIG + 1), its1, color="steelblue", alpha=0.8)
ax_bar.axvline(1.5, color="red", lw=1.2, ls="--", label="gap after k=1")
ax_bar.set_xlabel("Mode index (2...)")
ax_bar.set_ylabel("ITS (ps)")
ax_bar.set_title(f"ADP ITS at tau={TAU}ps (from T(5ps))\n"
                 f"ITS_2={its1[0]:.1f}ps, ITS_3={its1[1]:.1f}ps, "
                 f"gap_2/3={its1[0]/its1[1]:.2f}x", fontsize=9)
ax_bar.legend(fontsize=7)

fig.savefig(os.path.join(OUT, "panel0_reference.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "panel0_reference.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nPanel 0 done. Outputs in {OUT}/")
print(f"  panel0_reference.pdf")
print(f"  tw_committor.npz")
print(f"  adp_its.npz")
print(f"  adp_eigvecs.npz")
print(f"  adp_pcca.npz")
print()
print("=== SUMMARY ===")
print(f"Triple-well:")
print(f"  Burst convergence (5→20): median Δp = {np.median(conv5):.4f}")
print(f"  Symmetry A↔B: mean err = {mirror_err.mean():.4f}")
print(f"  Separatrix cells (max-p<0.6): {n_sep}/{N_ANC}")
print(f"Alanine dipeptide (tau={TAU}ps):")
print(f"  Occupied cells: {occupied.sum()}/{N_CELLS}")
print(f"  lambda_2 = {vals1[1]:.6f},  lambda_3 = {vals1[2]:.6f}")
print(f"  lambda_2/lambda_3 = {vals1[1]/vals1[2]:.4f}")
print(f"  ITS_2 = {its1[0]:.1f} ps,  ITS_3 = {its1[1]:.1f} ps")
print(f"  ITS_2/ITS_3 = {its1[0]/its1[1]:.2f}x")
print(f"  PCCA+ backend: {pcca_used},  state sizes: {state_counts}")
