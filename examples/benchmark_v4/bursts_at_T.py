# -*- coding: utf-8 -*-
"""
benchmark_v4 gate check at a new temperature, REUSING existing MetaD-grid seeds
(no MetaD rerun). Runs a small set of unbiased bursts at --temp from the saved
seed configurations, builds the discrete transfer operator (same τ), and plots
the eigenspectrum to test whether the C7eq<->alphaR (psi) process separates.
"""
import os, sys, time, argparse
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openmm as mm
from openmm import app, unit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from amore.sims.openmm_sim import DEFAULT_PDB, FORCE_AMBER, phi, psi
DATA = os.path.join(HERE, "data"); FIG = os.path.join(HERE, "figures")

ap = argparse.ArgumentParser()
ap.add_argument("--temp", type=float, default=300.0)
ap.add_argument("--per_cell", type=int, default=50)       # low amount for gate
ap.add_argument("--src_tag", default="T450_m200")         # seeds from this run
ap.add_argument("--src_per_cell", type=int, default=200)  # seeds/cell in source
ap.add_argument("--burst_ps", type=float, default=5.0)
ap.add_argument("--platform", default="Reference")
ap.add_argument("--tag", default="")
args = ap.parse_args()

DT = 0.002; FRICTION = 1.0; TEMP = args.temp
BURST_STEPS = int(round(args.burst_ps / DT))
NB = 40; NCELL = NB * NB; edges = np.linspace(-np.pi, np.pi, NB + 1)
TAG = args.tag or f"T{int(TEMP)}_m{args.per_cell}"
print(f"[{TAG}] reuse seeds from {args.src_tag}; bursts at {TEMP}K, {args.per_cell}/cell, τ={args.burst_ps}ps")

# ── load seeds (blocks of src_per_cell per occupied cell) → take per_cell each ─
seeds_all = np.load(os.path.join(DATA, f"vac_X0_{args.src_tag}.npy")).astype(np.float64)
spc = args.src_per_cell
nocc = len(seeds_all) // spc
assert nocc * spc == len(seeds_all), f"seed count {len(seeds_all)} not divisible by {spc}"
take = min(args.per_cell, spc)
idx = np.concatenate([np.arange(k*spc, k*spc + take) for k in range(nocc)])
seeds = seeds_all[idx]
print(f"seeds: {len(seeds)} ({take}/cell × {nocc} cells)")

# ── bursts at TEMP, single reused Reference context ──────────────────────────
pdb = app.PDBFile(DEFAULT_PDB); ff = app.ForceField(*FORCE_AMBER)
system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, removeCMMotion=False)
integ = mm.LangevinMiddleIntegrator(TEMP*unit.kelvin, FRICTION/unit.picosecond, DT*unit.picoseconds)
sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName(args.platform))
nat = system.getNumParticles()
X0 = seeds; Xtau = np.empty_like(X0)
t0 = time.perf_counter()
for i in range(len(seeds)):
    sim.context.setPositions(X0[i].reshape(nat, 3))
    sim.context.setVelocitiesToTemperature(TEMP*unit.kelvin)
    sim.context.getIntegrator().step(BURST_STEPS)
    Xtau[i] = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten()
    if i % 20000 == 0:
        print(f"  burst {i}/{len(seeds)} ({time.perf_counter()-t0:.0f}s)", flush=True)
print(f"bursts done in {time.perf_counter()-t0:.0f}s")

phi0, psi0, phitau, psitau = phi(X0), psi(X0), phi(Xtau), psi(Xtau)
np.save(os.path.join(DATA, f"vac_X0_{TAG}.npy"), X0.astype(np.float32))
np.save(os.path.join(DATA, f"vac_Xtau_{TAG}.npy"), Xtau.astype(np.float32))
for nm, v in [("phi0", phi0), ("psi0", psi0), ("phitau", phitau), ("psitau", psitau)]:
    np.save(os.path.join(DATA, f"vac_{nm}_{TAG}.npy"), v.astype(np.float32))

# ── transfer operator ────────────────────────────────────────────────────────
def cells(ph, ps):
    return (np.clip(np.digitize(ph, edges)-1, 0, NB-1)*NB + np.clip(np.digitize(ps, edges)-1, 0, NB-1))
ci, cj = cells(phi0, psi0), cells(phitau, psitau)
C = np.zeros((NCELL, NCELL)); np.add.at(C, (ci, cj), 1.0)
rs = C.sum(1, keepdims=True); occ = rs[:, 0] > 0
T = np.zeros_like(C); T[occ] = C[occ]/rs[occ]; T = 0.999*T + 1e-3/NCELL
from scipy.linalg import eig
vals, lvecs = eig(T.T); order = np.argsort(vals.real)[::-1]; vals = vals[order]; lvecs = lvecs[:, order]
its = np.array([-args.burst_ps/np.log(abs(l)) if 0 < abs(l) < 1 else np.inf for l in vals])
NEIG = 10
print(f"\n[{TAG}] eigenvalues / ITS:")
for i in range(NEIG):
    print(f"  EV{i+1}: λ={vals[i].real:.4f}  ITS={its[i]:8.1f} ps")
np.save(os.path.join(DATA, f"vac_transfer_eigvals_{TAG}.npy"), vals[:NEIG])
np.save(os.path.join(DATA, f"vac_transfer_eigvecs_{TAG}.npy"), lvecs[:, :NEIG])

def grid(v):
    g = np.full(NCELL, np.nan); g[occ] = v[occ]; return g.reshape(NB, NB).T
ext = [-180, 180, -180, 180]
fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
axes[0].plot(range(1, NEIG+1), vals[:NEIG].real, "o-"); axes[0].axhline(0, color="k", lw=.5)
axes[0].set_title(f"eigenvalues [{TAG}]"); axes[0].set_xlabel("index"); axes[0].set_ylabel("Re λ")
for k in range(7):
    v = grid(lvecs[:, k].real); vmax = np.nanmax(np.abs(v))
    im = axes[k+1].imshow(v, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[k+1].set_title(f"EV{k+1}  λ={vals[k].real:.3f}  ITS={its[k]:.0f}ps")
    axes[k+1].set_xlabel("φ"); axes[k+1].set_ylabel("ψ"); plt.colorbar(im, ax=axes[k+1], fraction=.046)
fig.suptitle(f"vacuum ADP {TEMP}K transfer operator [{TAG}] — {len(seeds)} pairs, τ={args.burst_ps}ps (seeds reused from {args.src_tag})")
plt.tight_layout(); fig.savefig(os.path.join(FIG, f"vac_eigenvectors_{TAG}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"\nsaved figures/vac_eigenvectors_{TAG}.png")
print("GATE: does EV2 or EV3 show a ψ sign-flip (top↔bottom within the left φ basin) above the bath?")
