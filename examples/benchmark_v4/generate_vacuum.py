# -*- coding: utf-8 -*-
"""
benchmark_v4 — regenerate the vacuum alanine-dipeptide Koopman dataset and its
discrete transfer operator (reproducing examples/MD/alaninedipeptide_discrete.py,
cleaned + OpenCL + single reused context for fast bursts).

Goal: enough per-cell statistics (well-tempered MetaD seeding + ~MAX_PER_CELL
bursts/cell) to resolve the C7eq<->alphaR (psi) process as a distinct slow
eigenvector — the gate before training ISOKANN (benchmark_v4).

Outputs (data/):
  vac_X0.npy, vac_Xtau.npy        (n,66) Koopman pair coords [nm]
  vac_phi0/psi0/phitau/psitau.npy (n,)   dihedrals
  vac_transfer_eigvals.npy        (NEIG,) complex
  vac_transfer_eigvecs.npy        (1600,NEIG) left eigvecs on 40x40 grid
figures/: vac_eigenspectrum.png, vac_eigenvectors.png, vac_metad_fes.png
"""
import os, sys, time, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openmm as mm
from openmm import app, unit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from amore.sims.openmm_sim import DEFAULT_PDB, FORCE_AMBER, phi, psi
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--temp", type=float, default=450.0)
ap.add_argument("--max_per_cell", type=int, default=150)
ap.add_argument("--metad_ns", type=float, default=50.0)
ap.add_argument("--burst_ps", type=float, default=5.0)
ap.add_argument("--platform", default="Reference")
ap.add_argument("--tag", default="")
args = ap.parse_args()

DT = 0.002; FRICTION = 1.0
TEMP = args.temp
BURST_STEPS = int(round(args.burst_ps / DT))
METAD_STEPS = int(round(args.metad_ns * 1000 / DT))
PACE = 1000; SAVE_EVERY = 1000
NB = 40; NCELL = NB * NB
edges = np.linspace(-np.pi, np.pi, NB + 1)
PHI_IDX = (4, 6, 8, 14); PSI_IDX = (6, 8, 14, 16)
TAG = (args.tag or f"T{int(TEMP)}_m{args.max_per_cell}")
print(f"[{TAG}] temp={TEMP} max/cell={args.max_per_cell} metad={args.metad_ns}ns "
      f"burst={args.burst_ps}ps platform={args.platform}")

# ── build system + MetaD ─────────────────────────────────────────────────────
pdb = app.PDBFile(DEFAULT_PDB); ff = app.ForceField(*FORCE_AMBER)
system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, removeCMMotion=False)
phi_cv = mm.CustomTorsionForce("theta"); phi_cv.addTorsion(*PHI_IDX, [])
psi_cv = mm.CustomTorsionForce("theta"); psi_cv.addTorsion(*PSI_IDX, [])
phi_b = app.BiasVariable(phi_cv, -np.pi, np.pi, 0.2, True, 100)
psi_b = app.BiasVariable(psi_cv, -np.pi, np.pi, 0.2, True, 100)
metad = app.Metadynamics(system, [phi_b, psi_b], TEMP*unit.kelvin, 15.0,
                         1.2*unit.kilojoules_per_mole, PACE, biasDir=DATA,
                         saveFrequency=PACE*100)
integ = mm.LangevinMiddleIntegrator(TEMP*unit.kelvin, FRICTION/unit.picosecond, DT*unit.picoseconds)
plat = mm.Platform.getPlatformByName(args.platform)
sim = app.Simulation(pdb.topology, system, integ, plat)
sim.context.setPositions(pdb.positions); sim.minimizeEnergy()
sim.context.setVelocitiesToTemperature(TEMP*unit.kelvin)

print(f"Running MetaD: {METAD_STEPS} steps (~{args.metad_ns} ns) ...")
nh = METAD_STEPS // PACE; snaps = []
t0 = time.perf_counter()
for k in range(nh):
    metad.step(sim, PACE)
    if (k * PACE) % SAVE_EVERY == 0:
        st = sim.context.getState(getPositions=True)
        snaps.append(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten().astype(np.float64))
    if k % 2000 == 0:
        print(f"  hill {k}/{nh}  ({time.perf_counter()-t0:.0f}s)", flush=True)
metad_traj = np.array(snaps)
print(f"MetaD done: {len(metad_traj)} frames in {time.perf_counter()-t0:.0f}s")
fes = metad.getFreeEnergy(); fes = (fes - fes.min()).value_in_unit(unit.kilojoules_per_mole)

# ── seed selection: cap MAX_PER_CELL per occupied 40x40 cell ─────────────────
tphi = phi(metad_traj); tpsi = psi(metad_traj)
bi = np.clip(np.digitize(tphi, edges) - 1, 0, NB - 1)
bj = np.clip(np.digitize(tpsi, edges) - 1, 0, NB - 1)
rng = np.random.default_rng(0); sidx = []
for gi in range(NB):
    for gj in range(NB):
        c = np.where((bi == gi) & (bj == gj))[0]
        if len(c):
            sidx.append(rng.choice(c, args.max_per_cell, replace=True))
sidx = np.concatenate(sidx); seeds = metad_traj[sidx]
nocc = len(sidx) // args.max_per_cell
print(f"seeds: {len(seeds)} ({nocc}/{NCELL} cells occupied)")

# ── bursts: single reused context, fresh velocities each burst ───────────────
integ2 = mm.LangevinMiddleIntegrator(TEMP*unit.kelvin, FRICTION/unit.picosecond, DT*unit.picoseconds)
bsim = app.Simulation(pdb.topology,
                      ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, removeCMMotion=False),
                      integ2, plat)
nat = system.getNumParticles()
X0 = seeds.astype(np.float64); Xtau = np.empty_like(X0)
print(f"bursts: {len(seeds)} x {BURST_STEPS} steps ({args.burst_ps} ps) ...")
t0 = time.perf_counter()
for i in range(len(seeds)):
    bsim.context.setPositions(X0[i].reshape(nat, 3))
    bsim.context.setVelocitiesToTemperature(TEMP*unit.kelvin)
    bsim.context.getIntegrator().step(BURST_STEPS)
    Xtau[i] = bsim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten()
    if i % 20000 == 0:
        el = time.perf_counter()-t0
        print(f"  burst {i}/{len(seeds)}  ({el:.0f}s, {1000*el/max(i,1):.1f} ms/burst)", flush=True)
print(f"bursts done in {time.perf_counter()-t0:.0f}s")

phi0, psi0 = phi(X0), psi(X0); phitau, psitau = phi(Xtau), psi(Xtau)
sfx = ("_" + args.tag) if args.tag else ""
np.save(os.path.join(DATA, f"vac_X0{sfx}.npy"), X0.astype(np.float32))
np.save(os.path.join(DATA, f"vac_Xtau{sfx}.npy"), Xtau.astype(np.float32))
for nm, v in [("phi0", phi0), ("psi0", psi0), ("phitau", phitau), ("psitau", psitau)]:
    np.save(os.path.join(DATA, f"vac_{nm}{sfx}.npy"), v.astype(np.float32))

# ── discrete transfer operator on 40x40 grid ─────────────────────────────────
def cells(ph, ps):
    return (np.clip(np.digitize(ph, edges)-1, 0, NB-1) * NB
            + np.clip(np.digitize(ps, edges)-1, 0, NB-1))
ci, cj = cells(phi0, psi0), cells(phitau, psitau)
C = np.zeros((NCELL, NCELL)); np.add.at(C, (ci, cj), 1.0)
rs = C.sum(1, keepdims=True); occ = rs[:, 0] > 0
T = np.zeros_like(C); T[occ] = C[occ] / rs[occ]
EPS = 1e-3; T = (1-EPS)*T + EPS/NCELL
from scipy.linalg import eig
NEIG = 12
vals, lvecs = eig(T.T)
order = np.argsort(vals.real)[::-1]; vals = vals[order]; lvecs = lvecs[:, order]
its = np.array([-args.burst_ps/np.log(abs(l)) if 0 < abs(l) < 1 else np.inf for l in vals])
print("\nTransfer-op eigenvalues / ITS (ps):")
for i in range(NEIG):
    print(f"  EV{i+1}: lambda={vals[i].real:.4f}  ITS={its[i]:8.1f} ps")
np.save(os.path.join(DATA, f"vac_transfer_eigvals{sfx}.npy"), vals[:NEIG])
np.save(os.path.join(DATA, f"vac_transfer_eigvecs{sfx}.npy"), lvecs[:, :NEIG])

def grid(v):
    g = np.full(NCELL, np.nan); g[occ] = v[occ]; return g.reshape(NB, NB).T
ext = [-180, 180, -180, 180]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(range(1, NEIG+1), vals[:NEIG].real, "o-"); ax[0].axhline(0, color="k", lw=.5)
ax[0].set_title(f"transfer-op eigenvalues [{TAG}]"); ax[0].set_xlabel("index"); ax[0].set_ylabel("Re λ")
ax[1].bar(range(2, 2+NEIG-1), its[1:NEIG]); ax[1].axhline(args.burst_ps, color="r", ls="--", label="τ")
ax[1].set_yscale("log"); ax[1].set_title("ITS (ps)"); ax[1].set_xlabel("mode"); ax[1].legend()
plt.tight_layout(); fig.savefig(os.path.join(FIG, f"vac_eigenspectrum{sfx}.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
im = axes[0].contourf(np.linspace(-180,180,100), np.linspace(-180,180,100), fes.T, levels=25, cmap="RdYlBu_r")
axes[0].set_title("MetaD FES (kJ/mol)"); plt.colorbar(im, ax=axes[0], fraction=.046)
for k in range(7):
    v = grid(lvecs[:, k].real); vmax = np.nanmax(np.abs(v))
    im = axes[k+1].imshow(v, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[k+1].set_title(f"left EV{k+1}  λ={vals[k].real:.3f}  ITS={its[k]:.0f}ps")
    plt.colorbar(im, ax=axes[k+1], fraction=.046)
for a in axes: a.set_xlabel("φ"); a.set_ylabel("ψ")
fig.suptitle(f"vacuum ADP transfer operator [{TAG}] — {len(seeds)} pairs, τ={args.burst_ps}ps")
plt.tight_layout(); fig.savefig(os.path.join(FIG, f"vac_eigenvectors{sfx}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"\nsaved figures vac_eigenspectrum{sfx}.png, vac_eigenvectors{sfx}.png + data")
print("GATE: does a psi-process (C7eq<->alphaR; sign flip across psi within the left phi basin) appear as EV2/EV3 above the bath?")
