# -*- coding: utf-8 -*-
"""
minimal_gen_data.py — data generation for examples/minimal.ipynb.

Runs well-tempered MetaD on vacuum alanine dipeptide in (phi, psi) for coverage,
saves the reconstructed FES, selects burst seeds on a phi/psi grid (cap per cell),
and propagates one 0.1 ps unbiased Koopman burst per seed.

This mirrors examples/isokann_benchmark/lib/generate_data.py (generate_adp) and
examples/MD/alaninedipeptide_metad_phi_psi.py, but additionally persists the FES
and the phi/psi grid axes so the notebook can plot the FES directly.

Outputs under OUT (default /scratch/htc/<user>/amore_minimal):
  metad_fes.npy        (G, G) free energy [kJ/mol], shifted to min 0
  metad_axes.npy       (G,)  phi=psi axis [rad] (shared, endpoint=False)
  X0.npy   (n, 66)  anchor coords [nm]   (grid-capped MetaD frames)
  Xtau.npy (n, 66)  0.1 ps burst endpoints [nm]
  phi0.npy psi0.npy phitau.npy psitau.npy  (n,) dihedrals [rad]
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import openmm as mm
from openmm import app, unit
from amore.sims.openmm_sim import DEFAULT_PDB, phi as adp_phi, psi as adp_psi

DT, FRICTION = 0.002, 1.0
PHI_IDX, PSI_IDX = (4, 6, 8, 14), (6, 8, 14, 16)
GRID = 100          # FES grid points per CV
NB = 40             # seed-selection grid (phi,psi) cells per dim

# globals per worker
_W = {}


def _worker_init(pdb_path, temp):
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, removeCMMotion=False)
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, FRICTION / unit.picosecond,
                                        DT * unit.picoseconds)
    sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference"))
    _W.update(sim=sim, nat=system.getNumParticles(), temp=temp)


def _burst_chunk(args):
    idx, seeds, burst_steps = args
    sim, nat, temp = _W["sim"], _W["nat"], _W["temp"]
    out = np.empty_like(seeds)
    for i in range(len(seeds)):
        sim.context.setPositions(seeds[i].reshape(nat, 3) * unit.nanometer)
        sim.context.setVelocitiesToTemperature(temp * unit.kelvin)
        sim.context.getIntegrator().step(burst_steps)
        out[i] = (sim.context.getState(getPositions=True)
                  .getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten())
    return idx, out


def generate(out_dir, temp=300.0, metad_ns=50.0, max_per_cell=30, burst_ps=0.1, procs=14):
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.normpath(DEFAULT_PDB)
    burst_steps = int(round(burst_ps / DT))
    metad_steps = int(round(metad_ns * 1000 / DT))
    PACE = 1000
    print(f"[minimal] temp={temp} metad={metad_ns}ns max/cell={max_per_cell} "
          f"burst={burst_ps}ps ({burst_steps} steps) procs={procs}", flush=True)

    # ── MetaD for (phi, psi) coverage (single proc, Reference platform) ──
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, removeCMMotion=False)
    phi_cv = mm.CustomTorsionForce("theta"); phi_cv.addTorsion(*PHI_IDX, [])
    psi_cv = mm.CustomTorsionForce("theta"); psi_cv.addTorsion(*PSI_IDX, [])
    phi_b = app.BiasVariable(phi_cv, -np.pi, np.pi, 0.35, True, GRID)
    psi_b = app.BiasVariable(psi_cv, -np.pi, np.pi, 0.35, True, GRID)
    metad = app.Metadynamics(system, [phi_b, psi_b], temp * unit.kelvin, 15.0,
                             1.2 * unit.kilojoules_per_mole, PACE, biasDir=out_dir,
                             saveFrequency=PACE * 100)
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, FRICTION / unit.picosecond,
                                        DT * unit.picoseconds)
    sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(pdb.positions); sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temp * unit.kelvin)

    nh = metad_steps // PACE; snaps = []; t0 = time.perf_counter()
    print(f"[minimal] MetaD {metad_steps} steps (~{metad_ns} ns) ...", flush=True)
    for k in range(nh):
        metad.step(sim, PACE)
        st = sim.context.getState(getPositions=True)
        snaps.append(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten().astype(np.float64))
        if k % 2000 == 0:
            print(f"   hill {k}/{nh} ({time.perf_counter()-t0:.0f}s)", flush=True)
    traj = np.array(snaps)
    print(f"[minimal] MetaD done: {len(traj)} frames in {time.perf_counter()-t0:.0f}s", flush=True)

    # FES + axes
    fes = metad.getFreeEnergy()
    if unit.is_quantity(fes):
        fes = fes.value_in_unit(unit.kilojoules_per_mole)
    fes = np.asarray(fes); fes = fes - fes.min()
    axis = np.linspace(-np.pi, np.pi, GRID, endpoint=False)
    np.save(os.path.join(out_dir, "metad_fes.npy"), fes.astype(np.float32))
    np.save(os.path.join(out_dir, "metad_axes.npy"), axis.astype(np.float32))

    # ── seed selection: cap max_per_cell per occupied NBxNB cell ──
    edges = np.linspace(-np.pi, np.pi, NB + 1)
    tphi, tpsi = adp_phi(traj), adp_psi(traj)
    bi = np.clip(np.digitize(tphi, edges) - 1, 0, NB - 1)
    bj = np.clip(np.digitize(tpsi, edges) - 1, 0, NB - 1)
    rng = np.random.default_rng(0); sidx = []
    for gi in range(NB):
        for gj in range(NB):
            c = np.where((bi == gi) & (bj == gj))[0]
            if len(c):
                sidx.append(rng.choice(c, max_per_cell, replace=True))
    sidx = np.concatenate(sidx)
    seeds = traj[sidx].astype(np.float64)
    print(f"[minimal] seeds: {len(seeds)} ({len(sidx)//max_per_cell} occupied cells)", flush=True)

    # ── bursts: fan out across Reference-platform workers ──
    import multiprocessing as mp
    t0 = time.perf_counter()
    Xtau = np.empty_like(seeds)
    nproc = max(1, procs)
    chunks = np.array_split(np.arange(len(seeds)), nproc * 4)
    tasks = [(c, seeds[c], burst_steps) for c in chunks if len(c)]
    print(f"[minimal] bursts: {len(seeds)} x {burst_steps} steps over {nproc} procs ...", flush=True)
    with mp.Pool(nproc, initializer=_worker_init, initargs=(pdb_path, temp)) as pool:
        for idx, out in pool.imap_unordered(_burst_chunk, tasks):
            Xtau[idx] = out
    print(f"[minimal] bursts done in {time.perf_counter()-t0:.0f}s", flush=True)

    X0 = seeds
    np.save(os.path.join(out_dir, "X0.npy"), X0.astype(np.float32))
    np.save(os.path.join(out_dir, "Xtau.npy"), Xtau.astype(np.float32))
    np.save(os.path.join(out_dir, "phi0.npy"), adp_phi(X0).astype(np.float32))
    np.save(os.path.join(out_dir, "psi0.npy"), adp_psi(X0).astype(np.float32))
    np.save(os.path.join(out_dir, "phitau.npy"), adp_phi(Xtau).astype(np.float32))
    np.save(os.path.join(out_dir, "psitau.npy"), adp_psi(Xtau).astype(np.float32))
    print(f"[minimal] saved arrays to {out_dir} ({len(X0)} pairs)", flush=True)


if __name__ == "__main__":
    import getpass
    default_out = f"/scratch/htc/{getpass.getuser()}/amore_minimal"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--metad_ns", type=float, default=50.0)
    ap.add_argument("--max_per_cell", type=int, default=30)
    ap.add_argument("--procs", type=int, default=14)
    args = ap.parse_args()
    generate(args.out, metad_ns=args.metad_ns, max_per_cell=args.max_per_cell, procs=args.procs)
