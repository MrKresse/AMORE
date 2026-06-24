# -*- coding: utf-8 -*-
"""
generate_data.py — regenerate the reversible benchmark systems' raw data on scratch.

Two systems share this generator (the directed-ring toy is generated lazily by
systems.py / toy code, since it is pure-numpy and tiny):

  tw   2D triple-well Koopman pairs + empirical basin committors  (p_A,p_B,p_C)
       Reproduces examples/benchmark/00_simulate_triple_well.py and
       examples/benchmark_v2/panel0.py exactly (same potential, sigma, dt, lag,
       grid, 20 bursts, 4x4 patch CV splits) — only vectorised for speed. The
       committor is a statistic of the bursts and is RNG-ordering independent, so
       the vectorised pairs give the same reference within Monte-Carlo error.

  adp  vacuum alanine dipeptide, 300 K, tau=0.1 ps Koopman pairs.
       Reproduces examples/benchmark_v4/generate_vacuum.py: well-tempered MetaD in
       (phi,psi) for coverage, cap MAX_PER_CELL bursts per occupied 40x40 cell,
       then one 0.1 ps burst per anchor. Bursts run on the OpenMM *Reference*
       platform (fastest for a 22-atom system here — ~130 ps/s/core, ~14x the
       'CPU' platform) and are fanned out across processes.

Outputs (under paths.DATA):
  triple_well_koopman.npz      anchors,bursts,x0,x1,patch_splits,wells,...
  tw_committor.npz             anchors,wells,p_A,p_B,p_C
  vac_X0_T300_0p1.npy          (n,66) anchor coords [nm]
  vac_Xtau_T300_0p1.npy        (n,66) 0.1 ps endpoints
  vac_{phi0,psi0,phitau,psitau}_T300_0p1.npy

Usage:
  python generate_data.py tw
  python generate_data.py adp [--metad_ns 50 --max_per_cell 50 --procs 14]
  python generate_data.py all
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
sys.path.insert(0, paths.AMORE_SRC)

# ───────────────────────────── triple well ──────────────────────────────────
TW_WELLS = np.array([[-1.2, 0.0], [1.2, 0.0], [0.0, 1.5]])
TW_DEPTH, TW_WIDTH, TW_WALL = 5.0, 0.5, 0.3
TW_GRID_NX = TW_GRID_NY = 40
TW_LO = np.array([-2.0, -0.8]); TW_HI = np.array([2.0, 2.3])
TW_EMAX = 10.0
TW_DT, TW_SIGMA, TW_LAG = 5e-4, 1.2, 0.30
TW_NBURST = 20
TW_PATCH_N, TW_NHOLD, TW_NSEEDS, TW_SEED = 4, 4, 5, 42


def _tw_energy(X):
    V = np.zeros(len(X))
    for c in TW_WELLS:
        V += -TW_DEPTH * np.exp(-((X - c) ** 2).sum(1) / (2 * TW_WIDTH))
    V += TW_WALL * (X ** 2).sum(1)
    return V


def _tw_force(X):
    g = np.zeros_like(X)
    for c in TW_WELLS:
        d = X - c
        g += (TW_DEPTH / TW_WIDTH) * d * np.exp(-(d ** 2).sum(1, keepdims=True) / (2 * TW_WIDTH))
    g += 2 * TW_WALL * X
    return -g


def _tw_basin(pts):
    d = np.linalg.norm(pts[:, None, :] - TW_WELLS[None], axis=-1)
    return d.argmin(1)


def generate_triple_well():
    t0 = time.perf_counter()
    rng = np.random.default_rng(TW_SEED)
    xs = np.linspace(TW_LO[0], TW_HI[0], TW_GRID_NX)
    ys = np.linspace(TW_LO[1], TW_HI[1], TW_GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_all = np.column_stack([XX.ravel(), YY.ravel()])
    grid = grid_all[_tw_energy(grid_all) < TW_EMAX]
    N = len(grid)
    print(f"[tw] anchors below E={TW_EMAX}: {N}/{len(grid_all)}")

    n_steps = max(1, int(round(TW_LAG / TW_DT)))     # 600
    sq = TW_SIGMA * np.sqrt(TW_DT)
    X = np.repeat(grid, TW_NBURST, axis=0).astype(np.float64)   # (N*K, 2)
    print(f"[tw] propagating {len(X)} points x {n_steps} steps (vectorised)...")
    for _ in range(n_steps):
        X += _tw_force(X) * TW_DT + sq * rng.standard_normal(X.shape)
    bursts = X.reshape(N, TW_NBURST, 2).astype(np.float32)

    # empirical basin committor (fraction of bursts in each basin) — panel0 convention
    basis = _tw_basin(bursts.reshape(-1, 2)).reshape(N, TW_NBURST)
    p = np.stack([(basis == k).mean(1) for k in range(3)], 1).astype(np.float64)

    # 4x4 patch CV splits (reproduces 00_simulate_triple_well.py)
    px = np.clip(np.floor((grid[:, 0] - TW_LO[0]) / (TW_HI[0] - TW_LO[0]) * TW_PATCH_N).astype(int), 0, TW_PATCH_N - 1)
    py = np.clip(np.floor((grid[:, 1] - TW_LO[1]) / (TW_HI[1] - TW_LO[1]) * TW_PATCH_N).astype(int), 0, TW_PATCH_N - 1)
    patch_id = px * TW_PATCH_N + py
    srng = np.random.default_rng(TW_SEED + 100)
    splits = np.zeros((TW_NSEEDS, N), dtype=np.int8)
    for s in range(TW_NSEEDS):
        for h in srng.choice(TW_PATCH_N * TW_PATCH_N, TW_NHOLD, replace=False):
            splits[s, patch_id == h] = 1

    x0 = np.repeat(grid, TW_NBURST, axis=0).astype(np.float32)
    np.savez(os.path.join(paths.DATA, "triple_well_koopman.npz"),
             anchors=grid.astype(np.float32), bursts=bursts,
             x0=x0, x1=bursts.reshape(-1, 2),
             patch_splits=splits, wells=TW_WELLS.astype(np.float32),
             grid_lo=TW_LO.astype(np.float32), grid_hi=TW_HI.astype(np.float32),
             n_bursts=np.array([TW_NBURST]), lagtime=np.array([TW_LAG]),
             sigma=np.array([TW_SIGMA]))
    np.savez(os.path.join(paths.DATA, "tw_committor.npz"),
             anchors=grid.astype(np.float32), wells=TW_WELLS.astype(np.float32),
             p_A=p[:, 0], p_B=p[:, 1], p_C=p[:, 2])
    print(f"[tw] done in {time.perf_counter()-t0:.1f}s -> triple_well_koopman.npz, tw_committor.npz")


# ───────────────────────────── alanine dipeptide ────────────────────────────
ADP_DT, ADP_FRICTION = 0.002, 1.0
ADP_PHI_IDX = (4, 6, 8, 14); ADP_PSI_IDX = (6, 8, 14, 16)
ADP_NB = 40

# globals populated per worker
_W = {}


def _adp_worker_init(pdb_path, temp):
    import openmm as mm
    from openmm import app, unit
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, removeCMMotion=False)
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, ADP_FRICTION / unit.picosecond,
                                        ADP_DT * unit.picoseconds)
    sim = app.Simulation(pdb.topology, system, integ,
                         mm.Platform.getPlatformByName("Reference"))
    _W["sim"] = sim
    _W["nat"] = system.getNumParticles()
    _W["temp"] = temp
    _W["unit"] = unit


def _adp_burst_chunk(args):
    """Propagate a chunk of seeds burst_steps each; return (idx, Xtau_flat)."""
    idx, seeds, burst_steps = args
    sim = _W["sim"]; nat = _W["nat"]; unit = _W["unit"]; temp = _W["temp"]
    out = np.empty_like(seeds)
    for i in range(len(seeds)):
        sim.context.setPositions(seeds[i].reshape(nat, 3) * unit.nanometer)
        sim.context.setVelocitiesToTemperature(temp * unit.kelvin)
        sim.context.getIntegrator().step(burst_steps)
        out[i] = (sim.context.getState(getPositions=True)
                  .getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten())
    return idx, out


def generate_adp(temp=300.0, metad_ns=50.0, max_per_cell=50, burst_ps=0.1,
                 procs=14, tag="T300_0p1"):
    import openmm as mm
    from openmm import app, unit
    from amore.sims.openmm_sim import phi as adp_phi, psi as adp_psi
    pdb_path = os.path.normpath(paths.DEFAULT_PDB)

    edges = np.linspace(-np.pi, np.pi, ADP_NB + 1)
    burst_steps = int(round(burst_ps / ADP_DT))
    metad_steps = int(round(metad_ns * 1000 / ADP_DT))
    PACE = 1000
    print(f"[adp:{tag}] temp={temp} metad={metad_ns}ns max/cell={max_per_cell} "
          f"burst={burst_ps}ps ({burst_steps} steps) procs={procs}")

    # ── MetaD for (phi,psi) coverage (single process, Reference platform) ──
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, removeCMMotion=False)
    phi_cv = mm.CustomTorsionForce("theta"); phi_cv.addTorsion(*ADP_PHI_IDX, [])
    psi_cv = mm.CustomTorsionForce("theta"); psi_cv.addTorsion(*ADP_PSI_IDX, [])
    phi_b = app.BiasVariable(phi_cv, -np.pi, np.pi, 0.2, True, 100)
    psi_b = app.BiasVariable(psi_cv, -np.pi, np.pi, 0.2, True, 100)
    metad = app.Metadynamics(system, [phi_b, psi_b], temp * unit.kelvin, 15.0,
                             1.2 * unit.kilojoules_per_mole, PACE, biasDir=paths.DATA,
                             saveFrequency=PACE * 100)
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, ADP_FRICTION / unit.picosecond,
                                        ADP_DT * unit.picoseconds)
    sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(pdb.positions); sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temp * unit.kelvin)

    nh = metad_steps // PACE; snaps = []; t0 = time.perf_counter()
    print(f"[adp:{tag}] MetaD {metad_steps} steps (~{metad_ns} ns) ...")
    for k in range(nh):
        metad.step(sim, PACE)
        st = sim.context.getState(getPositions=True)
        snaps.append(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer).flatten().astype(np.float64))
        if k % 2000 == 0:
            print(f"   hill {k}/{nh} ({time.perf_counter()-t0:.0f}s)", flush=True)
    traj = np.array(snaps)
    print(f"[adp:{tag}] MetaD done: {len(traj)} frames in {time.perf_counter()-t0:.0f}s")

    # ── seed selection: cap max_per_cell per occupied 40x40 cell ──
    tphi, tpsi = adp_phi(traj), adp_psi(traj)
    bi = np.clip(np.digitize(tphi, edges) - 1, 0, ADP_NB - 1)
    bj = np.clip(np.digitize(tpsi, edges) - 1, 0, ADP_NB - 1)
    rng = np.random.default_rng(0); sidx = []
    for gi in range(ADP_NB):
        for gj in range(ADP_NB):
            c = np.where((bi == gi) & (bj == gj))[0]
            if len(c):
                sidx.append(rng.choice(c, max_per_cell, replace=True))
    sidx = np.concatenate(sidx)
    seeds = traj[sidx].astype(np.float64)
    nocc = len(sidx) // max_per_cell
    print(f"[adp:{tag}] seeds: {len(seeds)} ({nocc} occupied cells)")

    # ── bursts: fan out across Reference-platform worker processes ──
    import multiprocessing as mp
    t0 = time.perf_counter()
    Xtau = np.empty_like(seeds)
    nproc = max(1, procs)
    chunks = np.array_split(np.arange(len(seeds)), nproc * 4)
    tasks = [(c, seeds[c], burst_steps) for c in chunks if len(c)]
    print(f"[adp:{tag}] bursts: {len(seeds)} x {burst_steps} steps over {nproc} procs ...")
    with mp.Pool(nproc, initializer=_adp_worker_init, initargs=(pdb_path, temp)) as pool:
        for idx, out in pool.imap_unordered(_adp_burst_chunk, tasks):
            Xtau[idx] = out
    print(f"[adp:{tag}] bursts done in {time.perf_counter()-t0:.0f}s")

    X0 = seeds
    phi0, psi0 = adp_phi(X0), adp_psi(X0)
    phitau, psitau = adp_phi(Xtau), adp_psi(Xtau)
    for nm, v in [("X0", X0), ("Xtau", Xtau)]:
        np.save(os.path.join(paths.DATA, f"vac_{nm}_{tag}.npy"), v.astype(np.float32))
    for nm, v in [("phi0", phi0), ("psi0", psi0), ("phitau", phitau), ("psitau", psitau)]:
        np.save(os.path.join(paths.DATA, f"vac_{nm}_{tag}.npy"), v.astype(np.float32))
    print(f"[adp:{tag}] saved vac_*_{tag}.npy ({len(X0)} pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tw", "adp", "all"])
    ap.add_argument("--metad_ns", type=float, default=50.0)
    ap.add_argument("--max_per_cell", type=int, default=50)
    ap.add_argument("--procs", type=int, default=14)
    args = ap.parse_args()
    print(paths.summary())
    if args.what in ("tw", "all"):
        generate_triple_well()
    if args.what in ("adp", "all"):
        generate_adp(metad_ns=args.metad_ns, max_per_cell=args.max_per_cell, procs=args.procs)
