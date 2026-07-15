# -*- coding: utf-8 -*-
"""
sysbuild.py -- build & serialize the OpenMM System for 2cm2 pose2 (Phase 0).

Run with fsafarov's `cuda124` conda env (has openmmforcefields/openff/AmberTools for the
KB8 ligand's GAFF template; the AMORE .venv has neither):

    /scratch/htc/fsafarov/conda/envs/cuda124/bin/python examples/2cm2_full/lib/sysbuild.py

The production trajectory (`generate_initial_trajectory_pose2_1.py`) used `constraints=HBonds`
with OpenMM's default `rigidWater=True`. We build that same "rigid" system here ONLY as a
comparison point for the Phase-0 stability demo -- the system actually used for chi-MEP work is
the "flexible" one (`constraints=None, rigidWater=False`), because the projected-Langevin
integrator (`amore.mep.constrained.sample_levelset_projected`) moves positions by hand every step
and never calls OpenMM's own constraint solver (SETTLE/RATTLE). Rigid bonds/angles carry no
bonded potential term (the geometry is held by the solver alone), so under our uncorrected
manual updates they have nothing to hold them together and blow up.

Topology + starting positions come straight from the existing solvated PDB (already equilibrated,
already has water+ions) -- no re-solvation via Modeller.addSolvent.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "2cm2", "lib"))
import data  # noqa: E402  (paths only; no torch/MDAnalysis needed from it)

from openmm import app, unit, XmlSerializer
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator

SMILES = "Cc1ccc2c(c1)c(cc(n2)C(F)(F)F)N3CCNCC3"     # KB8, from generate_initial_trajectory_pose2_1.py
OUT_DIR = os.environ.get("CM2_MFEP_SCRATCH", "/scratch/htc/jkresse/2cm2_mfep")


def build_forcefield():
    # antechamber/parmchk2 (AmberTools, needed by GAFFTemplateGenerator) live in the cuda124
    # env's own bin/, but that's not on PATH unless the env is `conda activate`d -- invoking
    # the interpreter by full path (as we do) skips that.  Mirror
    # generate_initial_trajectory_pose2_1.py's own PATH prepend.
    env_bin = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "bin")
    if env_bin not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ["PATH"]
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")
    molecule = Molecule.from_smiles(SMILES)
    molecule.assign_partial_charges("gasteiger")
    gaff = GAFFTemplateGenerator(molecules=[molecule], forcefield="gaff-2.2.20")
    ff.registerTemplateGenerator(gaff.generator)
    return ff


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pdb_path = data.pdb_path()
    print(f"[sysbuild] loading {pdb_path}", flush=True)
    pdb = app.PDBFile(pdb_path)
    print(f"[sysbuild] {pdb.topology.getNumAtoms()} atoms, box "
          f"{pdb.topology.getPeriodicBoxVectors()}", flush=True)

    ff = build_forcefield()

    t0 = time.time()
    print("[sysbuild] building RIGID system (constraints=HBonds, rigidWater=True; "
          "mirrors the production trajectory -- comparison point only) ...", flush=True)
    system_rigid = ff.createSystem(
        pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds, rigidWater=True,
    )
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    print("[sysbuild] building FLEXIBLE system (constraints=None, rigidWater=False; "
          "used for all chi-MEP work) ...", flush=True)
    system_flex = ff.createSystem(
        pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None, rigidWater=False,
    )
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(OUT_DIR, "system_rigid.xml"), "w") as f:
        f.write(XmlSerializer.serialize(system_rigid))
    with open(os.path.join(OUT_DIR, "system_flex.xml"), "w") as f:
        f.write(XmlSerializer.serialize(system_flex))
    # topology travels as the PDB itself (already on scratch); record its path for downstream
    with open(os.path.join(OUT_DIR, "topology_pdb_path.txt"), "w") as f:
        f.write(pdb_path)

    print(f"[sysbuild] wrote system_rigid.xml, system_flex.xml -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
