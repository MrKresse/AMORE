# -*- coding: utf-8 -*-
"""
pipeline.py -- helpers for the AMORE openmmtorch benchmark notebook.

Compares `amore.mep.constrained.sample_levelset_projected` (the existing constrained-MFEP
sampler: plain Python/PyTorch autograd chi evaluation, one host<->device round trip per
op) against `sample_levelset_projected_torchforce` (chi's value/gradient fused into
OpenMM's own force evaluation via a CustomCVForce+TorchForce in a dedicated force group --
see `build_cv_gradient_system`) on two systems:

  - alanine dipeptide in vacuum -- correctness + first performance check (small, cheap,
    catches export/tracing bugs fast)
  - 2cm2's flexible-constraint pocket system (`examples/2cm2_full`) -- the real
    performance benchmark (50275 atoms, PME, the trained pocket chi model)

Must run under the `omtorch` micromamba env (`openmm-torch` is not importable in the
AMORE `.venv`, nor in fsafarov's `molsim` env -- see POSTMORTEM.md for why and how
`omtorch` was built).  No reimplementation of the level-set integrators themselves --
those live in `amore.mep`.
"""
from __future__ import annotations
import os
import sys
import time
import getpass

import numpy as np
import torch as pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "2cm2_full", "lib"))

import openmm as mm
from openmm import app, unit

from amore.sims import OpenMMSimulation
from amore.sims.openmm_sim import OpenMMSimulator
from amore.isokann import ChiNetMulti
from amore.features import make_featurizer
from amore.mep.core import _chi_and_grad
from amore.mep.constrained import (
    export_cv_torchscript,
    export_cv_torchscript_module,
    build_cv_gradient_system,
    sample_levelset_projected,
    sample_levelset_projected_torchforce,
    _minimize_seed,
)

import comfeat  # noqa: E402  (examples/2cm2_full/lib -- pocket featurizer/model loader)

USER = getpass.getuser()


class FaceCV(pt.nn.Module):
    """s = chi_i -- membership i vs the rest.  Local copy of `amore.mep.simplex.FaceCV`:
    that module also pulls in the projected/restrained-sampling entry points we don't
    need here, so a two-line duplicate is simpler than importing it."""

    def __init__(self, model, i):
        super().__init__()
        self.model, self.i = model, int(i)

    def forward(self, feats):
        return self.model(feats)[..., self.i:self.i + 1]


# ─────────────────────────── ADP (vacuum) setup ──────────────────────────────
N_ADP_ATOMS = 22
ADP_PAIRS = np.array([(i, j) for i in range(N_ADP_ATOMS) for j in range(i + 1, N_ADP_ATOMS)])


def adp_featurizer():
    return make_featurizer(ADP_PAIRS)


def load_adp_model(k=3, device="cpu", seed=0):
    """Load the pathway_benchmark's cached k=3 softmax ISOKANN model if present (a real
    trained CV, consistent with the rest of the repo's ADP work); else fall back to a
    randomly-initialised net of the same architecture.  A random net is fine for a pure
    correctness/timing check -- it does not depend on the CV being physically meaningful,
    only on chi/grad(chi) being a real, nonlinear function of the coordinates."""
    ckpt = f"/scratch/htc/{USER}/amore_pathway/isokann_k3.pt"
    net = ChiNetMulti(len(ADP_PAIRS), k, hidden=[128, 32, 8]).to(device)
    if k == 3 and os.path.exists(ckpt):
        net.load_state_dict(pt.load(ckpt, map_location=device))
        print(f"loaded trained ADP ISOKANN model from {ckpt}")
    else:
        pt.manual_seed(seed)
        print("no cached ADP model found -- using a randomly-initialised net "
              "(fine for a correctness/timing check, not a physically meaningful CV)")
    net.eval()
    return net


def adp_sim(dt=2e-3, temp=300.0, platform="gpu"):
    return OpenMMSimulation(steps=1, dt=dt, temp=temp, platform=platform)


# ─────────────────────────── 2cm2 pocket setup ────────────────────────────────
def load_pocket_model(k=3, device="cuda"):
    return comfeat.load_trained_model_pocket(k, device=device)


def pocket_featurizer(device="cuda"):
    return comfeat.make_torch_featurizer_pocket(device=device)


def pocket_system_and_pdb(scratch=None):
    scratch = scratch or f"/scratch/htc/{USER}/2cm2_mfep"
    with open(os.path.join(scratch, "system_flex.xml")) as f:
        system_xml = f.read()
    with open(os.path.join(scratch, "topology_pdb_path.txt")) as f:
        pdb_path = f.read().strip()
    pdb = app.PDBFile(pdb_path)
    return system_xml, pdb


# ─────────────────────── shared: build an OpenMMSimulator from a System ───────
def make_simulator(system, topology, dt, temp, friction=1.0, platform_name="CUDA"):
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, friction / unit.picosecond,
                                        dt * unit.picoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    pysim = app.Simulation(topology, system, integ, platform, {"Precision": "mixed"})
    return OpenMMSimulator(pysim, steps=1, temp=temp, dt=dt)


# ─────────────────────── old-vs-new comparison driver ─────────────────────────
def compare_samplers(sim_old, nu, featurizer, sim_new, cv_force_index, grad_force_group,
                     x0, chi_level, steps, burnin=0, seed=0):
    """Run `sample_levelset_projected` (old) then `sample_levelset_projected_torchforce`
    (new) from the SAME seed positions with the SAME numpy random draws (same
    `np.random.seed` before each), and compare.  Returns a dict with both raw results,
    wall-clock totals, the per-step timing breakdown each sampler already returns, and
    the position-trajectory / mean-force agreement (the correctness check)."""
    sim_old.set_positions(x0)
    np.random.seed(seed)
    t0 = time.perf_counter()
    res_old = sample_levelset_projected(sim_old, nu, featurizer, x0, chi_level, steps,
                                        burnin=burnin, time_breakdown=True)
    t_old = time.perf_counter() - t0

    sim_new.set_positions(x0)
    np.random.seed(seed)
    t0 = time.perf_counter()
    res_new = sample_levelset_projected_torchforce(sim_new, cv_force_index, grad_force_group,
                                                    x0, chi_level, steps, burnin=burnin,
                                                    time_breakdown=True)
    t_new = time.perf_counter() - t0

    pos_diff = np.max(np.abs(res_old["positions"] - res_new["positions"]))
    lam_diff = np.max(np.abs(res_old["lambdas"] - res_new["lambdas"]))
    return dict(
        res_old=res_old, res_new=res_new,
        t_old=t_old, t_new=t_new,
        ms_step_old=t_old / steps * 1e3, ms_step_new=t_new / steps * 1e3,
        speedup=t_old / t_new,
        pos_max_diff=float(pos_diff), lambda_max_diff=float(lam_diff),
    )


# ─────────────────────────────── plotting ──────────────────────────────────
def plot_timing_breakdown(ax, result, title):
    """Stacked bar chart: OpenMM / chi-grad / retract, ms per step, old vs new."""
    labels = ["old\n(python autograd)", "new\n(TorchForce-fused)"]
    parts = ["openmm", "chi_grad", "retract"]
    colors = {"openmm": "#4C72B0", "chi_grad": "#DD8452", "retract": "#55A868"}
    old_t = result["res_old"]["timing"]
    new_t = result["res_new"]["timing"]
    bottom = np.zeros(2)
    for p in parts:
        vals = np.array([old_t[p], new_t[p]]) * 1e3
        ax.bar(labels, vals, bottom=bottom, label=p, color=colors[p])
        bottom += vals
    ax.set_ylabel("ms / step")
    ax.set_title(title)
    ax.legend(fontsize=8)
