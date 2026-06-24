# -*- coding: utf-8 -*-
"""
paths.py — central location config for the consolidated ISOKANN benchmark.

Large artifacts (MD trajectories, Koopman pairs, trained chi maps, model
checkpoints) live on scratch, NOT in the git tree:

    $AMORE_BENCH_SCRATCH                 (env override)
    else  /scratch/<user>/amore_bench    (default; falls back to ./_scratch)

Layout under the scratch root:
    data/    generated systems (triple_well_koopman.npz, vac_*_T300_0p1.npy, ring_*.npz)
    runs/    per (system, variant, seed) training caches (chi_best.npy, val_loss.npy, ...)
    figures/ optional cached figures

Only ./<helpers>.py and the two notebooks live in the git tree; everything they
read/write of any size goes through these paths.
"""
from __future__ import annotations
import os
import getpass

# this file lives in examples/isokann_benchmark/lib/ -> repo root is three levels up
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AMORE_SRC = os.path.join(REPO, "src")


def _default_scratch() -> str:
    env = os.environ.get("AMORE_BENCH_SCRATCH")
    if env:
        return env
    user = getpass.getuser()
    for cand in (f"/scratch/htc/{user}/amore_bench", f"/scratch/{user}/amore_bench"):
        parent = os.path.dirname(cand)
        if os.path.isdir(parent) or os.path.isdir(cand):
            return cand
    return os.path.join(HERE, "_scratch")   # last-resort local fallback


SCRATCH = _default_scratch()
DATA = os.path.join(SCRATCH, "data")
RUNS = os.path.join(SCRATCH, "runs")
FIGURES = os.path.join(SCRATCH, "figures")

for _d in (DATA, RUNS, FIGURES):
    os.makedirs(_d, exist_ok=True)

# the bundled alanine-dipeptide PDB shipped with the repo
DEFAULT_PDB = os.path.join(REPO, "data", "alanine-dipeptide-nowater.pdb")


def summary() -> str:
    return (f"scratch root : {SCRATCH}\n"
            f"  data       : {DATA}\n"
            f"  runs       : {RUNS}\n"
            f"  figures    : {FIGURES}")


if __name__ == "__main__":
    print(summary())
