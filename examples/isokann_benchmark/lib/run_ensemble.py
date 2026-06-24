# -*- coding: utf-8 -*-
"""
run_ensemble.py — train every (system, variant, seed) in parallel and cache to
paths.RUNS. Both notebooks read the same cache; re-running skips finished work.

New-design families (see harness.py): membership variants (isa, vamp, schurisa, gpcca)
use a softmax head at k=3; basis variants (gramschmidt, pseudoinv, cross, svd_power) use
a linear head at k=2 (constant-deflated). No warm-up in the main pipeline.

  python run_ensemble.py reversible    # TW + ADP, 6 methods, 5 seeds
  python run_ensemble.py nonrev        # adds Schur-ISA/GPCCA + directed_ring
  python run_ensemble.py compare       # ADP ISA: softmax vs linear vs linear+warmup (why-softmax section)
  python run_ensemble.py all
Options: --seeds N  --procs P  --force
"""
from __future__ import annotations
import os, sys, time, argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

REVERSIBLE_VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd_power", "vamp"]
NONREV_EXTRA = ["schurisa", "gpcca"]

SYSTEMS = {}


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def _train_job(args):
    tag, variant, seed, force, head, warmup = args
    import harness
    t = time.perf_counter()
    res = harness.train_chi(SYSTEMS[tag], variant, seed, use_gpu=False, force=force,
                            verbose=False, head=head, warmup=warmup)
    return (tag, variant, seed, head, warmup, res.get("n_iter", 0), res.get("k_eff", 0),
            float(time.perf_counter() - t), bool(res.get("cached", False)))


def build_jobs(which, seeds):
    import systems
    jobs = []          # (tag, variant, seed, head, warmup)
    def add_rev():
        SYSTEMS.setdefault("triple_well", systems.load_triple_well())
        SYSTEMS.setdefault("adp_300k_0p1", systems.load_adp_300k_0p1(max_anchors=25000))
    if which in ("reversible", "all"):
        add_rev()
        for tag in ("triple_well", "adp_300k_0p1"):
            for v in REVERSIBLE_VARIANTS:
                for s in range(seeds):
                    jobs.append((tag, v, s, None, False))
    if which in ("nonrev", "all"):
        add_rev()
        SYSTEMS["directed_ring"] = systems.simulate_directed_ring(n_anchors=2500, K=8, burst=250)
        for tag in ("triple_well", "adp_300k_0p1"):
            for v in NONREV_EXTRA:
                jobs.append((tag, v, 0, None, False))
        for v in REVERSIBLE_VARIANTS + NONREV_EXTRA:
            if v == "vamp":
                continue
            jobs.append(("directed_ring", v, 0, None, False))
    if which in ("compare", "all"):
        # why-softmax: ADP ISA in three forms across seeds
        SYSTEMS.setdefault("adp_300k_0p1", systems.load_adp_300k_0p1(max_anchors=25000))
        for s in range(seeds):
            jobs.append(("adp_300k_0p1", "isa", s, None, False))         # softmax (default)
            jobs.append(("adp_300k_0p1", "isa", s, "linear", False))     # linear, no warm-up
            jobs.append(("adp_300k_0p1", "isa", s, "linear", True))      # linear + 1-D warm-up
    seen = set(); uniq = []
    for j in jobs:
        if j not in seen:
            seen.add(j); uniq.append(j)
    return uniq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["reversible", "nonrev", "compare", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(paths.summary())
    jobs = build_jobs(args.which, args.seeds)
    print(f"{len(jobs)} train jobs, procs={args.procs}")
    t0 = time.perf_counter()
    targs = [(t, v, s, args.force, h, w) for (t, v, s, h, w) in jobs]
    with mp.Pool(args.procs, initializer=_init_worker) as pool:
        done = 0
        for (tag, v, s, h, w, ni, ke, dt, cached) in pool.imap_unordered(_train_job, targs):
            done += 1
            lab = f"{v}{'/'+h if h else ''}{'+warm' if w else ''}"
            print(f"  [{done}/{len(targs)}] {tag}/{lab}/seed{s} iters={ni} k_eff={ke} "
                  f"{'[cache]' if cached else f'{dt:.0f}s'}", flush=True)
    print(f"DONE {args.which} in {(time.perf_counter()-t0)/60:.1f} min")
