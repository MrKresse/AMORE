# -*- coding: utf-8 -*-
"""
run_benchmark.py — batch driver: train every (system, variant, seed) and cache to runs/.

Trains the five variants — three reversible baselines (ISA, GramSchmidt, SVD-Power) and
the two non-reversible ideas (Schur-ISA, GPCCA) — on the three systems:
    triple_well    reversible 2D triple-well        (benchmark_v3 data; shape vs committors)
    adp_300k_0p1   reversible alanine dipeptide      (benchmark_v4 data, 300 K, 0.1 ps)
    directed_ring  NON-REVERSIBLE 3-well ring        (cyclic; shape vs complex pair + fate AUROC)

Results are cached per run; re-running skips finished work. The notebook
(nonrev_benchmark.ipynb) loads the same cache and recomputes all tables/figures live.

    python run_benchmark.py                 # all systems
    python run_benchmark.py --only ring     # one system (ring|tw|adp)
"""
from __future__ import annotations
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_systems as ts
import train_eval as te

N_SEEDS = 3


class RingCfg(te.Cfg):
    MAX_ITER = 1000; MIN_ITER = 300; W = 200
    POWER_N_ITER = 60; POWER_EPOCHS = 50; NONREV_MAXITER = 60

class TWCfg(te.Cfg):
    MAX_ITER = 1200; MIN_ITER = 400; W = 250
    POWER_N_ITER = 80; POWER_EPOCHS = 50; NONREV_MAXITER = 60

class ADPCfg(te.Cfg):
    # v4-faithful: 25k anchors / 2500 iters so the reversible baselines (GramSchmidt,
    # SVD-Power) reach full strength (GramSchmidt φ r~0.97).
    # WARMUP=1200 (not the 100 default): the 1-D ShiftScale committor on 231-D ADP is badly
    # under-converged at 100 full-batch steps (φ doesn't emerge until ~600-1200), which is
    # what made ISA look randomly fragile. At 1200 the committor locks onto φ and ISA is
    # stable across all seeds (k_eff=3). 2-D TW/ring converge at 100, so they keep the default.
    MAX_ITER = 2500; MIN_ITER = 400; W = 250; WARMUP = 1200
    POWER_N_ITER = 80; POWER_EPOCHS = 50; NONREV_MAXITER = 40


def build_systems(which):
    out = {}
    if which in ("all", "ring"):
        out["directed_ring"] = (ts.simulate_directed_ring(n_anchors=2500, K=8, burst=250), RingCfg)
    if which in ("all", "tw"):
        out["triple_well"] = (ts.load_triple_well(), TWCfg)
    if which in ("all", "adp"):
        out["adp_300k_0p1"] = (ts.load_adp_300k_0p1(max_anchors=25000), ADPCfg)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "ring", "tw", "adp"])
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    t_start = time.perf_counter()
    systems = build_systems(args.only)
    for tag, (system, cfg) in systems.items():
        print(f"\n{'='*70}\nSYSTEM: {tag}  (reversible={system['reversible']})  "
              f"N={len(system['feat'])} F={system['feat'].shape[1]}\n{'='*70}", flush=True)
        for variant in te.VARIANTS:
            print(f"\n-- {te.LABELS[variant]} --", flush=True)
            for seed in range(args.seeds):
                te.train_chi(system, variant, seed, cfg=cfg, verbose=True)
    print(f"\nALL DONE in {(time.perf_counter()-t_start)/60:.1f} min", flush=True)
