# isokann_benchmark — consolidated multi-D ISOKANN benchmark

Two self-contained notebooks that benchmark the multi-dimensional ISOKANN target
transforms in `src/amore` against **numerical ground truth**, sharing one helper layer
and one scratch cache. This supersedes the older redundant `benchmark/`,
`benchmark_v2/`, `benchmark_v3/`, `benchmark_v4/` and `nonrev_benchmark/` directories.

```
reversible.ipynb        reversible systems: triple_well + alanine dipeptide (0.1 ps)
                        6 methods × 5 seeds; eigenfunction↔membership bridge, why-softmax
                        comparison, dual-metric tables, χ maps, train/held-out loss curves
nonreversible.ipynb     Schur-ISA / GPCCA + the non-reversible directed_ring (committor +
                        cyclic-feasibility diagnostics + proof, 1 seed)
build_reversible.py     regenerates reversible.ipynb     (the two top-level builder scripts)
build_nonreversible.py  regenerates nonreversible.ipynb
README.md
lib/                    all machinery (imported by the notebooks via sys.path):
  paths.py              scratch location config (AMORE_BENCH_SCRATCH env override)
  generate_data.py      regenerate raw MD data (triple_well, ADP) -> scratch
  systems.py            uniform system loaders (feat, bursts, coords, refs, ev_refs, labels)
  ground_truth.py       TW committors+eigvecs, ADP operator EV2/EV3 + stationary, ring committor
  harness.py            families (membership softmax k=3 / basis linear k=2), training + scoring
  plotting.py           figure/table helpers (aligned columns, dual metrics)
  run_ensemble.py       parallel batch driver -> runs/ cache
  schur_isotargets.py, nonrev_targets.py, theory_demo.py   vendored non-reversible pieces
```

## Methods (all from `src/amore`, no reimplementation) — two families
A 3-state system has a 3-D slow subspace = constant + 2 non-trivial eigenfunctions ≡ 3
memberships. Each method outputs its natural representation:

- **membership family** — softmax head (`ChiNetMulti`), k=3 (3 memberships): `isa`, `vamp`
  (= VAMPnets), `schurisa`, `gpcca`. The softmax simplex head prevents the amplitude
  collapse / mode-selection the *linear* ISA suffered on 231-D ADP, so **no warm-up** is
  needed — softmax-ISA recovers both φ and ψ from scratch (eig|r|≈0.91 on ADP).
- **basis family** — linear head (`ChiNetMultiLinear`), k=2, constant-deflated: `gramschmidt`,
  `pseudoinv`, `cross`, `svd_power`. Output the 2 non-trivial eigenfunctions; **PCCA+**
  (`harness.to_memberships`) rotates {const, EV₂, EV₃} into 3 memberships.

Scoring: `eig|r|` (the 2 eigenfunctions vs EV₂/EV₃, both systems) and `memb|r|` (3 memberships
vs committors, triple-well). `harness.eigfns` / `harness.to_memberships` convert between views.
The legacy 1-D warm-up is kept only for the "why softmax" comparison section.

## Reproduce
```bash
# 1. raw data (TW seconds; ADP MetaD ~9 min single core + bursts seconds, 14 procs)
python lib/generate_data.py all
# 2. training ensembles (parallel; ~60 min on 16 cores for `all`)
python lib/run_ensemble.py all      # reversible + nonrev + the why-softmax comparison
# 3. (re)build + execute notebooks (loads the cache; retrains anything missing)
python build_reversible.py && python build_nonreversible.py
jupyter nbconvert --to notebook --execute --inplace reversible.ipynb nonreversible.ipynb
```
Large artifacts (MD coords, trained χ maps, checkpoints) live under
`$AMORE_BENCH_SCRATCH` (default `/scratch/<user>/amore_bench`), never in git.

## What is held constant vs varied
Constant per system: data/anchors/bursts, trunk arch in→[128,32,8] (Tanh hidden), Adam
lr 1e-3, grad-clip 5, training budget + plateau stop + best-on-held-out checkpoint, 80/20
split, Hungarian-|r|/SD/k_eff scoring, no warm-up. Varied: the method (family = head+target),
the seed (5), the system.
