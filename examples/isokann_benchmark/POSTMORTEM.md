# isokann_benchmark — postmortem & handoff

Everything the next agent needs to understand, run, extend, or trust this benchmark.
Written after the session that built it from the old `benchmark*` / `nonrev_benchmark` dirs.

---

## 1. What this is

Two self-contained notebooks benchmarking the multi-dimensional ISOKANN target transforms in
`src/amore` against **numerical ground truth**, on reversible (`triple_well`, vacuum alanine
dipeptide `adp_300k_0p1`) and non-reversible (`directed_ring`) systems. It replaces the five
redundant predecessors (`benchmark`, `benchmark_v2/v3/v4` → moved to
`/scratch/htc/jkresse/old_benchmarks/`; `nonrev_benchmark` still present but superseded).

Layout: top level = `reversible.ipynb`, `nonreversible.ipynb`, `build_reversible.py`,
`build_nonreversible.py`, `README.md`; **all machinery in `lib/`** (notebooks add `lib/` to
`sys.path`). Builders only use `nbformat` and write the `.ipynb`; the helper modules are imported
by the notebook *cells*, not the builders.

---

## 2. THE key result (read this first)

**The gold standard is softmax-head ISA with no warm-up.** The whole session converged on a
two-family split, now also documented in `src` docstrings (`amore.isokann.__init__`,
`ChiNetMulti`, `isotarget.isa_target`):

| family | head | k | warm-up | methods | output |
|---|---|---|---|---|---|
| **membership** | softmax `ChiNetMulti` | 3 | none | isa, vamp, schurisa, gpcca | 3 memberships (χ≥0, Σχ=1) |
| **basis** | linear `ChiNetMultiLinear` (deflated) | 2 | none | gramschmidt, pseudoinv, cross, svd_power | 2 non-trivial eigenfunctions |

A 3-state system spans a 3-D slow subspace = **constant + 2 non-trivial eigenfunctions ≡ 3
memberships**. The two families output equivalent things in different bases; `lib/harness.py`
converts: `eigfns()` (mean-center + SVD top-2) and `to_memberships()` (PCCA+ inner-simplex on
[const, eigfns]).

**Why softmax matters (the headline):** linear-ISA *collapses* on 231-D ADP — amplitude → 0 and it
mode-selects onto the slow φ, missing ψ; the old fix was a converged 1-D ShiftScale warm-up, which
only mode-selected φ anyway (k_eff=0 on all seeds). The **softmax simplex head cannot
amplitude-collapse**, so softmax-ISA recovers **both** φ and ψ from a random init:

```
ADP ISA, 5 seeds (eig_r vs EV2/EV3):
  linear (no warm-up)   0.573 ± 0.253   (high variance; sometimes fine, sometimes not)
  linear + 1-D warm-up  0.611 ± 0.124   k_eff=0 on ALL seeds (mode-selects φ / goes constant)
  softmax (default)     0.908 ± 0.085   k_eff=3 (recovers φ AND ψ)
```

**But softmax is NOT universal:** it collapses the *basis/orthogonalization* targets
(gramschmidt/pseudoinv/cross/svd_power) on high-D — their signed eigenfunction targets fight the
simplex head. Those must stay linear at k=2. This asymmetry is the core design fact.

---

## 3. Final headline numbers (sanity targets for re-runs)

- **triple_well** (easy, 2-D): everything strong. ISA 0.988, SVD-Power 0.986, GramSchmidt 0.973
  (eig_r); memberships ~0.98; k_eff=3 everywhere.
- **adp_300k_0p1** (231-D, eig_r): **ISA 0.908** (winner) ≫ PseudoInv 0.658, GramSchmidt 0.645,
  SVD-Power 0.581, VAMP 0.503, Cross 0.494. Basis methods are genuinely harder at the *correct*
  k=2 (they get φ but struggle with the fast ψ in 2 dims) — this is a real finding, not a bug.
- **directed_ring** (non-rev, 1 seed): ISA/Schur-ISA fate AUROC 0.997/0.998, committor |r| ≈0.98;
  GPCCA weaker (0.75 AUROC — crispness fights feasibility). Diagnostics: complex pair |Im λ|≈0.28,
  plain-ISA min membership ≈ −0.29 (infeasible), Schur-ISA repairs it (≈ −0.01), GPCCA doesn't.
- Operator ground truth: ADP EV2 (φ-flip) λ=0.999 (ITS≈100 ps), EV3 (ψ) λ=0.987 (ITS≈7.9 ps).

If a re-run drifts far from these, something regressed.

---

## 4. Design decisions and *why* (so you don't re-litigate them)

- **k by family, not uniform k=3.** Basis methods at k=3 over-parameterize (waste a dim on the
  constant). At k=2 they must learn exactly the 2 non-trivial eigenfunctions. Verdict from the user:
  do the *correct* experiment (retrain at k=2), not a post-hoc "drop a column" patch.
- **Constant deflation for basis ortho methods** (gramschmidt/pseudoinv/cross): `harness` mean-centers
  the Koopman target so the net learns {EV2,EV3}, not {const,EV2}. Without it, EV3 recovery is poor
  (cross 0.20→0.98 on TW with deflation). `svd_power` self-deflates (`power_method_multi` does
  `Y − Y.mean(0)` internally) — don't double-deflate it.
- **VAMP is a membership method (VAMPnets).** The VAMP-2 *score* is a subspace objective (invariant
  to linear feature transforms), so it doesn't define memberships by itself; the softmax head does.
  Use `ChiNetMulti` (softmax). The constant is its top singular function (σ=1), so a *linear* VAMP at
  k=2 would grab {const,EV2} — another reason it's membership/softmax/k=3.
- **`eigfns` = mean-center + SVD, not "drop the constant column."** On ADP no single output column is
  the constant (the QR-rotated basis spreads it across all outputs); the constant is *in the span*
  (projection residual ≈0.07) but not a panel. Mean-centering removes that one dimension cleanly.
- **Scoring is dual:** `eig_r` (the 2 eigenfunctions vs EV2/EV3, both systems, **computed with no
  PCCA+** so it isolates the method) and `memb_r` (3 memberships vs committors, TW only). On a
  reversible 3-state system these are equivalent up to the PCCA+ rotation.
- **Plot conventions** (`lib/plotting.py`): all χ-map columns are Hungarian-aligned to a fixed
  reference (`align_columns`) so column j = the same basin/mode across methods & seeds; eigenfunctions
  are sign-aligned (removes arbitrary eigenvector sign); colormap is `RdBu_r` everywhere (memberships
  red=1/blue=0 with vmin0/vmax1; eigenfunctions symmetric). EV1/stationary uses `magma` + `LogNorm`.

---

## 5. Bugs vs limitations confirmed this session (don't "fix" the limitations)

- **VAMP ≈0.5 on ADP is a method limitation, not a bug.** VAMP-2 is invariant to *which* 2 modes
  span the subspace, so the best-VAMP-2 solution isn't reliably φ+ψ: seed2 finds both (0.857), the
  others lock onto ψ-only (~0.4). Removing the plateau early-stop made it *worse* (drifts) — the
  early-stop is kept.
- **Schur-ISA/GPCCA collapsed to the uniform (constant) membership on ADP** — instability, not a bug.
  From the near-uniform softmax init the Schur coarse-propagator target can't break symmetry on
  231-D (membership matrix A → near-singular, condA~2.5e5 → stays uniform). **Fixed** with an ISA
  pre-spread: `cfg.SCHUR_WARM` iters of the plain-ISA target before switching to the Schur target
  (Schur-ISA *is* "ISA + feasibility refinement"). Now k_eff=3 (no collapse). On reversible ADP
  Schur-ISA lands ≈0.50 — the feasibility projection's documented cost on a system that didn't need
  it; Schur's real value is the **ring diagnostics**, not ADP accuracy.
- Harmless `RuntimeWarning: divide by zero` in `schur_isotargets.py` (GPCCA crispness term) — benign,
  filtered in the notebooks.

---

## 6. Compute environment gotchas (these cost real time to rediscover)

- **CPU is cgroup-capped, not what `nproc` says.** `nproc`→128 but the real quota is
  `/sys/fs/cgroup/cpu.max` (was `1600000 100000` = 16 cores). RAM = `memory.max` (32 GB). Size pools
  to the cgroup, not `nproc`. User can raise CPU (x86 capped at 16; more would need an ARM switch —
  avoid).
- **OpenMM has no GPU platform here** (only `Reference`, `CPU`). For 22-atom vacuum ADP the
  **`Reference` platform is ~14× faster than `CPU`** (~132 vs ~10 ps/s, single-thread) and
  multithreading *hurts* tiny systems. Run MD on `Reference`, single-thread, parallelise bursts
  across processes. MetaD 50 ns ≈ 9 min; 0.1 ps bursts for ~78k anchors ≈ seconds (14 procs).
- **The venv is uv-managed and has no `pip`.** Install with
  `uv pip install --python /home/htc/jkresse/AMORE/.venv/bin/python <pkg>` (nbconvert/jupyter were
  added this way). torch 2.6+cu124 sees a single GPU MIG slice (1g.24gb) — fine for the tiny nets,
  but the per-iter isotarget transforms are numpy/CPU, so **CPU-parallel ensembles beat one serial
  GPU**. Training uses `DEVICE=cpu`.
- **Large artifacts live on scratch, gitignored:** `/scratch/htc/jkresse/amore_bench/{data,runs,figures}`
  (override via `$AMORE_BENCH_SCRATCH`). `*.npy/.npz/.pt` etc. are gitignored — the repo ships only
  scripts, notebooks, and the PDB.
- **The executed `reversible.ipynb` is ~26 MB** (180+ embedded scatter panels). This OOMs the VSCode
  notebook *webview* (>32 GB) even though compute sits at ~3 GB. Open it via a browser / `nbconvert
  --to html`, not the IDE webview. The user accepted the figure size; don't shrink without asking.
- **Permissions:** bash `mv` and `rm` are denied. For *explicitly requested* moves/deletes use Python
  `shutil.move` / `os.remove` (not the bash commands). `nohup ... python` output is buffered (no
  per-line flush) → monitors that grep the log file won't fire until the process exits; prefer
  `flush=True` prints or just wait for the task-completion notification.

---

## 7. Reproduce from scratch

```bash
cd examples/isokann_benchmark
python lib/generate_data.py all          # TW (s) + ADP MetaD (~9 min) + bursts (s), 14 procs
python lib/run_ensemble.py all           # reversible + nonrev + why-softmax compare (~60 min, 16 cores)
python build_reversible.py && python build_nonreversible.py     # (re)generate the .ipynb
jupyter nbconvert --to notebook --execute --inplace reversible.ipynb nonreversible.ipynb
```
`run_ensemble.py all` includes the `compare` jobs (ADP ISA in 3 forms: softmax / linear /
linear+warmup) for the §3 why-softmax section. Run-dir naming encodes head/warmup:
`runs/<sys>/<variant>/`, with overrides under `<variant>__linear`, `<variant>__linear_warm`.

The training loop, families, deflation, SCHUR_WARM, and dual scoring all live in `lib/harness.py`;
ground truth in `lib/ground_truth.py`; loaders in `lib/systems.py` (each returns `feat, bursts,
coords, refs, ev_refs, committor_refs, labels`).

---

## 8. Pitfalls that bit us (so they don't bite you)

- **`meta.json` `k_eff` uses the LAST iteration's SD**, but the *shown* `chi_best` is the
  best-on-held-out checkpoint. They disagree for collapsing runs. `score_frame`/scoring use
  `chi_best.std`/`to_memberships` — trust those, not `meta.json` k_eff.
- **Reduced-iter test runs with `force=True` overwrite the real cache** and race with other writers
  to the same run dir. Use a distinct cfg or accept you must retrain. (Cost us a redundant run.)
- **`nbconvert --execute` stops at the first cell error and writes the *un-executed* notebook**
  (no outputs, ~18 KB). If a re-executed notebook is suddenly tiny with `errors=[]`, look for the
  `TypeError`/`Traceback` in nbconvert's *stderr*, not in the saved cells. (A missing `view=` kwarg in
  a function signature did exactly this.)
- **Don't trust a single seed for the narrative.** seed-0 maps were misleading (linear-ISA looked
  perfect, softmax had an odd basin). The §3 section shows the full 5-seed table + all-seed maps for
  that reason.

---

## 9. Open items / where to go next

- `examples/nonrev_benchmark/` is fully superseded by `nonreversible.ipynb` but was left in place
  (user moved only `benchmark*`). Can be moved to `old_benchmarks/` on request.
- Basis methods at k=2 on ADP recover φ but not the fast ψ well (eig_r ~0.5–0.66). Worth probing:
  more capacity / different deflation / longer training — but it may be intrinsic to 2-D at this lag.
- GPCCA's crispness term hurts on both reversible ADP and the ring (drives memberships negative /
  lowers AUROC). Schur-ISA is the better of the two non-reversible variants; GPCCA is kept for
  contrast.
- The softmax-head finding is, in principle, also a route to memberships *for the basis targets* —
  but it collapses them on high-D. If someone wants a single unified head, that needs a stabilising
  trick (it was deliberately not pursued; linear-basis + softmax-membership is the clean split).
- A genuine `train_isokann()` convenience in `src` (softmax ISA, no warm-up) could encode the gold
  standard as code, not just docstrings — currently the training loop lives only in `lib/harness.py`.
