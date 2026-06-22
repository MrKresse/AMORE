# MEF reprogramming benchmark — postmortem

ISOKANN+AMORE vs CellRank 2 / GPCCA on the Schiebinger MEF→iPSC reprogramming dataset
(CR2 RealTimeKernel "mef" analysis), built as the exact analogue of the pharyngeal-endoderm
benchmark. This document records what was built, the decisions, and — more usefully — the
mistakes and the fixes, so the next person doesn't repeat them.

## 1. What was delivered

- `realtime_kernel/{pharyngeal,mef}/` reorganisation (matching CR2's repo layout); the
  pharyngeal benchmark moved intact, `cellrank2_reproducibility/` kept shared at `cr2_benchmark/`.
- `mef/` pipeline: `config.py` → `01_cr2_reproduce.py` (+ `_sparse_schur_patch.py`) → `02_cbc.py`
  (cr2-py310 env) → `03_train_armK.py` / `03b_continue_armK.py` / `04_train_armK_hvg.py` (amore env)
  → `mef_isokann_benchmark.ipynb` (primary deliverable, k=6) + `mef_k4_vs_k6_comparison.ipynb`
  (built by `_make_compare_nb.py`).
- **Result (k=6, converged):** GPCCA reproduces CR2 exactly (TSI=1.0, all four purities=1.0);
  ISOKANN edges GPCCA on mean cell-fate AUROC **0.995 vs 0.982** (beats on Stromal 0.987 vs 0.937 and
  Trophoblast 0.998 vs 0.996, ties Neural 1.000, a hair behind on IPS 0.994 vs 0.997); drivers
  competitive-to-better (corr-χ recAUC 0.550 > GPCCA 0.525; every ISOKANN readout @100 = 7 of 10 vs 6).

## 2. Data acquisition — figshare AWS-WAF

CR2's `reprogramming_schiebinger(subset_to_serum=True)` ships the serum subset (165 892×19 089) **with
the published RealTimeKernel transition matrix in `obsp['transition_matrix']`**, the force-directed
embedding, and `var['highly_variable']` (1479 HVGs) — so no WOT recompute is needed.

- **Trap:** figshare blocks scripted downloads with an AWS-WAF JS challenge (HTTP 202,
  `x-amzn-waf-action: challenge`; curl/urllib get the challenge HTML → 0-byte file). The empty
  placeholder must be deleted or it blocks the loader's read.
- **Fix:** `pip install selenium`, drive **headless Chrome** (already installed) to the ndownloader URL;
  Chrome solves the JS challenge and downloads the 3.85 GB h5ad. `Page.setDownloadBehavior` + poll for
  `.crdownload → final`, verify the `\x89HDF` magic.

## 3. Windows has no SLEPc/PETSc — the GPCCA backend

GPCCA on 165k cells needs a sparse Schur decomposition, which pyGPCCA only does via SLEPc (no Windows
build). Without it CellRank densifies → **205 GiB**. Fate probabilities have the same problem (scipy
"direct" densifies; gmres stalls on the hardest lineage).

- **Fix** (`_sparse_schur_patch.py`, imported at top of `01` after cellrank): (a) tell CellRank petsc is
  available so it keeps the sparse path; (b) replace `pygpcca.sorted_schur` with a SciPy-ARPACK partial
  real-Schur (top-m eigenpairs → real basis → QR → R=QᵀPQ; same `_check_conj_split`/`_check_schur`
  guards); (c) replace `_solve_lin_system` so SPARSE systems use a single SuperLU factorization for all
  RHS. All numerically the SLEPc result; Schur ~7 s, fate solve ~8 min.
- **Ordering gotcha:** emit the essential artifacts BEFORE the optional drivers+CBC — CBC over the
  91 797-cell `MEF/other` source is very slow (capped to 4000), and the notebook uses neither.

## 4. The central scientific lesson — the χ state count

This was the crux, and it took three tries to get right.

- **k=4 (wrong).** Naively set k = CR2's 4 terminal macrostates. Result: the column Hungarian-labelled
  "IPS" was in fact the **MEF source** — its top-2000 cells were 100% `MEF/other` — IPS fate AUROC only
  0.879, χ-binned driver recovery 1/10, and the ISA loss spiked. The user caught this from the χ map and
  the loss curve.
- **Why:** the dataset has **6** metastable sets (4 terminals + the huge non-metastable source + an
  Epithelial/MET intermediate). The Koopman spectrum is unambiguous: λ = [1.000, 0.998, 0.996, 0.995,
  0.991, 0.988] then a **0.025 gap** (the largest) before 0.963. ISOKANN's ISA produces a **partition-of-
  unity membership over every cell**, so with k=4 the source has nowhere to go but *into* a terminal
  committor. (GPCCA avoids this because it can leave the source as *transient* / unassigned; ISOKANN
  cannot.)
- **k=5 vs k=6 — the constant-mode subtlety (user-raised).** "There's one membership less than
  eigenfunctions, so the gap after 6 → k=5." That rule is right for **committor / reaction coordinates**
  (m metastable sets ↔ m−1 nontrivial slow eigenfunctions; the constant λ=1 carries no information). But
  amore's ISA produces PCCA+ **memberships**, and there #memberships = #dominant eigenvalues **including**
  the constant = m. Verified empirically: the k=6 χ **sum to 1** (partition of unity, row-sum 1.00±0.05),
  the constant *is* their sum (not a separate/wasted mode), all 6 columns are full-rank and peak on 6
  distinct populations (IPS, MEF/other, Neural, Trophoblast, Stromal, Epithelial), none constant. So
  **k=6**, then Hungarian-map the 4 terminals (the 2 extra columns = source + intermediate). GPCCA's own
  `n_states` count is the same convention (it includes the constant Schur vector).
- **Biology check.** 4 terminal states (IPS, Neural, Trophoblast, Stromal — late, day 14–17, low outflow);
  source = MEF/other (day 4); intermediates = MET, Epithelial (high outflow, transit-through). IPS **is**
  a terminal (late, absorbing, high self-fate), the opposite end from the MEF source.

## 5. Performance — why training was slow and CPU sat at 15–30%

The bottleneck was **`amore.isotarget._indexmap`** (the ISA simplex vertex search): a pure-Python
`for i in range(n)` loop over all 165 892 cells, run every ISA iteration. Single-threaded (GIL) → one
core saturated ≈ 6% of 16, bursts when torch ran → the 15–30% the user observed. ~4.3 s/call × hundreds
of calls dominated runtime.

- **Fix:** vectorised the inner-simplex search (project all row-differences onto the current basis at
  once; BLAS, multi-threaded). **Verified bit-for-bit identical** vertices on random + real kchi, **77×
  faster** per call (4.26 s → 0.055 s). Replaced in the shared library — benefits every ISOKANN training.
- **Caveat:** this only removed *half* the cost; the remaining floor is the torch eval/SGD loop on 165k
  cells (a full-dataset forward every iter), ~3–4 s/iter at ~2 torch threads. A GPU would help that half —
  but this box has an **AMD 7900 XTX** (no CUDA; torch-directml is the only Windows path and it conflicts
  with the env's torch 2.10), and the ISA half is CPU-Python regardless. Net: not worth a GPU here.

## 6. Convergence — the late loss spike and Neural purity

The user flagged a loss spike near iter 500 and Neural top-30 purity stuck at 0.80. `03b_continue_armK.py`
warm-starts from the cached net, runs 300 more ISA iters, and keeps the **best iterate** (lowest ISA loss
among non-collapsed steps) so it can't end on a spike.

- **Finding:** the iter-600 model was already at the loss floor (~3e-4); the extra iters bounced 1e-3–8e-3
  and the χ wandered along the *flat bottom* of the ISA landscape to a better point. Neural purity
  0.80→1.00, mean AUROC 0.990→0.995.
- **Whack-a-mole:** top-30-absolute-χ purity is brittle (30 cells); at any converged state one lineage's
  very top picks up a few boundary cells from an adjacent saturated mode, and it **shifts between lineages
  every continuation** (Neural 1.0→Stromal 0.83→0.73) while the AUROCs (all ≥0.987, mean 0.995) stay put. So
  we phrase purity robustly in the narrative ("~0.93, one lineage ~0.73–0.83, a brittle metric — AUROC is the
  robust read") rather than chasing the exact value, and **stopped continuing** once the AUROCs converged.
- **Loss curve.** Extended to ~750 iters via a short low-LR continuation so it honestly reflects the training
  past 600; the §4 caption notes the ISA loss is intrinsically noisy (vertex reassignment flips) and that the
  deployed χ is the **best iterate**, not the last step. (Two redundant §6 driver-recovery plots were collapsed
  to one after the naive-gradient line was dropped.)

## 7. Mistakes made (and the lessons)

1. **Overwrote the k=4 models without a backup** before retraining k=6 (03/04 write into the same
   `artifacts/armK*`). Reverting cost an hour of retraining. → Added an `ARMK_SUFFIX` env knob so a
   k-sweep writes to its own dir, and **always snapshot `artifacts/armK*` before any retrain**.
2. **Notebook read `chi.npy` mid-overwrite** — a notebook re-run launched while/just-after the continuation
   was saving produced a confusing mix of old/new numbers. → Only re-run notebooks once training has fully
   finished and the artifacts are stable.
3. **1472-vs-1479 bias-table bug.** After dropping the naive gradient I added `corr_ips = signed_corr(Xk,…)`
   (1472 filtered genes) but compared it via `spearmanr(corr_ips, lnorm)` against the full 1479-gene
   loading norm → `ValueError` that **silently aborted nbconvert** (file left stale; only caught because
   the file mtime was older than the model). → When mixing the filtered (`keep`) and full gene universes,
   index both consistently (`lnorm[keep]`), and always check the notebook actually wrote (mtime / "Writing
   bytes").
4. **HVG model collapses at k=6** (2 of 6 memberships die under the lighter HVG schedule) → its mean AUROC
   craters. Kept it only for parity with the pharyngeal analysis, with the caveat stated; the 50-PC model
   is the ISOKANN representative.
5. **conda-run buffers stdout on Windows** — long training runs show nothing until exit. Judge progress by
   process CPU/WS, not the output file. (Also: the bare `amore` python.exe fails `import numpy` with exit
   127 — missing DLLs on PATH — so always drive it via `conda run -n amore`.)

## 8. Reproduce

```
# CR2 side (cr2-py310 env): GPCCA baseline + shared artifacts
conda run -n cr2-py310 python 01_cr2_reproduce.py      # imports _sparse_schur_patch
# ISOKANN side (amore env): train, then converge
conda run -n amore python 03_train_armK.py             # k from config.K_CHI (=6)
conda run -n amore python 03b_continue_armK.py         # +300 iters, best-iterate
conda run -n amore python 04_train_armK_hvg.py         # HVG cross-check
# notebooks
conda run -n amore jupyter nbconvert --to notebook --execute --inplace mef_isokann_benchmark.ipynb
conda run -n amore python _make_compare_nb.py && \
  conda run -n amore jupyter nbconvert --to notebook --execute --inplace mef_k4_vs_k6_comparison.ipynb
```
k=4 baseline (for the comparison notebook): `K_CHI_OVERRIDE=4 ARMK_SUFFIX=_k4 conda run -n amore python 03_train_armK.py`
(and the same for `04`). Models: `armK` (primary k=6), `armK_k4` (k=4), `armK_k6`/`armK_hvg_k6` (backups).
