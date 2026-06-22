# Human hematopoiesis benchmark — postmortem

ISOKANN+AMORE vs CellRank 2 / GPCCA on the **NeurIPS-2021 human bone-marrow hematopoiesis**
dataset (CR2 **PseudotimeKernel** "hematopoiesis" analysis), built as the exact analogue of the
MEF and pharyngeal-endoderm benchmarks. This records what was built, the decisions, and the
central scientific finding.

## 1. What was delivered

- `pseudotime_kernel/hematopoiesis/` pipeline (mirrors `realtime_kernel/mef/` one-for-one):
  `config.py` → `01_cr2_reproduce.py` (+ `_sparse_schur_patch.py`) (cr2-py310 env) →
  `03_train_armK.py` / `03b_continue_armK.py` / `04_train_armK_hvg.py` (amore env) →
  `hematopoiesis_isokann_benchmark.ipynb` (**primary deliverable, k=3** — reads `armK_k3`; switched from
  k=4 after §4b established the operator has 3 Perron λ=1 basins = GPCCA(3) = pDC/CD14+ Mono/Normoblast,
  so the 3 ISA committors land on exactly those; cDC2 has no committor and gets a dedicated
  "non-absorbing bridge into CD14+ Mono" section. 3-basin mean AUROC 0.993 vs GPCCA 0.992) +
  `hematopoiesis_k_sweep_comparison.ipynb` (k=3..7, built by `_build_compare_nb.py`). The main notebook
  drives its per-lineage panels off the *recovered* lineages (`LINEAGES` = terminals with a committor),
  uses a matching k=3 HVG model (`armK_hvg_k3`); `armK` stays the k=4 model so the sweep `DIRS` is intact.
  Notebook built by `_build_nb.py`, headline/verdict finalised by `_finalize_nb.py`; `_score.py`
  is a quick standalone scorer used to compare the k models.
- **The 9 sections + panels are identical to the MEF notebook**: §1.5 Koopman-spectrum panel,
  all-χ-modes panel, local χ-MEP + raw-medoid path panels, training-loss panel, cell-fate
  AUROC table + bar, driver bias table + per-lineage TF table + full recovery-vs-k table and
  curve, top-30 driver heatmaps, curated-markers-along-χ panel.

## 2. The data — figshare, no WAF this time

CR2's processed hematopoiesis data is figshare article **23739102** ("Human hematopoiesis - DPT
analysis"), file `gex_preprocessed.h5ad` (1.26 GB, 67568×25629, log-norm X, `layers['counts']`,
`obsm['MultiVI_latent']` (18-D), `var['hvg_multiVI']` (4000, a categorical "True"/"False" column),
`l2_cell_type`). **Unlike the MEF download, a plain `curl -L` with a browser User-Agent worked** —
the ndownloader URL 302-redirected straight to the S3 presigned link with no AWS-WAF JS challenge
(so the selenium/headless-Chrome workaround was *not* needed here; keep it in mind, it is
file/Time-dependent). The figshare **API** (`api.figshare.com/v2/collections/6843633/articles`) is
never WAF-gated and is the reliable way to find the article/file IDs. Human TF list =
Aerts-lab `allTFs_hg38.txt` (1892 TFs), saved to `data/generic/`.

## 3. Pipeline specifics (pinned from dpt.py)

Subset to the 9 DPT cell types → **24 440 cells** (much smaller than MEF's 165k, so everything is
fast: GPCCA + fate ≈ 1 min, each ISOKANN run ≈ 2–6 min). Neighbours on `MultiVI_latent`, diffmap
(15 comps), DPT root = HSC cell argmax `X_diffmap[:,5]`, `dpt(n_dcs=6)`, `PseudotimeKernel(...,
threshold_scheme="soft")`. GPCCA: `compute_schur(20)` → `compute_macrostates(6)` →
`set_terminal_states([pDC, cDC2, CD14+ Mono, Normoblast])` → `compute_fate_probabilities(tol=1e-7)`.
GPCCA reproduced CR2: TSI 0.909, terminal purity {pDC 1.0, cDC2 0.9, CD14+ Mono 1.0, Normoblast 1.0}.

- **ISOKANN features** = 50 PCs on the shipped 4000-HVG mask (the direct analogue of MEF's
  `var['highly_variable']` PCA); the PCs give the loadings for the gradient→gene driver chain rule.
  The PseudotimeKernel T (built on the MultiVI graph + DPT) is the shared operator (kchi = T@χ);
  features ≠ kernel-construction space, exactly as in the MEF/pharyngeal benchmarks.
- **The Windows sparse-Schur patch is still needed** even at 24k cells (dense Schur would be
  4.7 GB / O(n³)); the ARPACK partial-Schur + SuperLU-fate monkeypatch from MEF works unchanged.
- **Env wrinkle:** the `amore` env lives under **Miniforge3** (`C:\Users\kr3ss\miniforge3\envs\amore`),
  NOT the `.julia` conda root that hosts `cr2-py310`. `conda run -n amore` fails
  ("EnvironmentLocationNotFound"); drive it via **`conda run -p C:/Users/kr3ss/miniforge3/envs/amore`**.
  `conda run` also chokes printing unicode (χ) to the Windows console — dump notebook/script output
  to a UTF-8 file and read that, don't print χ to stdout through conda run.

## 4. The central finding — basins, not fans (and why k is NOT load-bearing here)

This is the opposite lesson from MEF, and it is the interesting one.

- **The PseudotimeKernel operator has three strong basins, not four.** Its Koopman spectrum has
  **three Perron eigenvalues at λ=1** (three nearly-absorbing basins) and then only a *shallow* gap
  (largest 0.008, at k=5) — unlike MEF's sharp single λ=1 + 0.025 gap. So the metastable structure is
  ~3-dimensional.
- **ISOKANN matches/beats GPCCA on every fate that is a metastable basin.** On pDC, CD14+ Mono and
  Normoblast, ISOKANN's χ ties or beats GPCCA (3-basin mean AUROC 0.994 vs 0.992 at k=4, 0.997 vs
  0.992 at k=5/6; it beats GPCCA on Normoblast 0.997 vs 0.987 and CD14+ Mono 0.997 vs 0.993, ties pDC
  0.996). top-30 purities ≥0.90.
- **cDC2 is the one fate ISOKANN cannot recover — and it is exactly the non-basin.** cDC2 (833 cells)
  branches late off the pDC/DC axis. Its Hungarian-assigned committor scores AUROC ≈0.32–0.45, and —
  the decisive diagnostic — **no χ column peaks on cDC2 at any k∈{4,5,6}** (`any_col_peaks_cDC2=False`;
  best column ≈0.85, which is the shared pDC/DC axis bleeding over). GPCCA's *absorption* probability
  captures cDC2 (0.985); ISOKANN's *metastable membership* does not, because the two coincide only
  when a fate is metastable. This is the in-dataset version of the multi-kernel benchmark's
  basin-vs-fate-fan conclusion (cf. the pancreas endocrine fan).
- **WHY (user-spotted "bridge", §4b of the sweep notebook).** In the χ-UMAPs cDC2 sits *between* CD14+
  Mono and the HSC/pDC region. Quantified, it is a **non-absorbing bridge into the CD14+ Mono basin**:
  (a) intermediate pseudotime (median 0.39, vs Normoblast 0.83 / CD14+ Mono 0.52 / pDC 0.51 — not an
  endpoint); (b) low self-retention in T (0.76, vs absorbing terminals 0.93–0.98; on par with the
  Proerythroblast transit state); (c) its outgoing mass drains 0.13→CD14+ Mono, 0.04→pDC, 0.06→G/M prog.
  Consequently ISA's **CD14+ Mono committor = 0.95 on the cDC2 cells themselves** (cDC2 column ≈0) — ISA
  lumps cDC2 into the monocyte basin — and **GPCCA's own absorption routes the average cDC2 cell
  0.62→CD14+ Mono vs only 0.31→cDC2.** So ISA isn't missing a basin; it faithfully reports a non-basin
  that commits to CD14+ Mono, and GPCCA only separates cDC2 because `set_terminal_states` forces the
  tiny cDC2 core to be absorbing. Diagnostic: `figures/ksweep_cdc2_bridge.png`.
- **k is therefore NOT load-bearing here** (the reverse of MEF). The k-sweep notebook shows
  k∈{4,5,6} all give three clean basins + an unrecoverable cDC2; extra modes just re-partition the
  same three basins / add an HSC mode. We picked **k=4** (the number of terminal lineages) as the
  primary per the user's lead; k=5 is the spectral-gap k* and gives the marginally cleanest 3-basin
  numbers (all three beat/tie GPCCA), k=6 matches GPCCA's macrostate count — all documented in the
  sweep. (Contrast MEF, where k=4 was *wrong* because the source genuinely needed its own mode.)

### 4b. Extended k-sweep k=3..7 (user-raised: continue k5/k6, add k=7, then k=3)
The sweep was extended to **k=3,4,5,6,7** (`armK_k3`, `armK`, `armK_k5`, `armK_k6`, `armK_k7`). Findings:
- **k=3 is the operator's own count and the cleanest model.** T has **three Perron eigenvalues at λ=1**
  (three absorbing basins). GPCCA(3) = **pDC, CD14+ Mono, Normoblast** (verified by re-running the
  estimator); GPCCA only adds erythroid progenitors at n=4,5 and **cDC2 last at n=6** — so cDC2 is the
  least-metastable fate for GPCCA too. ISOKANN k=3 recovers the same three basins, with **0 weak modes and
  the lowest ISA loss of the whole sweep (2e-6)**, 3-basin mean AUROC 0.993 > GPCCA 0.992.
- **k=3 is ISA-seed-sensitive (the one real gotcha).** The first k=3 run (seed 0) hit a vertex
  local-optimum: it put a vertex on the transient **Proerythroblast** and **missed pDC** (pDC AUROC 0.59).
  Of seeds {0,1,2,3}, seeds 1 and 2 recover pDC/CD14/Normoblast; seeds 0 and 3 let the dominant erythroid
  arm grab a slot. Added `SEED_OVERRIDE` env to `03_train_armK.py`; promoted seed 1 to `armK_k3` (the bad
  run kept as `armK_k3_seed0_localopt`). This sensitivity appears **only at the tightest k=3**; k≥4 get pDC
  cleanly every time. GPCCA(3) is the ground-truth check that catches it.
- **Convergence vs k (user wanted to see the loss settle).** Same long best-iterate continuation for all:
  **k=3 flat (best 2e-6, 0 weak), k=4 flat (0 weak), k=6 flat (2e-5) but 2 weak modes, k=7 flat but
  2–3 weak (it spends its extra modes on HSC×2 + Proerythroblast).** **k=5 is the exception — its loss
  keeps oscillating even at 2200 iters** because its single starved mode (sd 0.096) keeps flipping ISA
  vertices. So "do they converge?" → yes for k=3/4/6/7 given enough iters; k=5 stays noisy.
- **cDC2 never recovered at any k=3..7** (`any_col_peaks_cDC2=False`; best column 0.69–0.85). The science is
  k-invariant; the sweep notebook now spans k=3..7 with a GPCCA(n)-progression table, the convergence
  table, and the per-column Hungarian-vs-real table. Snapshots: `armK_k{5,6}_raw600`,
  `armK_k3_seed0_localopt`, `armK_k3s{1,2,3}`.

### 4a. The k=6 convergence question (user-raised) — undertraining AND structural over-partition
The first k=5/k=6 runs were raw `03` (600 iters, *final* iterate kept); only k=4 got the `03b`
best-iterate continuation, so the comparison was unfair and k=6 looked un-converged (it ended at ISA
loss ≈9e-4, above its own min ≈4e-5, with two near-collapsed modes). **Fix:** taught `03b` to respect
`ARMK_SUFFIX` and gave k=5 (+400) and k=6 (+600) the same best-iterate continuation. After that:
- k=6 **does converge in loss** (best 3.4e-5) and all four terminal purities hit 1.0; its χ-binned
  driver recAUC even becomes the best of the sweep (0.263). So part of the "didn't converge" was real
  undertraining — now fixed.
- BUT it is **not a split complex eigenpair**: the leading 20 eigenvalues are **purely real**
  (max|Im λ|=0), so no conjugate pair is being cut. And it is **not only iters**: the **two weak modes
  persist (SD ≈0.078, 0.117) after 1200 iters**. The cause is the **near-degenerate real cluster** after
  mode 5 (modes 6–9 within ~0.007), which a 6th membership cannot cleanly claim. So **k=5 is the clean
  ceiling (1 weak mode), k=4 is fully strong (0 weak), k=6 over-partitions (2 weak)** — the weak-mode
  count 0/1/2 for k=4/5/6 is the diagnostic. cDC2 stays AUROC ≈0.32–0.48 with no χ column peaking on it
  at *any* k, so the basin-vs-fan finding is robust to convergence and to k. (`armK_k5_raw600` /
  `armK_k6_raw600` keep the pre-continuation snapshots.)

## 5. Drivers (focal lineage = pDC, CR2's own example)

19 of the curated 33-gene pDC TF/marker panel land in the 4000-HVG universe. On pDC (a clean basin)
the **correlation-with-χ** readout matches/edges GPCCA (recAUC 0.421 vs 0.408; curated @100 = 14 =
14, @50 = 10 = 10) at loading-norm bias ρ≈0.69; per-lineage TF50 recovery is competitive-to-better
(pDC 10 vs 9, Normoblast 7 vs 4, cDC2 6 vs 5). **Here the χ-binned gradient UNDERPERFORMS corr-χ**
(recAUC 0.224) — the reverse of the pharyngeal/MEF ranking — so corr-χ is the stronger ISOKANN driver
readout on this dataset; both are reported. The 4000-HVG ISOKANN model is again the weak input
representation (two memberships collapse), so the 50-PC model is the representative throughout.

## 6. ISA parallelization (user's explicit check)

Confirmed already in `src/amore/isotarget.py::_indexmap`: the inner-simplex vertex search is the
vectorised/BLAS version (projects all n row-differences at once; ~80× faster on large n, bit-for-bit
identical vertices) committed during the MEF benchmark. Nothing to add. At 24k cells it is not a
bottleneck anyway.

## 7. Reproduce

```
# CR2 side (cr2-py310): GPCCA baseline + shared artifacts
conda run -n cr2-py310 python 01_cr2_reproduce.py
# ISOKANN side (amore env = miniforge3): primary k=4 + continuation + HVG
conda run -p C:/Users/kr3ss/miniforge3/envs/amore python 03_train_armK.py        # K_CHI_OVERRIDE=4 (default)
conda run -p C:/Users/kr3ss/miniforge3/envs/amore python 03b_continue_armK.py
conda run -p C:/Users/kr3ss/miniforge3/envs/amore python 04_train_armK_hvg.py
# k-sweep alternates
K_CHI_OVERRIDE=5 ARMK_SUFFIX=_k5 conda run -p .../amore python 03_train_armK.py
K_CHI_OVERRIDE=6 ARMK_SUFFIX=_k6 conda run -p .../amore python 03_train_armK.py
# notebooks
conda run -p .../amore python _build_nb.py && \
  conda run -p .../amore jupyter nbconvert --to notebook --execute --inplace hematopoiesis_isokann_benchmark.ipynb && \
  conda run -p .../amore python _finalize_nb.py
conda run -p .../amore python _build_compare_nb.py && \
  conda run -p .../amore jupyter nbconvert --to notebook --execute --inplace hematopoiesis_k_sweep_comparison.ipynb
```
Models: `armK` (primary k=4, 03b-continued), `armK_k5`, `armK_k6`, `armK_hvg` (k=4 HVG cross-check).
