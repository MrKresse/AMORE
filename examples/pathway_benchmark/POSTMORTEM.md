# pathway_benchmark — postmortem & handoff

Everything the next agent needs to understand, run, extend, or trust this benchmark.
Written after the session that built it (it started life as `examples/minimal.ipynb`, grew into a
pathway benchmark, and was renamed). Read this before touching the notebook — several design
decisions look arbitrary but are the resolution of a long argument; don't re-litigate them.

---

## 1. What this is

The **full AMORE pipeline on vacuum alanine dipeptide, organised around the membership simplex
Δ²**. A k=3 softmax ISOKANN maps configs to (χ₁,χ₂,χ₃) with the three metastable states at the
simplex **vertices** and the pairwise interconversions on its **edges**. The notebook:

1. **Sampling** — well-tempered MetaD in (φ,ψ), the reconstructed FES, 0.1 ps Koopman grid bursts.
2. **ISOKANN** — three softmax memberships, ISA isotarget, no warm-up.
3. **The simplex** — the Δ² triangle (anchors → vertices) + the per-edge maps.
4. **Pathways** — the fundamental object is the **membership flow ∇χ_i** (one per state); each
   path is lifted by **MEP** (min energy), **MFEP** (min free energy), and **energy-free MFEP**
   (gradient-flow ensemble + path clustering), and the **edge it realises is a-posteriori**.
5. **Free energy** per tube vs MetaD; **sensitivity** per edge.

Layout mirrors `examples/isokann_benchmark/`: notebook at top, helpers in `lib/`
(`pipeline.py`, `gen_data.py`), `build.py` regenerates the `.ipynb`, `compute_pathways.py` is a
standalone heavy-compute helper (see §6).

---

## 2. THE key idea (read this first)

**Trace the membership flow ∇χ_i, NOT an edge coordinate, and realise the edge a-posteriori.**

The tempting object is an edge coordinate `s = ½(χ_i − χ_j + 1)`. It is degenerate: `s = ½` holds
on the genuine i↔j saddle **and** in the third basin (both have χ_i = χ_j ≈ 0). So the third basin
lies on the `s = ½` level set, and minimising/relaxing a seed there slides it into that basin — a
real failure we spent a long time fighting.

The membership χ_i has no such defect: the basins sit at χ_i ∈ {0,1}, **never** on the χ_i = ½
transition surface. So ∇χ_i is clean — seeds and paths can't fall into a foreign basin — and
"which edge a path realises" is just a label (the two states it connects). `reaction_path_face`
integrates ∇χ_i; the edge is read off afterward.

---

## 3. The hold-constraint saga (built, validated, then STRIPPED — don't rebuild it)

Before the membership-flow realisation above, we tried to keep edge paths on the simplex edge by
**holding the activity** a = χ_i + χ_j during the level-set projection. A full constraint stack
was added to `src/amore/mep` (`hold=` on `levelset_retract` / `energy_min_on_levelset` /
`reaction_integrator` / `reaction_path_minimum` / `sample_levelset_projected` /
`build_chi_mep_projected`, plus active-set / joint-Newton retraction and noise projection in the
constrained Langevin). It worked, but:

- a **shared** activity floor is infeasible for low-activity seeds (drove them into the third
  basin); per-seed holding worked but is fiddly;
- ultimately the drift is a **seed-initialisation** artifact at s≈½ from *aggressive* minimisation
  — light prep avoids it, and the path geometry handles the rest (the basin is off every s≠½ level
  set).

So the whole `hold=` machinery was **removed from src** at the user's request. **KEPT**: the MEP
**initial-state-minimisation fix** in `reaction_path_minimum` (the centre image is now projected
onto its level set like every other image, not patched in raw), and the **two-view API** in
`src/amore/mep/simplex.py` (`FaceCV` / `EdgeCV` / `ActivityCV`, `separatrix_frames`,
`reaction_path_face` / `reaction_path_edge`, `mfep_face` / `mfep_edge`) as plain integrations.
If you find yourself adding `hold=` back, re-read §2 first.

---

## 4. Seeds: the min/equil tension (don't crank minimisation)

Seeds are membership transition frames (χ_i ≈ ½), prepared by `pl.prepare_seed`
(short energy-min + short orthogonal-sampling equilibration, **no constraint**). The tension:

- **Heavy minimisation collapses the seed diversity** — the χ_i=½ surface has only a couple of
  low-energy points (the i–j and i–k saddles), so all seeds fall onto them and the (φ,ψ) spread
  that carries the **parallel routes** is lost. Equilibration does NOT undo it (the well is deep;
  ~hundreds of Langevin steps just rattle inside).
- So the production config is **medium**: `SEED_MIN=90, SEED_EQUIL=120, N_PREP=48` per membership.
  Lighter keeps more routes but higher-energy seeds; heavier is cleaner but collapses.

Parallel routes come from **seed diversity + clustering**, never from targeting seeds at named
regions (the user explicitly refused region priors — see §5).

---

## 5. Three robustness fixes that actually mattered (in `lib/pipeline.py`)

- **Edge realisation — `edge_of_path`.** Label by the WHOLE-path correlation: partner = the non-i
  membership most *anti-correlated* with χ_i along the path (the state being converted); the
  off-edge state stays flat. The earlier single-(χ_i≈0)-endpoint argmax is unstable and
  mis-colours paths (a χ₁ path at φ≈0 was wrongly "1–3"; corrected to "1–2").
- **Tube clustering — `cluster_paths`.** Use an **absolute** threshold (`thresh=0.8`, plus
  `min_size=2`) in the level-set-aligned (φ,ψ) path metric. That metric ≈ the **RMS angular
  separation [rad]** between two paths, so a fixed threshold resolves parallel routes regardless
  of lopsided sizes. A median-relative threshold over-segments the dominant route and merges the
  minor parallel ones — that, not seeding, was why edge 1–2's ψ>0 and φ>π/2 channels were missing.
  At 0.8: edge 1–2 → **3 tubes** (main, ψ>0, φ>π/2), edge 2–3 → **2** (the C7ax ψ± channels),
  edge 1–3 → 1.
- **Path plotting — `plot_path(break_thresh=1.8)`.** Break the line at any Δ>1.8 rad
  discontinuity, not only ±π wraps. Extreme-level-set medoids occasionally land in another basin
  (Δ≈3 rad, *just under* π) and otherwise draw a spurious straight line across the panel. Only
  ~0.9% of real steps exceed 1.8 rad (p99 ≈ 1.14), so legitimate path is untouched. Overlay also
  uses a **white halo** (`halo=True`, path_effects — the `cr2_benchmark` style).

Conventions: edge colours **1–2 black, 1–3 grey (deprioritised), 2–3 magenta** (orange/blue blend
into RdYlBu_r); plots at **300 dpi**; sensitivity in **viridis**; FES is `fes[psi,phi]`, plotted
`contourf(phi_axis, psi_axis, fes)` — all paths use the same `adp_phi`/`adp_psi`.

---

## 6. Compute gotchas (these cost real time)

- **`pl.run_ensemble` uses the `spawn` mp context, not fork.** Training the net in-process runs
  autograd backward, which poisons fork-based workers ("Unable to handle autograd's threading in
  combination with fork-based multiprocessing"). Consequence: any standalone driver that calls
  `run_ensemble` needs an `if __name__ == "__main__":` guard.
- **OpenMM Reference single-point force ≈ 14 ms/call** (chi-grad only ~1.4 ms), so MEP/MFEP paths
  are seconds-to-minutes each and the ensemble MUST be CPU-parallel (14 procs, 1 torch thread per
  worker). See [[amore-compute-env]].
- **The full ensemble (3 methods × 48 seeds × 3 memberships = 432 paths) exceeds the nbconvert
  5000 s cell timeout.** Compute it OUT of the notebook with `python compute_pathways.py` (no
  timeout; writes `pathways.pkl` in the `{prepared, results}` format the notebook loads), then run
  the notebook — its §4 cell just loads the cache. The §4 cell *can* compute inline for small task
  counts, but will time out at production scale.
- Caching on scratch (`/scratch/<user>/amore_pathway`, gitignored): `isokann_k3.pt` + `chi.npy`
  (model), `prep_seeds.npy` (prepared seeds), `pathways.pkl` (all paths). Delete the relevant one
  to force a re-train / re-seed / re-integrate.

---

## 7. Reproduce from scratch

```bash
cd examples/pathway_benchmark
# 1. data + model are cached on scratch; if absent, the notebook's §1/§2 cells generate them
#    (MetaD ≈ 9 min, ISA ≈ 6 min). Seeds: §4 preps them (or reuses prep_seeds.npy).
# 2. heavy pathway compute (decoupled from the cell timeout):
python compute_pathways.py            # ~115 min, writes pathways.pkl
# 3. build + execute the notebook (loads cached pathways -> just plots, ~5 min):
python build.py
jupyter nbconvert --to notebook --execute --inplace pathway_benchmark.ipynb \
    --ExecutePreprocessor.timeout=2000
```

`build.py` knobs are env-overridable (`PB_SEED_MIN`, `PB_N_PREP`, `PB_EF_STEPS`, …) for a faster
pass. The notebook is ~13 MB at 300 dpi — open via browser / `nbconvert --to html`, not the VSCode
notebook webview (large embedded-figure notebooks can OOM it; see the isokann_benchmark postmortem).

---

## 8. src changes made this session (outside this dir)

- `src/amore/mep/core.py` — `reaction_path_minimum` projects the initial state onto its level set
  (centre image treated like every other). **Kept.** The `hold=` additions were **reverted**.
- `src/amore/mep/constrained.py` — `hold=` additions **reverted** to the original single-CV form.
- `src/amore/mep/simplex.py` — **new**, the two-view API (face/edge), plain integrations.
- `src/amore/mep/__init__.py` — exports the simplex API.

---

## 9. Open items / where to go next

- **Edge 2–3 φ≈0 "direct into the basin" route** does not separate at `CLUST_THRESH=0.8` (folded
  into the ψ>0 cluster). A density-based clustering (DBSCAN on the precomputed path-distance
  matrix) or a per-edge adaptive threshold would surface it — still with no region priors.
- **Path quality**: MEP/MFEP tubes don't always span fully to the far vertex; the medium-min
  seeds and modest `MFEP_LS=50` make some MFEP medoid paths look straight. Bumping `MFEP_LS`
  improves them but needs recomputing the MFEP paths (~30–40 min).
- **State naming** is index-based (χ₁/χ₂/χ₃) with (φ,ψ) centroids; the C7ax/αR/C7eq `basin_name`
  heuristic is unreliable (circular-mean near ±π) and is not used for the deliverable labels.
- A `train_isokann()` convenience could live in `src` (currently the loop is `pipeline.py`).
