# Postmortem: inverse-PCCA+ spectral gap, from `src/amore` to 2cm2's full trajectory

Covers the whole session's arc: implementing `amore.inverse_pcca` (recover Koopman
eigenvalues/timescales/eigenfunctions from a learned chi, without forming the transfer
matrix), validating it on the ISOKANN benchmark systems, and applying it to find the
number of metastable states in 2cm2 — first on a chopped trajectory window, then on the
full trajectory, ending with a validated k=3 state model and its full PyMOL pipeline.

## What was built

**`src/amore/inverse_pcca.py`** (the reusable core):
- `inverse_pcca(chi, propagate, tau, reversible=..., weights=..., transform=...)` — the
  original deliverable. `Lambda_S = G_hat^{-1} C_hat`, real-Schur pinning for the
  non-reversible case (reuses `examples/isokann_benchmark/lib/schur_isotargets.py`, not
  re-derived).
- `group_conjugate_pairs(lam)` / `SpectralProcess` — groups a recovered spectrum into
  distinct physical processes (a genuine complex-conjugate pair is ONE process, not two
  raw eigenvalue slots).
- `find_spectral_gap(processes)` / `SpectralGap` — the largest modulus drop between
  consecutive processes; the automated "how many states" readout.
- `plot_complex_plane` / `plot_rate_matrix` — shared plotting, used by every notebook
  below instead of each duplicating `imshow`/scatter code.
- `rate_matrix(Lambda_S, tau)` — `Q = logm(Lambda_S)/tau`, the continuous-time generator.
  Works identically reversible or not (`Lambda_S` is always real); off-diagonal entries
  are a principled "which edge matters" readout, requested as an alternative to 2cm2's
  frame-counting heuristic.

**`examples/isokann_benchmark/lib/harness.py`**: `train_chi`/`_run_isotarget`/`_make_net`
etc. gained an optional `k` parameter (previously `N_STATES=3` was hardcoded through the
whole call chain, blocking any overparametrization experiment). Backward compatible
(`k=None` preserves the exact prior behavior; confirmed via the full pytest gate before
and after).

**`examples/isokann_benchmark/lib/ground_truth.py`**: `tw_reference_lambda_s()` and
`adp_reference_lambda_s()` — hard-partition the existing fine-grained binned reference
transfer operators into the known wells/basins and run them through `inverse_pcca` itself
(not a re-derivation), giving a reference `Lambda_S`/rate matrix directly comparable to a
trained model's.

**Notebooks**:
- `examples/isokann_benchmark/inverse_pcca.ipynb` — the main deliverable: triple
  well/ADP/ring validation, triple-well overparametrization (clean success), ADP
  overparametrization (informative failure), ring complex-pair handling, VAMP-2
  cross-check, coarse rate-matrix validation against the reference operators.
- `examples/2cm2/spectral_gap/spectral_gap.ipynb` — chopped-trajectory `[350,2000)` gap
  search; lag=1 showed no gap (everything compressed near `|lambda|=1`), lag=20 landed
  cleanly on k=3.
- `examples/2cm2/dim3/` — full per-dim pipeline (loss, simplex, edges, rate matrix,
  pathways, PyMOL) for k=3 on the chopped window.
- `examples/2cm2_full/spectral_gap.ipynb` — same question on the FULL 3000-frame
  trajectory: k=5,6,7,8,10,12 at lag=20. k=5/6/7 all trained cleanly and independently
  landed on k=3, confirming the chopped window's answer wasn't a truncation artifact;
  k=8/10/12 pushed the apparent gap higher but showed real training instability
  (non-converging loss), not a different true answer.
- `examples/2cm2_full/dim3/` — same full pipeline as `2cm2/dim3/`, for the full
  trajectory's validated k=3.

## Bugs found and fixed in EXISTING (not newly-written) shared code

Worth calling out separately from the new feature work, since these affect
`examples/2cm2/`'s own prior outputs too, not just this session's new notebooks:

1. **`_sorted_real_schur` ordering bug** (`schur_isotargets.py`, found early while
   building `inverse_pcca`'s non-reversible route) — a naive permutation broke
   triangularity; fixed in that module.
2. **`render.py`: `export_state_structures` missing `_pack_ligand`** — the edge-pathway
   export already packed the ligand into the protein's periodic image every frame before
   aligning; the state-render export didn't, so `unwrap()` alone could leave the ligand a
   whole box-vector away, LOOKING far outside the pocket when actually bound.
3. **`render.py`: `_pack_ligand` used centre-of-mass distance, not nearest contact** —
   picks the wrong periodic image whenever the true binding pocket sits far from the
   protein's own centroid (true for any elongated/multi-domain protein). Confirmed: one
   frame's COM-packed distance was 25 Å; true nearest-contact distance at that exact frame
   is 5.2 Å. Now checks all 27 candidate integer box-vector shifts and picks the one
   minimizing actual atom-atom distance.
4. **`render.py`: camera/pocket-selection never included the pocket itself** — even after
   fixing (2), `cmd.orient`/`cmd.zoom` only ever targeted the ligand, and the pocket
   selection used a strict 5 Å cutoff; a genuinely-real ~5.2 Å contact fell just outside
   both, rendering as an empty white frame. Fixed: pocket cutoff widened to 7 Å, camera
   frames ligand+pocket together (both the state-render and edge-session PyMOL scripts had
   this bug independently).
5. **`render.py`: `render_model_states`/`build_edge_pses` default `out_dir`** resolves
   relative to `render.py`'s OWN location (`lib/../dim{k}`), not the calling notebook's
   directory — harmless for `examples/2cm2/`'s own notebooks (that IS where they live),
   but reused from `examples/2cm2_full/` it silently served `examples/2cm2/dim3/`'s stale
   chopped-window renders under a fresh-looking cell, with `force=False` cache-hitting
   without complaint. Caught by checking the cell's own log line against `pwd()`, not an
   exception. Worked around by passing `out_dir` explicitly rather than changing the
   default (which `examples/2cm2/`'s own notebooks implicitly rely on).

None of bugs 2-5 raised an exception. All were caught by looking at a rendered PNG that
seemed visually off — emptier than its sibling states, or a log line naming a directory
that didn't match `pwd()` — not by anything failing loudly.

## Bug found in a test I wrote this session

`test_tw_rate_matrix_edge_ranking_matches_reference` initially asserted an exact tie-break
order between triple well's (well0-well2) and (well1-well2) edges. Both are legitimately
near-degenerate (`TW_WELLS` is nearly symmetric under well0<->well1), so the assertion was
testing noise, not signal — a real training-seed/discretization difference flipped which
of the two was marginally larger. Fixed to assert the robust claim instead (edge
(well0-well1) is the clear weakest, by a wide margin, in both reference and learned).

## Scientific findings (2cm2)

- **Chopped window `[350,2000)`, lag=1**: no spectral gap — every eigenvalue compressed
  near `|lambda|=1` for every k, unstable leading timescale across k. Root cause: lag too
  short relative to every relaxation process (`lambda = exp(-tau/t) ~= 1` when `tau << t`
  for everything).
- **Chopped window, lag=20**: clean gap at k=3 (Perron + one genuine complex-conjugate
  pair — an oscillatory/rotational process, not two independent real states).
- **Full trajectory, lag=20, k=5/6/7**: all three train cleanly (`k_eff=k` exactly, no
  collapse) and independently confirm k=3 — the chopped window's answer was NOT a
  truncation artifact.
- **Full trajectory, lag=20, k=8/10/12**: apparent gap grows (5, 6, 7) but training
  visibly destabilizes (oscillating loss, `k_eff` collapse) — noise past the point the
  model resolves cleanly, not evidence of more true states.
- **k=3 full-trajectory model**: trained, validated, full pipeline built and visually
  verified (loss converges cleanly, `k_eff=3`, edge table + rate matrix agree on which
  edge is weak, PyMOL renders correct after the bug fixes above).

## What's still open / not done

- ADP's rate-matrix vs. reference comparison is inconclusive: vacuum ADP's alphaR basin
  is only ~2% populated (its stability in the textbook SOLVATED landscape comes from
  solvent H-bonding, mostly absent here), so the permutation-resolution step that would
  let a 3-way comparison happen doesn't reliably find a valid bijection. Not a code bug;
  a genuine statistical limitation of this specific benchmark system.
- Kramers/Langer theory and TPT-from-committors were scoped OUT of the rate-matrix work
  (explicitly, by user choice) in favor of the fine-operator reference approach. Left as a
  documented option if independent, non-data-derived validation is wanted later.
- `examples/2cm2_full/`'s k=5,6,7,8,10,12 models are all trained and cached; only k=3 got
  the full downstream pipeline (states/edges/pathways/PyMOL). No other full-trajectory k
  has renders.
- `examples/2cm2/dim4`, `dim5`, `dim6` (chopped window, lag=1) were deleted (by the user)
  as superseded once lag=20/k=3 was established.

## Process notes for next time

- **Background training jobs can be killed by session/environment boundaries** with no
  Python-level error — the process just vanishes, no traceback. Check `scratch` cache
  survival before assuming a full re-run is needed; in this session a killed run had
  already cached COMs, features, and one full model, so the retry only needed the
  remainder.
- **Stale "stopped" task-notifications after a session boundary are not necessarily real
  failures** — check the actual process (`ps aux`) and output file state before treating
  them as crashes.
- **A trained model's softmax membership index order is arbitrary.** Nothing in ISA
  training ties membership 0 to any particular physical state. Always resolve the
  permutation (evaluate the trained net at known representative points, argmax) before
  comparing a learned `Lambda_S`/rate matrix to a physically-labeled reference — this
  produced an apparently-backwards result for triple well until corrected for.
- **Visual inspection caught every rendering bug in this session; none raised an
  exception.** When reusing a rendering/export pipeline in a new context (different
  trajectory, different directory), spot-check the actual output images against what's
  physically expected rather than trusting "it ran without error."
- **When reusing existing library code from a new directory**, watch for hardcoded
  paths relative to the library's own location rather than the caller's — they work by
  coincidence in the original context and fail silently (not loudly) elsewhere.

## Addendum: feature-set spectral-gap re-validations (ligand-inclusive, then pocket)

A later session re-asked "how many metastable states" on two further feature sets, each
time re-running the same overparametrization/spectral-gap check
(`amore.inverse_pcca.find_spectral_gap`) rather than assuming k=3 still held. Both
generating notebooks (`build_spectral_gap.py`/`spectral_gap.ipynb` for ligand-inclusive,
`build_spectral_gap_pocket.py`/`spectral_gap_pocket.ipynb` for pocket) were later removed
as superseded once the codebase settled on the pocket feature set exclusively -- this
addendum preserves the findings themselves.

**Ligand-inclusive features** (`data.build_features_lig`: protein residue-residue COM +
ligand-heavy-atom-to-residue-COM distances, 46,455-dim) -- built because the protein-only
chi has exactly zero gradient w.r.t. any ligand atom, blocking ligand-position-aware
chi-MEP work. Result: **k=4, not k=3**. Two independently-trained overparametrized models
(k_trained=5 and k_trained=6) agreed on a spectral-gap elbow at process 4 (k=5: drop
0.936->0.789, the largest of any run; k=6: drop 0.942->0.905) -- the same cross-k
convergence criterion that originally validated k=3. Interpretation: the protein-only
features are blind to where the ligand sits/how it's oriented within a pocket
conformation; adding the ligand's own heavy-atom positions gave chi enough information to
resolve a 4th distinguishable process (apparently two ligand sub-poses within what
protein-only chi had collapsed into one state).

**Pocket features** (`data.build_features_pocket`: whole-protein side-chain COM-COM +
all-atom 5A ligand-protein contacts, 23,212-dim, `res_pairs` pruned to a 25A cutoff --
built later to fix H-atom relaxation-lag artifacts the ligand-inclusive COM-only ligand
term caused in chi-MEP work) -- re-ran the same check. Result: **ambiguous, unlike the
clean ligand-inclusive agreement above**. k_trained=4 collapsed to k_eff=3 (one dimension
never differentiated) with gap at 2 states; k_trained=5 (full k_eff=5) found gap at 3
states; k_trained=6 (full k_eff=6) found gap at 4 states -- the two fully-resolved
overparametrized runs disagreed with each other (3 vs 4), rather than the ligand-inclusive
case's clean double-agreement. User's read, which this was left on: favor k=3, because
k=4's own explicit test for a 4th state (k_trained=4) declined to resolve one
(collapsed to k_eff=3) rather than confirming it, and k=5 (fully resolved) independently
agreed with that read -- two signals for 3 against k=6's one outlying signal for 4, and
k=6's extra process reads more plausibly as capacity-driven over-fragmentation (splitting
one real state into two neighboring sub-populations) than genuine signal. Not as airtight
as the ligand-inclusive validation's clean pairwise agreement; a k=7 tiebreaker run was
suggested but not done. If pocket-feature chi-MEP/state work is revisited, keep this
ambiguity in mind rather than treating k=3 pocket as settled to the same standard k=3
protein-only or k=4 ligand-inclusive were.
