# Postmortem: chi-MEP performance forensics, handoff for an `openmmtorch`/TorchForce attempt

Narrow scope (not a full session recap): why the projected-Langevin chi-MEP sampling loop
(`amore.mep.constrained.sample_levelset_projected`) is ~260x slower per step than plain
unbiased MD, what's already built toward fixing it via `openmmtorch`/`TorchForce`, and what
a next session needs to actually try it. The broader MFEP/edges/spectral-gap work this
session covered is intentionally NOT recapped here — see git history / the user directly if
that context is needed; this file is scoped to the performance investigation only.

## Open question to resolve FIRST, with the user, before implementing

This session ended up de-prioritizing the whole chi-MEP/MFEP approach ("dont do an MFEPs
anymore focus on the standard edges and the heatmaps") after repeated, not-fully-resolved
force-spike/instability issues near basin vertices (genuine chi-gradient collapse, not a
seed-preparation artifact -- confirmed by testing that minimizing+equilibrating the seed
thoroughly did NOT prevent explosions later in a chain). A faster per-step evaluation via
TorchForce does not fix that reliability problem -- it only makes iteration on it cheaper.
**Clarify with the user whether this is a standalone performance/methods curiosity, or
whether faster iteration is meant to justify revisiting MFEP reliability work.** Don't
assume the latter.

Also tangential but relevant if MFEP work IS revisited: the pocket feature set's own k=3
assumption came back ambiguous from a proper spectral-gap validation this session (k=5
trained -> gap at 3 states, k=6 trained -> gap at 4 states; user's own read favored 3, but
it's not as clean as the ligand-inclusive feature set's earlier unambiguous k=4 finding).
Worth keeping in mind if any of this feeds back into "how many states does chi need."

## Performance findings (all measured this session, on the current GPU allocation)

**Hardware**: `NVIDIA H100 NVL MIG 1g.24gb` -- a 1/7th-compute MIG partition of a physical
H100, not the full chip. 23.2 GB VRAM available to this partition.

**The chi model is small and VRAM is not remotely the bottleneck**: 98.5M params (385 MB
of that is one `23519 -> 4096` first linear layer), 428 MB allocated / 23.2 GB available at
runtime (~2% used). Ruled out directly, not inferred.

**Baseline unbiased MD** (no chi-network involvement at all, plain `OpenMMSimulator.step`,
`system_flex.xml`/pocket-model topology, `/scratch/htc/jkresse/2cm2_mfep/`):
- Flexible (no constraints), 0.5 fs/step: 941 steps/s -> ~2.1 min per 60 ps.
- Rigid water + HBonds constraints, 2.0 fs/step: 832 steps/s -> ~0.6 min per 60 ps.
- Same-dt (0.5 fs) apples-to-apples: rigid is only ~2.6% slower per step than flexible --
  SETTLE/RATTLE's own overhead is small; rigid's real wall-clock advantage is entirely the
  4x larger stable timestep, not cheaper per-step compute.
- Both NaN on step 1 from raw (unminimized) PDB positions -- flexible from real force-field
  strain, rigid from the constraint solver choking on the same strain with no bond-stretch
  DOF to absorb it into. `LocalEnergyMinimizer.minimize(ctx, tolerance=10.0,
  maxIterations=0)` (OpenMM's own defaults -- confirmed empirically sufficient and cheap,
  ~10s) before stepping fixes both. See `_minimize_seed` in `constrained.py` for the same
  fix already applied to the sampling code.

**The chi-MEP sampling loop itself** (`sample_levelset_projected`'s `time_breakdown=True`,
already-built-in per-step instrumentation, previously never actually surfaced): one live
measurement showed **~279 ms/step** (OpenMM force query 21.5 ms / 8%, chi-network gradient
173.2 ms / 62%, Newton retraction 84.4 ms / 30%) -- a ~260x slowdown vs the 941 steps/s
(~1.06 ms/step) unbiased baseline.

**That 279 ms/step figure could NOT be reproduced in isolation, under several increasingly
faithful attempts, all just now with nothing else running on the GPU:**
- `_chi_and_grad` alone (exact real code path incl. per-call numpy<->GPU transfer + the
  `.item()`/`.cpu()` sync it forces): 4.17 ms.
- Interleaved with real OpenMM force queries + `setPositions` on the actual
  `system_flex.xml` context (matching `_step`'s real call pattern), sustained over 500
  steps to rule out allocator fragmentation / warmup effects: converges to ~2.2 ms OpenMM +
  ~4.7 ms chi-grad + ~1.3 ms retract-only-call = **~8.2 ms/step total**, i.e. the isolated
  reproduction is ~34x FASTER than the live-measured number.
- Batch-size scaling (separately established, useful for any future batching work):
  3.43 ms/frame at batch=1 down to 0.19 ms/frame at batch=512 (~18x) -- confirms real but
  secondary underutilization at batch=1, not the main story here.

**Best-supported explanation for the gap, not fully proven**: GPU contention from
something else scheduled onto the same MIG instance at the time of the original
measurement. MIG isolates hardware *between* different instances but not between multiple
processes queued onto the *same* instance -- normal time-slicing still applies within one
slice. Circumstantial evidence: the *retraction* step (a single forward pass, no gradient,
the cheapest of the three operations) was inflated the MOST relative to its own isolated
cost (84.4 ms live vs 1.3 ms isolated, ~65x) -- more than the full gradient call (~37x) or
the OpenMM query (~10x). That pattern tracks with number of CPU<->GPU sync points per
operation (`.item()`/`.cpu()` calls are exactly where a process waits its turn on a shared
scheduler), not with anything intrinsic to the operations' actual FLOP cost. Not
independently confirmed (would need e.g. `nvidia-smi`-equivalent visibility into what else
was on the same MIG instance at that moment, which this session didn't have permission
for) -- flag this as a hypothesis, not an established fact, to whoever picks this up.

## Why TorchForce could still help regardless of the contention question

Even setting aside contention, the current loop is Python-orchestrated: every step does a
fresh host->device transfer, a from-scratch autograd graph forward+backward through the
featurizer (many small ops, e.g. per-residue COM gather/scatter + PBC minimum-image logic,
each a separate dispatched kernel), and 2+ explicit sync points
(`.item()`/`.cpu().numpy()`). `TorchForce` (via `openmmtorch`) fuses a *traced* CV pipeline
into a single call made from OpenMM's own C++ side as part of the normal force evaluation
-- no per-step Python dispatch, no explicit sync points, and (per `CustomCVForce`) the
whole thing can stay resident on the GPU across many steps the way native OpenMM stepping
already does (941 steps/s baseline). This should help independent of whatever the
contention story turns out to be, and would also remove `_step`'s current CPU-side Newton
retraction round-trip if the retraction itself is reformulated as an OpenMM-side operation
(not yet designed -- today's retraction is a simple explicit numpy calculation, would need
its own thought for how/whether to move it).

## What's already built toward this (`src/amore/mep/constrained.py`)

- `HAS_OPENMMTORCH` (line ~44): `try: from openmmtorch import TorchForce`. **Confirmed
  today: `openmmtorch` is NOT importable in the `molsim` conda env**
  (`/scratch/htc/fsafarov/conda/envs/molsim/bin/python -c "import openmmtorch"` ->
  `ModuleNotFoundError`). This is the first real blocker for a next session. Note `molsim`
  is `fsafarov`'s env, not this user's -- but write access exists in practice (h5py was
  successfully `pip install`ed into it earlier this session for an unrelated import gap),
  so installing `openmm-torch` there is plausible but unverified. Check conda-forge's
  `openmm-torch` package against the env's existing torch/CUDA versions before assuming a
  clean install; may need a fresh env if there's a version conflict.
- `_PairDistFeaturizer` (line ~81) / `_CVPipeline` (line ~104): a traceable `nn.Module`
  pipeline for a SIMPLE pairwise-distance featurizer (raw atom-index pairs -> Euclidean
  distances -> chi network). This does NOT match the featurizer in active use
  (`comfeat.make_torch_featurizer_pocket` in `examples/2cm2_full/lib/comfeat.py`:
  side-chain center-of-mass gather/scatter + all-atom 5A ligand-protein contacts, PBC
  minimum-image-aware). Whether `comfeat`'s featurizer traces cleanly via
  `torch.jit.trace`/`torch.jit.script` as-is is UNTESTED -- likely the real engineering
  work here, more involved than the existing scaffolding assumes. Start by trying
  `torch.jit.trace` on `comfeat.make_torch_featurizer_pocket`'s output module directly and
  see what breaks (data-dependent control flow / indexing patterns are the usual
  trace-vs-script pain points).
- `export_cv_torchscript(nu, pairs, example_positions_nm, path)` (line ~128): traces
  `_CVPipeline` and saves a `.pt` TorchScript file for `TorchForce`. Written for the
  simple-pairwise-distance case; would need a `comfeat`-featurizer-compatible variant (or a
  generalized version taking an arbitrary already-built featurizer module instead of raw
  atom-index pairs).
- `build_constrained_system(system, cv_model_path, target_cv_value, kappa)` (line ~171):
  wraps a `TorchForce` in a `CustomCVForce` with a harmonic restraint
  `0.5*kappa*(xi-c)^2`, added to a deep-copied `System`. This is a *restrained-sampling*
  (umbrella-style) formulation, not the current hard-Newton-retraction formulation
  `sample_levelset_projected` uses -- would need either adapting this to a retraction-style
  usage, or switching the whole sampling approach to restrained (softer, possibly avoids
  the Newton-step blowup issue as a side effect, but changes what's being measured/how
  mean force is computed -- worth deciding deliberately, not by default).

## Relevant paths

- `src/amore/mep/constrained.py` -- all of the above, plus `_chi_and_grad`/`_chi_val`
  (`src/amore/mep/core.py`, actually -- the low-level torch helpers), `_minimize_seed`,
  `sample_levelset_projected`.
- `examples/2cm2_full/lib/comfeat.py` -- `make_torch_featurizer_pocket`,
  `load_trained_model_pocket`, `NormalizedChiNet` (the actual model class in use, wraps a
  `ChiNetMulti` trunk with input normalization).
- `/scratch/htc/jkresse/2cm2_mfep/system_flex.xml`, `system_rigid.xml` -- pre-built OpenMM
  Systems for the pocket-model topology (50275 atoms), flexible-bond and
  rigid-water+HBonds-constrained respectively.
- `/scratch/htc/jkresse/2cm2/model_k3_pocket_0_2979_lag20_pbc.pt` -- the trained k=3 pocket
  model checkpoint (`comfeat.load_trained_model_pocket(3, device=...)` loads this).
- `/scratch/htc/fsafarov/conda/envs/molsim/bin/python` -- the env with working CUDA OpenMM
  + torch; run any of this work through it, not the plain AMORE `.venv` (which has no CUDA
  OpenMM).
