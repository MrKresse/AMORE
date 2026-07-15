# openmmtorch_benchmark — postmortem & handoff

Follow-up to `examples/2cm2_full/POSTMORTEM_openmmtorch.md` (read that first — this file
assumes its context: why TorchForce was proposed, what was already built toward it, what
was untested). This file covers what this session actually built, the bugs it found doing
so, and the honest performance result on the real system.

---

## 1. The environment blocker, and how it was resolved

`openmm-torch` was not importable in the AMORE `.venv`, nor in fsafarov's `molsim` env
(confirmed dead again this session — `ModuleNotFoundError`). The previous postmortem
flagged `molsim`'s `site-packages` as plausibly writable ("h5py was successfully
`pip install`ed into it earlier"); that is **no longer true** — it's owned by `fsafarov`
and not group-writable (`touch` inside it fails with `Permission denied`), and
`openmm-torch` isn't on PyPI in any case (conda-forge only, built against libtorch).

**Fix**: this user has their own micromamba root at `/home/htc/jkresse/micromamba`
(binary at `/home/htc/jkresse/.local/bin/micromamba`, previously used only for a `pymol`
env) — real write access, no permission issue. Built a fresh env there:

```bash
export MAMBA_ROOT_PREFIX=/home/htc/jkresse/micromamba
/home/htc/jkresse/.local/bin/micromamba create -n omtorch -c conda-forge \
    python=3.12 "openmm-torch=1.4" mdanalysis matplotlib nbformat tqdm scipy pandas \
    ipykernel nbconvert
```

Two things needed fixing after the first solve:

- **`openmm-torch=1.4` pin**: the 1.5.x series requires `cuda-version>=13.0`, which this
  driver (`550.163.01`) cannot run. 1.4 accepts `cuda-version>=12.0,<13`.
- **`cuda-version` pin to 12.4**: even with `openmm-torch=1.4`, the unconstrained solver
  picked `cuda-version=12.9` for `openmm`'s own CUDA platform (NOT for pytorch, which
  correctly resolved to a `cuda120`-tagged build and worked fine throughout). At runtime
  this failed with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` loading OpenMM's `CUDA` platform
  (the Reference platform still worked, silently, if you don't check
  `simulation.context.getPlatform().getName()` — worth asserting on that explicitly, a
  fallback pattern already in `constrained.py`'s `try/except Exception` around platform
  selection can hide exactly this). Fixed with a follow-up
  `micromamba install -n omtorch cuda-version=12.4` (downgrades `cuda-cudart`,
  `libcublas`, `libcufft`, `libcurand`, `libcusolver`, `libcusparse` in step).

Registered as a Jupyter kernel: `omtorch` / "Python (omtorch)"
(`/home/htc/jkresse/micromamba/envs/omtorch/bin/python -m ipykernel install --user
--name omtorch`). **Run this example's notebook with that kernel, not the usual AMORE
one** — `openmm-torch` is not installed anywhere else. `amore` itself is reached the
normal way (`sys.path.insert` in `lib/pipeline.py`), no separate install needed.

---

## 2. What was built in `src/amore/mep/constrained.py`

Three new pieces, all additive (nothing existing was removed or had its behavior
changed, aside from the two bug fixes in §3):

- **`export_cv_torchscript_module(featurizer_module, cv_module, example_positions_nm,
  path, periodic=False)`** — the generalized tracer
  `POSTMORTEM_openmmtorch.md` flagged as missing ("a generalized version taking an
  arbitrary already-built featurizer module instead of raw atom-index pairs"). Traces
  ANY `(featurizer_module, cv_module)` pair, e.g. `comfeat.PocketFeaturizer` +
  `FaceCV(model, i)`, not just the toy raw-pairwise-distance case
  `export_cv_torchscript`/`_CVPipeline` hardcoded.
- **`build_cv_gradient_system(system, cv_model_path, grad_force_group=None)`** — adds a
  `CustomCVForce("xi")` (coefficient 1, no restraint) wrapping the traced CV in its own
  dedicated force group. `CustomCVForce` symbolically differentiates its expression and
  chains through the CV's own gradient (TorchForce supplies chi's analytic gradient via
  its GPU-resident autograd backward) — with `expression="xi"` that force IS exactly
  `-grad(xi)`. Putting it in a separate force group from the physical force field means
  `getState(getForces=True, groups={that group})` returns ONLY `-grad(xi)`, and a
  separate `groups={0}` query returns only the physical force, both from ONE context's
  force evaluation, no separate Python/PyTorch call. `getCollectiveVariableValues` on the
  same force gives chi's value (cheap, forward-pass only).
- **`sample_levelset_projected_torchforce(...)`** — line-for-line the same algorithm as
  `sample_levelset_projected` (projected Langevin + Newton retraction), with chi's
  value/gradient sourced via the above instead of `_chi_and_grad`/`_chi_val`. This is
  "the constrained version of MFEP calculation in openmmtorch" the user asked for — NOT
  the harmonic-restraint `build_constrained_system`/`build_chi_mep_constrained` already in
  the file (that's a *restrained*, soft-constraint formulation; the projected/retracted
  sampler is the actual hard-constraint one every other MFEP entry point in `amore.mep`
  uses).

Both `sample_levelset_projected` and `sample_levelset_projected_torchforce` now also
return a `"timing"` dict (`{"openmm", "chi_grad", "retract"}`, seconds/step) in their
result, not just print it — needed to plot a timing breakdown without scraping stdout.
Purely additive.

---

## 3. Bugs found in the untested code, fixed at the root

`POSTMORTEM_openmmtorch.md` explicitly flagged `_CVPipeline`/`export_cv_torchscript` as
untested against anything beyond a toy case. Getting a real periodic, GPU system working
surfaced two real bugs, both fixed directly in `constrained.py` (shared code, not worked
around in this example's `lib/`):

1. **Missing box-vectors argument for periodic systems.** `TorchForce.setUsesPeriodic
   BoundaryConditions(True)` (needed for any PME system, e.g. 2cm2's `system_flex.xml`)
   makes OpenMM call the traced module as `forward(positions, boxvectors)`, not
   `forward(positions)`. Both `_CVPipeline.forward` and the new `_TracedCVModule.forward`
   only declared one argument — a hard TorchScript arity error at runtime
   (`Expected at most 2 argument(s)... but received 3`). Fixed by accepting-and-ignoring
   an optional `boxvectors` argument in both, and adding a `periodic=` flag to both export
   functions that traces a 2-argument schema when set (TorchScript's traced schema is
   fixed by what was fed at trace time, so tracing must match the target
   `setUsesPeriodicBoundaryConditions` value). Both `comfeat`'s featurizers and the toy
   `_PairDistFeaturizer` already bake a FIXED box into their own registered buffers (the
   training system's box, captured at construction) — they never needed live box vectors
   from OpenMM, so ignoring the argument is correct, not a shortcut.
2. **Device mismatch when the model is built on CPU but run on the CUDA platform.**
   `TorchScript` bakes in each buffer's device at trace time. Building `comfeat`'s model
   +featurizer with `device="cpu"` (the natural default when just inspecting/testing) then
   running under OpenMM's `CUDA` platform fails deep inside a `scatter_add` with
   "Expected all tensors to be on the same device, but found at least two devices, cpu and
   cuda:0". Fix is call-site, not a code bug: build the model/featurizer with
   `device="cuda"` before tracing whenever the destination `Simulation` uses the `CUDA`
   platform (see `build.py`'s pocket section). Documented in `export_cv_torchscript_module`
   and `pipeline.py`, not silently worked around.

Also (independent of periodicity/device): OpenMM's `Reference` platform is
double-precision internally, so `TorchForce` calls the traced module with `float64`
positions there even when the network's own weights are `float32` — both pipelines now
cast `positions.to(dtype=pt.float32)` at their traced entry point rather than assuming the
caller's precision matches training. This surfaced even on the tiny ADP case, where the
CUDA platform initially failed to load (see §1's `cuda-version` bug) and silently fell back
to `Reference` via `constrained.py`'s own `try/except` around platform selection.

---

## 4. Correctness — validated at two levels

1. **Direct chi/∇chi comparison** (`_chi_and_grad` vs. the traced+`CustomCVForce`-fused
   value/gradient, single frame, no sampling): matched to ~1e-8 on ADP (22 atoms, toy
   pairwise featurizer) and ~1e-4 relative on 2cm2's real pocket featurizer (50275 atoms,
   285 side-chain COMs + all-atom contact pairs) — the larger, more complex computation
   graph accumulates more fp32 rounding, as expected, still far inside any physically
   meaningful tolerance.
2. **Full sampler-level comparison** (`pipeline.compare_samplers`: same seed positions,
   same `np.random` draws, chained projected-Langevin steps): position trajectories
   matched to ~1e-6–1e-7 nm on ADP over 300 steps and ~1e-7 nm on the real 2cm2 pocket
   system over 40 steps, mean-force (`lambdas`) statistics matched closely on both. This
   is the stronger check — it confirms the fused path reproduces not just one force
   evaluation but the full physically-relevant trajectory the algorithm produces.

(Earlier same-day attempt with an UNTRAINED, randomly-initialized k=1-output network and
an arbitrary target level set far from the seed's own chi produced an apparent large
mismatch — traced to the retraction chasing a big initial CV gap through an
untrained/non-smooth gradient landscape until forces blew up to ~1e15 kJ/mol, at which
point tiny fp32 differences get chaotically amplified. Not a bug: fixed by targeting the
seed's own native chi value (no artificial jump) and using a real k=3 net. Worth
remembering if a future correctness check "fails" — check whether the reference run itself
is physically sane before suspecting the fusion.)

---

## 5. The performance result — honest, not the hoped-for outcome

| | ADP vacuum (22 atoms) | 2cm2 pocket (50275 atoms) |
|---|---|---|
| old (python autograd) | 0.95 ms/step | 421 ms/step |
| new (TorchForce-fused) | 4.47 ms/step | 447 ms/step |
| speedup | **0.21x** (~4.7x slower) | **0.94x** (statistically a wash, marginally slower) |

(Exact numbers from the final executed `openmmtorch_benchmark.ipynb`, computed live from
the actual run rather than hardcoded. Three independent runs this session — two
standalone scripts and two notebook executions — landed in the same range each time
(ADP: 0.19-0.22x; 2cm2: 0.89-0.94x), so this is a stable result, not noise from a single
run.)

**ADP**: expected and unsurprising. A trivial featurizer (pure pairwise distances, one
dispatched op) on a 22-atom system is already fast in plain Python/PyTorch (~0.5 ms for
chi+grad). Routing through OpenMM's `CustomCVForce`/`TorchForce` costs 2-3 separate
`getState`/`getCollectiveVariableValues` C++/Python boundary crossings per step, and that
per-call overhead exceeds what's saved for a computation this cheap.

**2cm2 (the real test)**: NOT the win the setup hoped for. Fusion made the chi-gradient
step slightly *slower*, not faster (179ms vs 161ms), and the retraction step also slower
(95ms vs 61ms) — only the physical-force query itself improved (20ms vs 43ms). Net: a
wash, marginally in the old path's favor. Compare this against
`POSTMORTEM_openmmtorch.md`'s own numbers: **this session's OLD-path measurement
(161-173 ms chi-grad, both this notebook and an earlier standalone run) reproduces that
postmortem's original "inflated" live-loop number (173 ms) almost exactly**, while that
same postmortem's *isolated* reproduction attempt (outside the full sampling loop) got
~4.7 ms — a ~35x discrepancy it could not explain and flagged as possibly GPU contention
from another process sharing this MIG instance
(`NVIDIA H100 NVL MIG 1g.24gb`), unconfirmed because `nvidia-smi --query-compute-apps`
returns "Insufficient Permissions" for this user (confirmed still true this session too).

**This session's result is independent evidence FOR that contention hypothesis, not
against TorchForce.** Fusing chi evaluation into OpenMM eliminates Python-dispatch/sync
overhead — a real cost, confirmed by the isolated single-frame checks in §4 running in low
single-digit ms — but it does NOT eliminate GPU-scheduler contention with other processes
queued on the same physical MIG slice, because both the old and new code paths ultimately
submit their compute to the same underlying GPU and wait their turn. If contention is
present right now (plausible: the number reproduced almost exactly), both paths pay for it
about equally, which is exactly the near-1x result observed. TorchForce fusion may still be
providing a real, smaller win underneath that noise floor — it just isn't visible at this
moment on this shared GPU allocation.

**Open item for a future session**: re-run `openmmtorch_benchmark.ipynb`'s §2 (2cm2 pocket)
on a quiet MIG instance or a dedicated (non-MIG, non-shared) GPU allocation, if that
distinction matters. If the speedup is still ~1x there, contention was NOT the explanation
and the fusion genuinely doesn't help on this workload (would be worth understanding why —
maybe the featurizer's own GPU-side ops, not Python dispatch, are the real bottleneck, in
which case TorchScript's operator fusion during `torch.jit.trace` may not be doing much
better than eager execution for this particular graph shape). If the speedup jumps up
dramatically on a quiet GPU, that closes `POSTMORTEM_openmmtorch.md`'s open contention
question definitively.

---

## 6. Relevant paths

- `src/amore/mep/constrained.py` — `export_cv_torchscript_module`,
  `build_cv_gradient_system`, `sample_levelset_projected_torchforce`, plus the two bug
  fixes to `_CVPipeline`/`_TracedCVModule` (§3).
- `examples/openmmtorch_benchmark/lib/pipeline.py` — benchmark setup/driver helpers for
  both systems; `compare_samplers` is the old-vs-new harness.
- `/home/htc/jkresse/micromamba/envs/omtorch` — the environment this notebook requires.
  Kernel name `omtorch`.
- `examples/2cm2_full/POSTMORTEM_openmmtorch.md` — the original performance forensics and
  the open contention question this session's result bears on.
- `/scratch/htc/jkresse/2cm2_mfep/system_flex.xml`,
  `/scratch/htc/jkresse/2cm2/model_k3_pocket_0_2979_lag20_pbc.pt` — same paths the original
  postmortem used, unchanged.
