# Julia Benchmark Specification

**Purpose:** Run the exact same benchmark as `benchmark_v3` in Julia using the
`ISOKANN.jl` implementation at `C:\Users\kr3ss\Desktop\ZIBwork\NEWISOKANN\ISOKANN.jl`.
The goal is a final verdict on the Python port in `AMORE/src/amore/isotarget.py`:
do the Python and Julia isotarget transforms produce the same results?

Read `BENCHMARK_V3_RESULTS.md` (in this directory) before starting — it contains
the reference numbers you need to match and the full benchmark design.

---

## 1. What You Are Comparing

The Python benchmark (`02_train_benchmark_v3.py`) imports from
`AMORE/src/amore/isotarget.py`, which is a Python port of
`NEWISOKANN/ISOKANN.jl/src/isotarget.jl`.  The port fixed several bugs from
v2 (see `BENCHMARK_V2_POSTMORTEM.md`).  The Julia benchmark uses isotarget.jl
directly, on the same saved simulation data, same architecture, same
hyperparameters.  Any difference in outcome is a divergence between the port
and the original.

**Isotarget variants to compare (4 core):**

| Python label | Python call | Julia struct |
|---|---|---|
| V2-ISA | `apply_target("isa", chi, kchi)` | `TransformISA()` |
| V3-GramSchmidt | `apply_target("gramschmidt", chi, kchi)` | `TransformGramSchmidt2()` ← NOT GramSchmidt1 |
| V4-PseudoInv | `apply_target("pseudoinv", chi, kchi)` | `TransformPseudoInv()` (see §5) |
| V5-Cross | `apply_target("cross", chi, kchi)` | `TransformCross(npoints=N, maxcols=3*k)` |

**Not in isotarget.jl (skip or add separately):**
- V6-Power-Iter: Python uses `power_method_multi` (subspace orthogonal iteration).
  Julia has `TransformSVD` but that is the old DMD `eigen(H)` approach — the v2 bug.
  Add Power-Iter as a separate condition using the algorithm in
  `AMORE/src/amore/isokann/power.py` translated to Julia (see §6).
- B-VAMP2: Not in isotarget.jl. Skip unless you implement it.

---

## 2. Environment Setup

### 2.1 Julia version

ISOKANN.jl requires Julia ≥ 1.10 (Manifest-v1.12.toml is present for 1.12+).
The Project.toml is at:

```
NEWISOKANN/ISOKANN.jl/Project.toml
```

### 2.2 Activate and instantiate

From the Julia REPL or a script preamble:

```julia
import Pkg
Pkg.activate(raw"C:\Users\kr3ss\Desktop\ZIBwork\NEWISOKANN\ISOKANN.jl")
Pkg.instantiate()   # downloads all declared deps — takes a few minutes first run
```

### 2.3 Add NPZ.jl for data loading

NPZ.jl is not in the existing Project.toml.  Add it:

```julia
Pkg.add("NPZ")
```

Or use PythonCall (already declared) to load via numpy — see §3.

### 2.4 Verify the environment

```julia
using ISOKANN
using Flux
sim = ISOKANN.Triplewell()
iso = Iso(sim; nk=8, model=ISOKANN.densenet(layers=[2,16,16,1], activation=tanh))
run!(iso, 5, 10)
println("Environment OK")
```

If this runs without error, proceed.

---

## 3. Loading the Simulation Data

All data files are in:
```
AMORE/examples/benchmark/data/
  triple_well_koopman.npz     — TW anchors + bursts
  alanine_koopman.npz         — ADP 5ps anchors + bursts
  alanine_0p1ps_koopman.npz   — ADP 0.1ps bursts (same anchors)
```

### 3.1 Load via NPZ.jl

```julia
using NPZ

tw   = npzread(raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark\data\triple_well_koopman.npz")
al5  = npzread(raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark\data\alanine_koopman.npz")
al01 = npzread(raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark\data\alanine_0p1ps_koopman.npz")
```

### 3.2 Array layouts

The Python files use (N, ...) row-major layout.  Julia is column-major.
**Transpose on load** so that the feature dimension is first (required by Flux):

```julia
# Triple-well
# tw["anchors"]      (1600, 2)  float32  → anchors_feat: (2, 1600)
# tw["bursts"]       (1600, 20, 2)        → bursts_feat:  (2, 20, 1600)
# tw["patch_splits"] (5, 1600)  int8
anchors_tw   = Float32.(tw["anchors"]')                  # (2, 1600)
bursts_tw    = permutedims(Float32.(tw["bursts"]), (3,2,1))  # (2, 20, 1600)
splits_tw    = tw["patch_splits"]                         # (5, 1600)

# ADP 5ps
# al5["anchors_feat"] (1578, 231)
# al5["bursts_feat"]  (1578, 20, 231)
# al5["patch_splits"] (5, 1578)
anchors_al   = Float32.(al5["anchors_feat"]')            # (231, 1578)
bursts_al5   = permutedims(Float32.(al5["bursts_feat"]), (3,2,1))  # (231, 20, 1578)
bursts_al01  = permutedims(Float32.(al01["bursts_feat"]), (3,2,1)) # (231, 20, 1578)
splits_al    = al5["patch_splits"]                        # (5, 1578)
```

### 3.3 Build SimulationData-equivalent tuples

The ISOKANN training loop expects `data` from which it can call
`features(data)` → (F, N) and `propfeatures(data)` → (F, K, N).
The simplest approach is to pass a plain tuple `(xs, ys)` where:
- `xs = anchors`                     shape (F, N)
- `ys = bursts`                      shape (F, K, N)

The isotarget functions call `model(xs)` and `expectation(model, ys)`.
Check `src/simulation.jl` — `features` and `propfeatures` accept tuples:

```julia
# Confirm:
using ISOKANN: features, propfeatures
data = (anchors_tw, bursts_tw)
features(data)      # should return anchors_tw  (2, 1600)
propfeatures(data)  # should return bursts_tw   (2, 20, 1600)
```

If this doesn't work, use `SimulationData` with an `ExternalSimulation` wrapper,
or directly construct the target by calling:
```julia
isotarget(transform, model, features(data), propfeatures(data))
```

---

## 4. Network Architecture

**Python spec (ChiNetMultiLinear):**
```
in_dim → 128 → 32 → 8 → k
Tanh hidden, Linear output, NO LayerNorm
Adam lr=1e-3
```

**Julia equivalent:**
```julia
using Flux, ISOKANN

function benchmark_net(in_dim::Int, k::Int)
    ISOKANN.densenet(
        layers       = [in_dim, 128, 32, 8, k],
        activation   = tanh,           # Tanh hidden — NOT sigmoid
        lastactivation = identity,     # Linear output
        layernorm    = false,          # No LayerNorm
    )
end

# Triple-well
net_tw = benchmark_net(2, 3)

# ADP (231 pairwise distances)
net_al = benchmark_net(231, 3)
```

**Critical:** use `tanh` not `Flux.sigmoid`.  The Python port uses Tanh hidden
layers specifically because the isotarget functions produce targets outside [0,1]
(GramSchmidt, Cross scale to ~±√N).  Sigmoid hidden layers caused training
failure in earlier experiments.

**Optimizer:**
```julia
opt = Flux.setup(Flux.Adam(1f-3), net)
```

No weight decay (Python uses plain Adam without regularization).

---

## 5. Variant-by-Variant Implementation

### 5.1 TransformISA

```julia
t = TransformISA(permute=true, whitening=false)
```

`whitening=false` matches the Python port's `isa_target` which does NOT whiten
before the indexmap call.  (The v2 postmortem notes that whitening was dropped in
the port — verify this is intentional before finalizing.)

### 5.2 TransformGramSchmidt2 ← use this one, not GramSchmidt1

```julia
t = TransformGramSchmidt2()
```

`TransformGramSchmidt2` applies `c = sqrt(size(chi, 2))` scaling (line ~245 in
isotarget.jl).  `TransformGramSchmidt1` does NOT — it is the old broken version.
The v3 Python fix added `* np.sqrt(n)` back; Julia's GramSchmidt2 has it.
Verify they agree numerically on a small test case before running the full
benchmark (see §7.1).

### 5.3 TransformPseudoInv

The Julia code has three variants: `TransformPinv1`, `TransformPinv2`,
`TransformPinv3`, and the older `TransformPseudoInv()` (check if this name
exists in the current codebase — it appears in the test suite).  Use:

```julia
t = TransformPseudoInv()
```

If that name doesn't exist, find the variant that most closely matches the Python
`pseudoinv_target`, which computes `chi @ pinv(kchi)` projected onto the dominant
Schur subspace.  From a quick read of isotarget.jl lines 415–475,
`TransformPinv1` uses `L * pinv(R)` with `partialschur` — this matches the
Python logic.

### 5.4 TransformCross

```julia
k = 3
t = TransformCross(npoints=size(anchors, 2), maxcols=3*k)
```

`maxcols=3*k=9` matches the Python `_CrossHist(maxcols=k_out * 3)`.
The Cross transform accumulates history in-place; construct a fresh instance
for each (variant, seed) run.

---

## 6. Power-Iter in Julia (translate from Python)

The Julia `TransformSVD` is the **old v2 DMD** method — it computes
`eigen(H)` where `H = U' K[χ] V S⁻¹`.  This is the bug documented in
`BENCHMARK_V2_POSTMORTEM.md` bug 3.

Implement `power_method_multi` in Julia.  The algorithm (from
`AMORE/src/amore/isokann/power.py`):

```julia
using LinearAlgebra, Flux

function power_method_multi!(model, xs, ys;
        n_iter=100, epochs_per_iter=50, lr=1f-3,
        lr_decay=0.97f0, batch=2048, collapse_eps=1f-3,
        verbose=true)

    opt = Flux.setup(Flux.Adam(lr), model)
    k   = size(model(xs[:, 1:1]), 1)

    losses = Float32[]
    spans  = zeros(Float32, n_iter, k)

    for it in 1:n_iter
        # 1. Koopman action: evaluate on propagated endpoints
        #    ys shape: (F, K, N) — flatten to (F, K*N), evaluate, reshape
        F, K, N = size(ys)
        ys_flat = reshape(ys, F, K*N)
        Flux.testmode!(model)
        Y = model(ys_flat)                    # (k, K*N)
        Y = reshape(Y, k, K, N)
        Y = dropdims(mean(Y; dims=2); dims=2) # (k, N) — average over bursts

        # 2. Orthogonal deflation via SVD
        Yc  = Y .- mean(Y; dims=2)
        U_f, _, _ = svd(Yc)                   # U_f: (k, min(k,N)) orthonormal columns
        Y_orth = Yc' * U_f                    # (N, k) — project anchors
        Y_orth = Y_orth'                      # (k, N)

        # Collapse guard
        for j in 1:k
            if std(Y_orth[j, :]) < collapse_eps
                Y_orth[j, :] .+= collapse_eps .* randn(Float32, N)
            end
        end

        # 3. Scale each row to [0, 1]
        for j in 1:k
            lo, hi = extrema(Y_orth[j, :])
            Y_orth[j, :] .= (Y_orth[j, :] .- lo) ./ max(hi - lo, 1f-6)
        end
        targets = Y_orth   # (k, N) — target for xs

        # 4. Inner SGD: train model(xs) → targets
        Flux.trainmode!(model)
        epoch_loss = 0f0
        for _ in 1:epochs_per_iter
            idx  = randperm(N)[1:min(batch, N)]
            loss, grads = Flux.withgradient(model) do m
                Flux.mse(m(xs[:, idx]), targets[:, idx])
            end
            Flux.update!(opt, model, grads[1])
            epoch_loss += loss
        end
        push!(losses, epoch_loss / epochs_per_iter)

        # LR decay — rebuild optimizer state with decayed lr each outer iteration
        opt = Flux.setup(Flux.Adam(lr * lr_decay^(it-1)), model)

        # Diagnostics
        Flux.testmode!(model)
        chi_all = model(xs)
        spans[it, :] = vec(maximum(chi_all; dims=2) .- minimum(chi_all; dims=2))
        verbose && it % 5 == 0 && @printf("iter %3d  loss=%.5f  spans=%s\n",
            it, losses[end], join(round.(spans[it,:]; digits=3), "  "))
    end

    return (; losses, spans)
end
```

**Note on LR decay in Flux:** Flux's Optimisers API doesn't have a simple
`scheduler.step()` equivalent.  The cleanest approach is to pass the current LR
as a closure or use `Optimisers.adjust!` on the optimizer state:

```julia
Optimisers.adjust!(opt, lr * lr_decay^(it-1))
```

Check the Optimisers.jl docs for the exact signature in the version declared in
Project.toml.

---

## 7. Training Loop

### 7.1 Sanity-check a single transform first

Before running the full benchmark, verify that GramSchmidt2 agrees with the
Python `gramschmidt_target` on a small example:

```julia
# Python: gramschmidt_target(kchi)  where kchi is (k, N)
# Julia:  isotarget(TransformGramSchmidt2(), model, xs, ys)
# Both should return arrays with RMS ≈ 1/sqrt(k) per row
chi = randn(Float32, 3, 100)   # mock chi, shape (k, N)
kchi = randn(Float32, 3, 100)  # mock kchi

# Manually call the Julia isotarget logic:
# GramSchmidt2: q, r = qr(kchi'); target = q' .* sign.(diag(r))  then scale by sqrt(N)
using LinearAlgebra
c = sqrt(size(kchi, 2))
q, r = qr(kchi')
t_jl = Matrix(q)' .* sign.(diag(r))   # (k, N)
@show rms_jl = sqrt(mean(t_jl .^ 2))  # should be ≈ 1/sqrt(k)
```

Compare to the Python side:
```python
from amore.isotarget import gramschmidt_target
import numpy as np
chi  = np.random.randn(3, 100).astype(np.float32)
kchi = np.random.randn(3, 100).astype(np.float32)
t_py = gramschmidt_target(kchi)
print("rms_py =", np.sqrt(np.mean(t_py**2)))
```

Both should be ≈ `1/sqrt(3) ≈ 0.577`.  If they disagree, diagnose before
proceeding.

### 7.2 Warm-up (isotarget variants only, NOT Power-Iter)

Identical to Python: 100 outer iterations of GramSchmidt2 on all k outputs
before switching to the variant's own target.

```julia
const WARMUP = 100

function run_with_warmup!(model, data, variant_target, opt;
        warmup=WARMUP, max_iter=5000, min_iter=1000,
        epochs=1, W=500, rel_tol=1f-3)

    xs  = features(data)    # (F, N)
    ys  = propfeatures(data) # (F, K, N)
    gs  = TransformGramSchmidt2()

    val_history = Float32[]
    best_val    = Inf
    best_params = deepcopy(Flux.state(model))

    for it in 1:max_iter
        # ── Target selection ──────────────────────────────────────────────
        target = if it <= warmup
            isotarget(gs, model, xs, ys)
        else
            try
                isotarget(variant_target, model, xs, ys)
            catch e
                e isa DomainError || rethrow(e)
                it % 100 == 0 && @printf("  [it=%d degenerate: %s]\n", it, e.msg)
                push!(val_history, NaN32)
                continue
            end
        end

        # ── Inner SGD ─────────────────────────────────────────────────────
        for _ in 1:epochs
            loss = Flux.train!(model, [(xs, target)], opt) do m, x, t
                Flux.mse(m(x), t)
            end
            push!(val_history, loss)
        end

        # ── Best checkpoint ───────────────────────────────────────────────
        val = val_history[end]
        if isfinite(val) && val < best_val
            best_val = val
            best_params = deepcopy(Flux.state(model))
        end

        # ── Early stopping ────────────────────────────────────────────────
        if it >= min_iter && length(val_history) >= W
            window = filter(isfinite, val_history[end-W+1:end])
            if !isempty(window)
                med = median(window)
                rng = maximum(window) - minimum(window)
                rng < rel_tol * abs(med) && break
            end
        end
    end

    # Restore best
    Flux.loadmodel!(model, best_params)
    return val_history
end
```

**Flux.train! API note:** The exact calling convention depends on the Flux
version in Project.toml.  In Flux 0.14+:

```julia
# Per-sample gradient update:
grads = Flux.gradient(model) do m
    Flux.mse(m(xs), target)
end
Flux.update!(opt, model, grads[1])
```

Check `using Flux; @doc Flux.train!` in the active environment for the
installed version.

### 7.3 Full benchmark runner

```julia
using Random, Statistics, Printf, NPZ, Flux, LinearAlgebra
using ISOKANN

# Hyperparameters — define once at the top of your script
# (duplicated here for reference; remove the first definition in §7.2 if combining)
# MAX_ITER=5000  MIN_ITER=1000  W=500  REL_TOL=1f-3  WARMUP=100  LR=1f-3  N_SEEDS=5  K=3

VARIANTS_JL = [
    ("isa",         () -> TransformISA()),
    ("gramschmidt", () -> TransformGramSchmidt2()),
    ("pseudoinv",   () -> TransformPseudoInv()),   # or TransformPinv1 — check §5.3
    ("cross",       (N) -> TransformCross(npoints=N, maxcols=3*K)),
    # ("svd_power",   () -> nothing),  # handled separately by power_method_multi!
]

function run_julia_benchmark(ds_name, anchors, bursts, splits, out_dir)
    F, N = size(anchors)
    mkpath(out_dir)

    for (vname, make_target) in VARIANTS_JL
        for seed in 0:N_SEEDS-1
            seed_dir = joinpath(out_dir, ds_name, vname, "seed_$seed")
            isfile(joinpath(seed_dir, "chi_best.npy")) && continue
            mkpath(seed_dir)

            Random.seed!(seed * 12345 + 7)
            model = benchmark_net(F, K)
            opt   = Flux.setup(Flux.Adam(LR), model)

            tr_mask = splits[seed+1, :] .== 0   # test mask unused: power/vamp2 use full data
            xs_tr   = anchors[:, tr_mask]
            ys_tr   = bursts[:, :, tr_mask]
            data_tr = (xs_tr, ys_tr)

            t = vname == "cross" ? make_target(sum(tr_mask)) : make_target()

            val_hist = run_with_warmup!(model, data_tr, t, opt)

            chi_best   = cpu(model(anchors))   # (k, N)
            chi_atstop = chi_best

            npzwrite(joinpath(seed_dir, "chi_best.npy"),    chi_best')
            npzwrite(joinpath(seed_dir, "chi_atstop.npy"), chi_atstop')
            npzwrite(joinpath(seed_dir, "val_loss.npy"),   Float32.(val_hist))

            sd_final = std(chi_best; dims=2)[:]
            k_eff    = sum(sd_final .> 0.05f0)
            @printf("  %-15s seed=%d  iters=%d  sd=[%s]  k_eff=%d\n",
                vname, seed, length(val_hist),
                join(round.(sd_final; digits=3), ", "), k_eff)
        end
    end
end
```

---

## 8. Datasets to Run

Run the same four datasets as the Python benchmark:

```julia
# ── Triple-well ──────────────────────────────────────────────────────────────
run_julia_benchmark("triple_well", anchors_tw, bursts_tw, splits_tw,
    raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v3\runs_julia")

# ── ADP 5ps ──────────────────────────────────────────────────────────────────
run_julia_benchmark("alanine_5ps", anchors_al, bursts_al5, splits_al,
    raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v3\runs_julia")

# ── ADP 0.1ps ─────────────────────────────────────────────────────────────────
run_julia_benchmark("alanine_0p1ps", anchors_al, bursts_al01, splits_al,
    raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v3\runs_julia")

# ── ADP multi-tau (5ps + 0.1ps joint) ────────────────────────────────────────
bursts_joint = cat(bursts_al5, bursts_al01; dims=2)   # (231, 40, 1578)
run_julia_benchmark("alanine_multitau", anchors_al, bursts_joint, splits_al,
    raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v3\runs_julia")
```

Outputs go to `runs_julia/` (a sibling of the existing `runs/` directory) so
they do not overwrite Python results.

---

## 9. Plotting and Scoring

The Python plot script (`03_plot_benchmark_v3.py`) reads from `runs/`.
Modify it to also read from `runs_julia/` by duplicating the `compute_metrics`
and `write_results_md` calls with `RUNS_DIR = runs_julia/`.  The scoring
functions (`pearson_r`, `hungarian_match_r`, `tw_shape_r`, `adp_shape_r`) are
purely numpy and work on any chi_best.npy files regardless of where they came
from.

Or produce a second markdown:

```python
# In 03_plot_benchmark_v3.py, add at the bottom:
RUNS_DIR_JULIA = os.path.join(HERE, "runs_julia")
if os.path.isdir(RUNS_DIR_JULIA):
    # re-run compute_metrics with RUNS_DIR overridden
    ...
    write_results_md(results_julia, tw_refs, adp_refs,
                     out_path=os.path.join(HERE, "BENCHMARK_V3_JULIA_RESULTS.md"))
```

Add an `out_path` parameter to `write_results_md` to support writing a second file.

---

## 10. What to Check for the Verdict

For each (dataset, variant) pair, compare:

| Metric | Python result (from BENCHMARK_V3_RESULTS.md) | Julia result | Acceptable delta |
|--------|----------------------------------------------|--------------|-----------------|
| Mean r (Pearson vs reference) | see table | your result | ≤ 0.05 |
| SD r across seeds | see table | your result | ≤ 0.03 |
| k_eff (modes with SD > 0.05) | see table | your result | ≤ 0.5 |
| Verdict | PASS/PARTIAL/FAIL | your result | must match |

**Expected outcome if the port is correct:**
- GramSchmidt: Julia ≈ Python, both PASS on TW (r≈0.86) and ADP-5ps (r≈0.82)
- ISA: Julia ≈ Python, both PARTIAL (r≈0.80, k_eff=0)
- Cross: Julia ≈ Python, both PARTIAL on TW (r≈0.77)

**A mismatch is a bug in the Python port.** The most likely places are:
1. `gramschmidt_target`: the `sqrt(n)` factor (GramSchmidt1 vs GramSchmidt2)
2. `isa_target`: the `whitening` flag default (Python uses `whitening=False`)
3. `cross_target`: the `maxcols` accumulator size
4. Array layout — Julia is (k, N), Python is (k, N) too — but check any
   transpose discrepancies in `expectation` vs `kchi_avg`

---

## 11. File Layout After Completion

```
AMORE/examples/benchmark_v3/
  runs/                          ← Python results (already done)
  runs_julia/                    ← Julia results (to be created)
    triple_well/
      isa/seed_0/{chi_best.npy, chi_atstop.npy, val_loss.npy}
      gramschmidt/...
      pseudoinv/...
      cross/...
    alanine_5ps/...
    alanine_0p1ps/...
    alanine_multitau/...
  BENCHMARK_V3_RESULTS.md        ← Python verdicts (existing)
  BENCHMARK_V3_JULIA_RESULTS.md  ← Julia verdicts (to be created)
  benchmark_v3_julia.jl          ← the Julia script you write
```

---

## 12. Critical Implementation Notes

1. **Array convention.** Julia isotarget functions expect `xs` shape `(F, N)` and
   `ys` shape `(F, K, N)`.  The `expectation` function averages over dim 2.
   Verify: `size(expectation(model, ys))` should be `(k, N)`.

2. **GramSchmidt2 vs GramSchmidt1.** This is the most important choice.
   `GramSchmidt2` has the `sqrt(N)` scaling that matches the Python port's fix.
   Using `GramSchmidt1` reproduces the v2 bug.

3. **ISA whitening.** The Python `isa_target` uses `whitening=False` by default.
   `TransformISA(whitening=false)` is the correct Julia equivalent.

4. **Seeds.** Python seeds as `seed * 12345 + 7`.  Julia: `Random.seed!(seed * 12345 + 7)`.
   Both use the same seeds (0–4).  Flux/Julia will still produce different random
   initializations than Python/PyTorch, so results will differ numerically but
   should be statistically equivalent across 5 seeds.

5. **chi_best.npy convention.** Save as `(N, k)` (transposed from Julia's `(k, N)`)
   to match the Python convention expected by `03_plot_benchmark_v3.py`.
   All `np.save` in the Python script stores `(N, k)` arrays; `npzwrite` in Julia
   with `chi_best'` handles this.

6. **No sigmoid output.** `lastactivation=identity` is mandatory.  The isotarget
   transforms produce targets outside [0,1]; sigmoid would saturate.

7. **TransformPseudoInv name.** If `TransformPseudoInv()` does not exist as a
   standalone name in the current isotarget.jl, use `TransformPinv1` initialized
   with appropriate history size, or grep the source for the exact exported name:
   ```julia
   using ISOKANN
   names(ISOKANN, all=true) .|> string |> x -> filter(s -> occursin("Pinv", s) || occursin("Pseudo", s), x)
   ```
