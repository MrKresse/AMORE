# benchmark_v3_julia.jl
#
# Julia counterpart of 02_train_benchmark_v3.py.  Runs the SAME benchmark
# (same saved simulation data, same architecture, same hyperparameters) using
# the genuine ISOKANN.jl isotarget transforms, to decide whether the Python
# port in AMORE/src/amore/isotarget.py reproduces the Julia original.
#
# Rather than `using ISOKANN` (which loads OpenMM/CondaPkg, Chemfiles, Plots,
# Molly, ... — heavy and fragile), we `include` the genuine src/isotarget.jl
# into a minimal module that supplies exactly the names it references.  The
# transform CODE is therefore byte-for-byte the upstream original; only the
# surrounding harness is ours.
#
# Outputs -> runs_julia/{dataset}/{variant}/seed_{s}/{chi_best,chi_atstop,
#            val_loss,chi_sd_history}.npy   (chi saved transposed to (N,k)).

import Pkg
const ISO_PROJ = raw"C:\Users\kr3ss\Desktop\ZIBwork\NEWISOKANN\ISOKANN.jl"
Pkg.activate(ISO_PROJ)

using LinearAlgebra
using Statistics
using Random
using Printf

# ── Dependency-free IO (NPZ.jl unavailable: adding it breaks env resolution) ──
const JLDATA = raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v3\_jldata"

# Read a raw binary written C-order by _export_data.py, reshape to Julia dims.
function readbin(name, ::Type{T}, dims) where {T}
    bytes = read(joinpath(JLDATA, name * ".bin"))
    Array{T}(reshape(reinterpret(T, bytes), dims))
end

# Minimal .npy writer (numpy-loadable). fortran_order=True dumps Julia's native
# column-major bytes directly; numpy reconstructs the given shape correctly.
function npywrite(path, A::Array{Float32})
    dims = size(A)
    shapestr = length(dims) == 1 ? "($(dims[1]),)" : "(" * join(dims, ", ") * ")"
    hdr = "{'descr': '<f4', 'fortran_order': True, 'shape': $shapestr, }"
    total = 10 + length(hdr) + 1                      # 6 magic + 2 ver + 2 hlen + body
    pad = (64 - (total % 64)) % 64
    hdrfull = hdr * repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93]); write(io, codeunits("NUMPY")); write(io, UInt8[0x01, 0x00])
        write(io, UInt16(length(hdrfull)))            # little-endian on x86
        write(io, codeunits(hdrfull))
        write(io, A)
    end
end

# ── Minimal host module for the genuine isotarget.jl ──────────────────────────
module IsoT
    using LinearAlgebra
    using Statistics: mean
    import Combinatorics
    import CUDA
    using CUDA: CuArray, CuMatrix, cu
    import PCCAPlus
    using Functors: @functor
    import Flux
    using Flux: cpu, gpu
    # Types referenced only in method signatures of functions we never call.
    struct Iso end
    struct SimulationData end
    include(joinpath(raw"C:\Users\kr3ss\Desktop\ZIBwork\NEWISOKANN\ISOKANN.jl", "src", "isotarget.jl"))
end
using .IsoT: isotarget, expectation,
             TransformISA, TransformGramSchmidt2, TransformPseudoInv, TransformCross
import Flux
import Optimisers

# ── Hyperparameters (match 02_train_benchmark_v3.py) ──────────────────────────
const MAX_ITER  = 5000
const MIN_ITER  = 1000
const W         = 500
const REL_TOL   = 1f-3
const WARMUP    = 100
const LR        = 1f-3
const GRAD_CLIP = 5f0
const N_SEEDS   = 5
const K         = 3
const POWER_N_ITER          = 100
const POWER_EPOCHS_PER_ITER = 50

const RUNS_DIR = raw"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v3\runs_julia"

# ── Network: ChiNetMultiLinear equivalent (Tanh hidden, linear out, no norm) ──
benchmark_net(in_dim::Int, k::Int) = Flux.Chain(
    Flux.Dense(in_dim => 128, tanh),
    Flux.Dense(128 => 32, tanh),
    Flux.Dense(32 => 8, tanh),
    Flux.Dense(8 => k),
)

make_opt(model) = Flux.setup(
    Optimisers.OptimiserChain(Optimisers.ClipNorm(GRAD_CLIP), Optimisers.Adam(LR)),
    model)

# Per-mode standard deviation of chi over anchors. chi: (k, N) -> (k,)
chi_sd(chi) = vec(std(chi; dims=2))

function plateau_stop(val_history, it)
    (it < MIN_ITER || length(val_history) < W) && return false
    recent = val_history[end-W+1:end]
    finite = filter(isfinite, recent)
    length(finite) < W ÷ 2 && return false
    rng = maximum(finite) - minimum(finite)
    med = median(finite)
    return rng < REL_TOL * max(abs(med), 1f-12)
end

# Build a fresh transform instance for a variant. Cross needs npoints.
function make_transform(variant::String, npoints::Int)
    variant == "isa"         && return TransformISA(permute=true, whitening=false)
    variant == "gramschmidt" && return TransformGramSchmidt2()
    variant == "pseudoinv"   && return TransformPseudoInv()
    variant == "cross"       && return TransformCross(npoints=npoints, maxcols=3K)
    error("unknown variant $variant")
end

# ── Isotarget variant training (mirrors run_isotarget in the Python script) ───
# `warmup` = number of leading GramSchmidt-on-all-k iterations (set 0 to disable).
function run_isotarget(variant, xs_tr, ys_tr, xs_te, ys_te, xs_all, model; warmup=WARMUP)
    opt = make_opt(model)
    gs  = TransformGramSchmidt2()
    t   = make_transform(variant, size(xs_tr, 2))

    val_history = Float32[]
    sd_history  = Vector{Float32}[]
    best_val = Inf32
    best_chi = nothing

    for it in 1:MAX_ITER
        if it <= warmup
            target = isotarget(gs, model, xs_tr, ys_tr)
            loss, grads = Flux.withgradient(m -> Flux.mse(m(xs_tr), target), model)
            Flux.update!(opt, model, grads[1])
            push!(val_history, loss)
            push!(sd_history, chi_sd(model(xs_all)))
            continue
        end

        # main loop: variant target (ISA/PseudoInv raise DomainError when degenerate)
        target = nothing
        try
            target = isotarget(t, model, xs_tr, ys_tr)
        catch e
            e isa DomainError || rethrow(e)
            push!(val_history, NaN32)
            push!(sd_history, chi_sd(model(xs_all)))
            continue
        end

        loss, grads = Flux.withgradient(m -> Flux.mse(m(xs_tr), target), model)
        Flux.update!(opt, model, grads[1])

        # validation on the held-out test split (fresh, history-less Cross)
        val = NaN32
        try
            tgt_te = variant == "cross" ?
                isotarget(make_transform("cross", size(xs_te, 2)), model, xs_te, ys_te) :
                isotarget(t, model, xs_te, ys_te)
            val = Flux.mse(model(xs_te), tgt_te)
        catch e
            e isa DomainError || rethrow(e)
        end
        push!(val_history, val)

        chi_all = model(xs_all)
        push!(sd_history, chi_sd(chi_all))
        if isfinite(val) && val < best_val
            best_val = val
            best_chi = copy(chi_all)
        end

        plateau_stop(val_history, it) && break
    end

    chi_atstop = model(xs_all)
    best_chi === nothing && (best_chi = chi_atstop)
    return val_history, sd_history, best_chi, chi_atstop
end

# ── Power iteration (faithful translation of src/amore/isokann/power.py) ──────
function power_method_multi(model, x0, x1; n_iter=POWER_N_ITER,
        epochs_per_iter=POWER_EPOCHS_PER_ITER, lr=LR, lr_decay=0.97f0,
        batch=2048, collapse_eps=1f-3)
    opt = Flux.setup(Optimisers.Adam(lr), model)
    n = size(x0, 2)
    k = size(model(x0[:, 1:1]), 1)
    losses = Float32[]
    sds = Vector{Float32}[]

    for it in 1:n_iter
        Y  = model(x1)                       # (k, n)
        Yc = Y .- mean(Y; dims=2)            # center each row (= each chi column)
        Yt = Matrix(Yc')                     # (n, k)
        U  = svd(Yt).U                       # (n, k) orthonormal columns
        U  = Matrix(U)
        for j in 1:k                         # collapse guard
            if std(@view U[:, j]) < collapse_eps
                U[:, j] .+= collapse_eps .* randn(Float32, n)
            end
        end
        tgt = Matrix{Float32}(undef, k, n)   # scale each column to [0,1]
        for j in 1:k
            lo, hi = extrema(@view U[:, j])
            @views tgt[j, :] .= (U[:, j] .- lo) ./ (hi - lo + 1f-6)
        end

        epoch_loss = 0f0
        for _ in 1:epochs_per_iter
            idx = randperm(n)[1:min(batch, n)]
            l, grads = Flux.withgradient(m -> Flux.mse(m(x0[:, idx]), tgt[:, idx]), model)
            Flux.update!(opt, model, grads[1])
            epoch_loss += l
        end
        push!(losses, epoch_loss / epochs_per_iter)
        push!(sds, vec(maximum(model(x0); dims=2) .- minimum(model(x0); dims=2)) ./ (2f0 * sqrt(3f0)))
        Optimisers.adjust!(opt, lr * lr_decay^it)   # ExponentialLR steps after each iter
    end
    return losses, sds
end

# ── Dataset runner ────────────────────────────────────────────────────────────
const VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd_power"]

function run_dataset(ds_name, anchors, bursts, splits;
        variants=VARIANTS, warmup=WARMUP, suffix="")
    F, N = size(anchors)
    println("\n", "="^65)
    @printf("Dataset: %s   F=%d  N=%d  k=%d  warmup=%d\n", ds_name, F, N, K, warmup)
    println("="^65)

    for variant in variants
        outvar = variant * suffix
        @printf("\n  Variant: %s\n", outvar)
        for seed in 0:N_SEEDS-1
            seed_dir = joinpath(RUNS_DIR, ds_name, outvar, "seed_$seed")
            if isfile(joinpath(seed_dir, "chi_best.npy"))
                @printf("    seed=%d  [done, skipping]\n", seed); continue
            end
            mkpath(seed_dir)

            Random.seed!(seed * 12345 + 7)
            model = benchmark_net(F, K)

            t0 = time()
            if variant == "svd_power"
                # full data; anchor i repeated Nk times = repeat_interleave to match burst pairs
                Nk = size(bursts, 2)
                x0 = Matrix{Float32}(undef, F, N * Nk)
                @inbounds for i in 1:N, j in 1:Nk
                    x0[:, (i-1)*Nk + j] = @view anchors[:, i]
                end
                x1 = reshape(bursts, F, N * Nk)                       # column (i-1)*Nk+j = burst j of anchor i
                losses, sds = power_method_multi(model, x0, x1)
                chi_best = model(anchors); chi_atstop = chi_best
                val_history = losses; sd_history = sds
            else
                tr = splits[seed+1, :] .== 0
                te = splits[seed+1, :] .== 1
                xs_tr = anchors[:, tr]; ys_tr = bursts[:, :, tr]
                xs_te = anchors[:, te]; ys_te = bursts[:, :, te]
                val_history, sd_history, chi_best, chi_atstop =
                    run_isotarget(variant, xs_tr, ys_tr, xs_te, ys_te, anchors, model; warmup=warmup)
            end
            elapsed = time() - t0

            npywrite(joinpath(seed_dir, "chi_best.npy"),   Array{Float32}(permutedims(chi_best)))
            npywrite(joinpath(seed_dir, "chi_atstop.npy"), Array{Float32}(permutedims(chi_atstop)))
            npywrite(joinpath(seed_dir, "val_loss.npy"),   Float32.(val_history))
            sdmat = isempty(sd_history) ? zeros(Float32, 0, K) :
                    Array{Float32}(permutedims(reduce(hcat, sd_history)))
            npywrite(joinpath(seed_dir, "chi_sd_history.npy"), sdmat)

            sd_final = vec(std(chi_atstop; dims=2))
            k_eff = sum(sd_final .> 0.05f0)
            @printf("    seed=%d  iters=%4d  sd=[%s]  k_eff=%d  t=%.0fs\n",
                seed, length(val_history),
                join((@sprintf("%.3f", s) for s in sd_final), ", "), k_eff, elapsed)
            flush(stdout)
        end
    end
end

# ── Load data (feature-first, Flux layout) from raw binaries ──────────────────
function loaddata()
    (; anchors_tw   = readbin("tw_anchors", Float32, (2, 1600)),        # (F, N)
       bursts_tw    = readbin("tw_bursts",  Float32, (2, 20, 1600)),    # (F, K, N)
       splits_tw    = readbin("tw_splits",  Int32,   (5, 1600)),
       anchors_al   = readbin("al_anchors",      Float32, (231, 1578)),
       bursts_al5   = readbin("al5_bursts",      Float32, (231, 20, 1578)),
       bursts_al01  = readbin("al01_bursts",     Float32, (231, 20, 1578)),
       bursts_joint = readbin("al_joint_bursts", Float32, (231, 40, 1578)),
       splits_al    = readbin("al_splits",       Int32,   (5, 1578)))
end

function main()
    d = loaddata()
    run_dataset("triple_well",     d.anchors_tw, d.bursts_tw,   d.splits_tw)
    run_dataset("alanine_5ps",     d.anchors_al, d.bursts_al5,  d.splits_al)
    run_dataset("alanine_0p1ps",   d.anchors_al, d.bursts_al01, d.splits_al)
    run_dataset("alanine_multitau",d.anchors_al, d.bursts_joint,d.splits_al)
    println("\nALL DONE")
end

# Only auto-run when executed directly (so other scripts can `include` this).
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
