# Ablation: ISA with NO GramSchmidt warm-up (warmup=0), same seeds/init as the
# warm-up runs. Tests whether the 100-iter GramSchmidt warm-up is necessary.
# Reuses every definition from benchmark_v3_julia.jl (which no longer auto-runs);
# writes to runs_julia/{ds}/isa_nowarmup/seed_*.
include("benchmark_v3_julia.jl")

d = loaddata()
run_dataset("triple_well",      d.anchors_tw, d.bursts_tw,    d.splits_tw;
            variants=["isa"], warmup=0, suffix="_nowarmup")
run_dataset("alanine_5ps",      d.anchors_al, d.bursts_al5,   d.splits_al;
            variants=["isa"], warmup=0, suffix="_nowarmup")
run_dataset("alanine_0p1ps",    d.anchors_al, d.bursts_al01,  d.splits_al;
            variants=["isa"], warmup=0, suffix="_nowarmup")
run_dataset("alanine_multitau", d.anchors_al, d.bursts_joint, d.splits_al;
            variants=["isa"], warmup=0, suffix="_nowarmup")
println("\nNOWARMUP DONE")
