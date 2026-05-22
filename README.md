# AMORE

**A**nalysis and **M**odelling via **O**ptimised **R**epresentations and **E**igenfunctions.

Python implementation of multi-dimensional ISOKANN (Iterative Squaring Of Koopman operator ANalysis) for learning Koopman eigenfunctions from molecular simulation data.

## What is ISOKANN?

ISOKANN learns slow collective variables (chi functions) directly from pairs of trajectory frames `(x₀, x_τ)` by minimising the distance between `chi(x₀)` and the Koopman expectation `E[chi(x_τ) | x₀]`. The learned chi functions approximate the dominant eigenfunctions of the transfer operator, revealing metastable states and slow dynamics.

## Package structure

```
src/amore/
  sims/           OpenMM-based simulation (Langevin, metadynamics, burst pairs)
  isokann/        SVD-Power iteration (power_method_multi) and chi networks
  features/       Pairwise distance featuriser
  mep/            Minimum energy path (AMORE-MD loop)
  chi.py          Chi sensitivity analysis
  io.py           Structure I/O helpers
```

## Examples

```
examples/
  MD/             Alanine dipeptide and Mueller-Brown potential workflows
  GRN/            LARRY hematopoiesis scRNA-seq benchmark
  benchmark/      ISOKANN isotarget variant comparison v1 (5 variants + SVD-Power)
  benchmark_v2/   Benchmark v2: 6 variants, linear output, multi-tau ADP test
```

## Installation

```bash
conda env create -f environment.yml
conda activate amore
pip install -e .
```

For MD simulations (OpenMM):
```bash
conda install -c conda-forge openmm
```

For scRNA-seq examples (LARRY):
```bash
pip install scanpy anndata deeptime
```

## Running the benchmark

```bash
# Step 1: generate simulation data
python examples/benchmark/00_simulate_triple_well.py
python examples/benchmark/01_simulate_alanine.py

# Step 2: run training (6 variants × 3 datasets × 5 seeds)
python examples/benchmark_v2/02_train_benchmark_v2.py

# Step 3: generate figures
python examples/benchmark_v2/03_plot_benchmark_v2.py
```

Results are documented in `examples/benchmark_v2/BENCHMARK_V2_RESULTS.md`.
