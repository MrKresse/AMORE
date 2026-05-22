# Benchmark v2 — Run instructions

## What this benchmark tests

Three datasets × 6 variants × 5 seeds:

| Dataset | Description | Expected finding |
|---|---|---|
| `triple_well` | 2D triple-well, τ=0.30, k=3 | All methods should recover 3 basins |
| `alanine_5ps` | ADP 450K, τ=5ps, k=3 | Only ONE non-trivial slow mode exists (ITS≈500ps). chi_2/chi_3 will collapse to noise for all methods. Test: graceful vs catastrophic collapse |
| `alanine_multi_tau` | ADP 450K, τ=5ps+0.1ps, k=3 | The 0.1ps lag carries the C7eq↔αR mode. Test: can any method separate all three macrostates? |

Variants: ISA, GramSchmidt, PseudoInv, SVD (new in v2), Cross, VAMP2.

Architecture: `231 → [128,32,8] → 3` (sigmoid hidden, **linear output**).

## Data files needed

All in `AMORE/examples/benchmark/data/`:

| File | Description |
|---|---|
| `triple_well_koopman.npz` | TW anchors + 20-burst data |
| `alanine_koopman.npz` | ADP anchors + 20-burst data at τ=5ps |
| `alanine_multilag.npz` | ADP 1200 anchors, 1-burst data at τ=0.1,0.2,0.5,1,2,5ps |

Panel 0 reference data in `benchmark_v2/panel0/` (already generated).

## Steps to run on GPU cluster

### 1. Copy files

Copy the entire `ZIBwork/AMORE/` directory to the cluster. Key paths needed:
```
AMORE/src/amore/
AMORE/examples/benchmark/data/
AMORE/examples/benchmark/targets.py
AMORE/examples/benchmark_v2/
```

### 2. Environment

The `amore` conda environment should already be set up (or recreate from `environment.yml` if available). Required packages: `torch`, `numpy`, `scipy`, `sklearn`, `matplotlib`.

For CUDA support, ensure PyTorch was installed with the correct CUDA version for the cluster's GPU. Check with:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 3. Run training

```bash
cd AMORE/examples/benchmark_v2/
python 02_train_benchmark_v2.py
```

The script auto-detects CUDA (`torch.cuda.is_available()`). On a modern GPU, expected runtime:
- Triple-well (5000 iters × 6 variants × 5 seeds): ~30 min
- ADP 5ps (5000 iters × 6 variants × 5 seeds): ~2–3 hrs
- ADP multi-tau (5000 iters × 6 variants × 5 seeds): ~2–3 hrs

The script resumes automatically if interrupted — it checks for existing `chi_atstop.npy` before each run.

Progress is printed per seed:
```
  V3-GramSchmidt
    seed=0  iters= 847  sd=[0.412, 0.051, 0.008]  k_eff=2  t=43s
```
`k_eff` = number of chi modes with SD > 0.05 (live modes). The key finding to watch:

- **Triple-well**: expect k_eff=3 for good methods, k_eff<3 for failing ones
- **ADP 5ps**: expect k_eff=1 for ALL methods (only 1 slow mode exists) — this is correct, not a failure
- **ADP multi-tau**: watch for k_eff=2 or k_eff=3 — this would mean a method separated basins using both timescales

### 4. Run plotting

```bash
python 03_plot_benchmark_v2.py
```

Outputs to `benchmark_v2/figures/`:
- `panelA_{dataset}.pdf` — convergence curves
- `panelB_{dataset}.pdf` — val loss / correlation / SD boxplots  
- `panelC_{dataset}.pdf` — per-mode SD bar chart
- `panelD_TW_maps.pdf` — chi maps on (x,y)
- `panelD_{alanine_*}.pdf` — chi on (φ,ψ) + chi-space scatter

### 5. Key results to report

After training, check:
1. **Which ADP-5ps methods collapse gracefully?** SD→0 on chi_2/chi_3 with chi_1 still live is a PASS. All three collapsing is a FAIL.
2. **Does any ADP-multi-tau method achieve k_eff≥2?** If yes, it has separated C7eq from αR using the 0.1ps signal. Panel D will show 3 clusters on (φ,ψ).
3. **Triple-well ranking**: correlation with empirical committors (Panel B) gives the definitive method ranking.

## Files summary

| File | Purpose |
|---|---|
| `panel0.py` | Reference quality check (already run) |
| `01b_adp_its_data.py` | Multi-lag data generation (already run) |
| `02_train_benchmark_v2.py` | Main training script |
| `03_plot_benchmark_v2.py` | Plotting script |
| `panel0/` | Reference data and figures |
| `runs/` | Training outputs (created by training script) |
| `figures/` | Plot outputs (created by plotting script) |
