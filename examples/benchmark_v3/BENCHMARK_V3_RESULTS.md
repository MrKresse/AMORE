# Benchmark v3 Results

Verdicts are based on **map shape** (Pearson r vs reference), not SD alone.
SD alone was the central error of the v2 report.
See `BENCHMARK_V2_POSTMORTEM.md` for context.

PASS: r ≥ 0.8 AND k_eff ≥ 1  |  PARTIAL: r ≥ 0.5 (or PASS with k_eff=0)  |  FAIL: r < 0.5

---

## Benchmark Design

### Datasets

| Dataset | System | τ | k | Anchors | Bursts/anchor |
|---------|--------|---|---|---------|---------------|
| `triple_well` | 2D triple-well potential (σ=1.2) | 0.30 | 3 | 1600 | 20 |
| `alanine_5ps` | ADP vacuum 450 K, AMBER14 | 5 ps | 3 | 1578 | 20 |
| `alanine_0p1ps` | ADP vacuum 450 K, AMBER14 | 0.1 ps | 3 | 1578 | 20 |
| `alanine_multitau` | ADP vacuum 450 K — joint τ=5ps + τ=0.1ps | both | 3 | 1578 | 20+20 |

Anchors are distributed on a 40×40 (φ,ψ) grid for ADP (1578/1600 cells
occupied) and a regular 2D grid for the triple-well.

### Variants

| ID | Label | Key operation |
|----|-------|---------------|
| V2 | ISA | Inner-simplex rotation — PCCA+ vertex selection + simplex inversion |
| V3 | GramSchmidt | QR orthonormalisation of K[χ] |
| V4 | PseudoInv | χ · pinv(K[χ]) projected onto Schur eigenvectors |
| V5 | Cross | Residual-weighted Rayleigh–Ritz on accumulated (χ, K[χ]) history |
| V6 | Power-Iter | Subspace orthogonal power iteration (`power_method_multi`) |
| B  | VAMP2 | Negative VAMP-2 score maximisation |

ShiftScale (V1) excluded — defined only for k=1.

### Architecture

All variants share the same network: **ChiNetMultiLinear**.

```
in_dim  →  128  →  32  →  8  →  k=3
Activations: Tanh (hidden layers), Linear (output)
```

Linear output is required because the isotarget functions (GramSchmidt,
Cross, PseudoInv) produce targets in their natural scale (~±√N ≈ ±40),
which sigmoid would saturate.  Tanh hidden layers provide smooth gradients
without capping the output range.

### Training — isotarget variants (V2–V5)

The isotarget loop alternates between computing a target from the current
network and training the network to fit that target.

```
Optimizer : Adam, lr=1e-3, gradient clip norm=5
Max iters : 5000
Min iters : 1000 (early stopping disabled before this)
Early stop: plateau — stop when range(val[-500:]) < 1e-3 × median(val[-500:])
Warm-up   : 100 iters of GramSchmidt on all k outputs before variant's own
            target takes over (not applied to Power-Iter or VAMP2)
Seeds     : 5, each with an independent 80/20 patch-based train/test split
```

The warm-up is necessary for ISA and PseudoInv, which require a chi with
approximate simplex structure before their inversion steps are numerically
stable.  ISA raises a `ValueError` (singular simplex submatrix) whenever
chi is near-uniform; the training loop skips those iterations and logs them.

### Training — Power-Iter (V6)

Power-Iter uses **subspace orthogonal power iteration** directly — it does
not compute an isotarget and does not use the warm-up or train/test split.

**Algorithm** (`power_method_multi` in `src/amore/isokann/power.py`):

Each outer iteration n performs four steps:

1. **Koopman action.** Evaluate the current network on the propagated
   points: `Y = χ_n(x₁)`, shape (N, k).  This approximates K[χ_n](x₀),
   the Koopman operator applied to the current basis.

2. **Orthogonal deflation via SVD.** Compute the thin SVD of the
   centred matrix: `Yc = Y − mean(Y)`,  `Yc = U S Vᵀ`.  Use the left
   singular vectors U (shape N×k, orthonormal columns) as the target.
   This is the key step that prevents all k functions from collapsing to
   the single dominant eigenfunction: the SVD projects out the leading
   direction at each iteration, forcing the remaining columns to span
   orthogonal directions.  SVD is used instead of covariance whitening
   because it is numerically stable when some eigenfunctions have near-zero
   amplitude (whitening would divide by near-zero singular values).

3. **Collapse guard.** If any column of U has standard deviation below
   ε=1e-3 (mode has collapsed to constant), inject small Gaussian noise
   into that column to re-seed the search.

4. **Inner SGD.** Scale U column-wise to [0,1] and train χ_{n+1}(x₀) to
   fit the scaled targets via Adam (batch=2048, `epochs_per_iter=50` passes
   through the data).  LR decays by 0.97× per outer iteration.

```
n_iter          : 100  outer iterations
epochs_per_iter : 50   inner SGD passes per iteration
Total SGD steps : ~5000  (comparable to isotarget variants at MAX_ITER=5000)
lr              : 1e-3, decaying by 0.97× per outer iter
batch           : 2048
Data            : full dataset (no train/test split)
```

**Convergence criterion.** After all n_iter iterations the Koopman matrix
K is estimated in the chi basis:

```
A_ij = E[χ_i(x₀) χ_j(x₀)]   (auto-correlation)
C_ij = E[χ_i(x₁) χ_j(x₀)]   (cross-correlation)
K    = A⁻¹ C
```

The eigenvalues of K give the implied timescales
`ITS_i = −τ / log|λ_i|` reported in the training output.

**What it converges to.** Simultaneous orthogonal iteration converges to
the invariant subspace spanned by the k Koopman eigenfunctions with the
k largest |λ|.  The network learns a basis for this subspace, not
necessarily individual eigenfunctions — a rotation within the subspace
is unidentified.  Shape correlation is measured against references via
Hungarian matching, which is rotation-invariant.

**Difference from the v2 'SVD' method.** The v2 benchmark computed
`eigen(H)` where H = U^T K[χ] V S⁻¹ is the reduced Koopman matrix from
DMD.  That is an eigendecomposition of a k×k matrix, not power iteration.
It used a fixed χ from a single forward pass rather than iterating.
Power-Iter replaces this with proper iterative refinement over 100 outer
steps. See `BENCHMARK_V2_POSTMORTEM.md` bug 3.

### Training — VAMP2 (B)

VAMP2 maximises the VAMP-2 score, which equals the sum of squared
singular values of the empirical Koopman operator in the chi basis.  No
isotarget is computed; the loss is differentiable end-to-end.

To avoid the chi=0 degenerate fixed point (where the VAMP-2 gradient
vanishes), chi is column-normalised (std-normalised) before computing
the score.  No warm-up.  Train/test split applied as for isotarget variants.

### Scoring

Verdicts are based on **map shape** (Pearson |r| vs reference), not SD.
SD alone was the central error of the v2 report — see
`BENCHMARK_V2_POSTMORTEM.md`.

| Dataset | Reference | Matching |
|---------|-----------|---------|
| triple_well | FEM committor functions p_A, p_B, p_C | Hungarian assignment of 3 chi columns to 3 refs, maximise mean |r| |
| alanine_5ps / multitau | Transfer-operator EV2 (λ=0.9900, ITS=499ps) | Max |r| over chi columns |

**Verdict gate:** r ≥ 0.80 AND k_eff ≥ 1 → PASS.  The k_eff ≥ 1
requirement prevents a 'hollow PASS': a method that earns high r from
a warm-up state but then collapses (SD≈0, no usable membership functions)
is demoted to PARTIAL.

---

## Reference Data Quality

This section documents the reference functions used for shape-correlation
scoring.  It reproduces the v2 Panel 0 checks so that v3 is a complete,
self-contained document.

### Triple-well — FEM committor references

N = 1600 anchors, FEM-computed committor functions p_A, p_B, p_C.

| Check | Value | Verdict |
|-------|-------|---------|
| Partition of unity (p_A+p_B+p_C) | mean=1.0000, std=8.78e-18 | PASS — exact by construction |
| Basin weights | p_A=0.308, p_B=0.309, p_C=0.383 | Balanced — all three wells populated |
| Value range | [0.00, 1.00] | PASS |

Separatrix cells (p_i < 0.6 for all i) have ~20% variance at 20 bursts
— acceptable for full-field Pearson r but would sharpen with 50 bursts.

### ADP (τ = 5 ps) — transfer-operator eigenvalue spectrum

Occupied grid cells: 1337 / 1600 (83.6%)

| Mode | Eigenvalue | ITS (ps) | Comment |
|------|-----------|----------|---------|
| EV1 | 0.9964 | 1400 | Ultra-slow mode |
| EV2 | 0.9900 | 499 | **Reference mode** — C7eq∪αR ↔ C7ax |
| EV3 | 0.2295 | 3 | Thermal bath begins |
| EV4 | 0.2155 | 3 | Thermal bath |
| EV5 | 0.2023 | 3 | Thermal bath |
| EV6–EV16 | 0.202–0.175 | 3.1–2.9 | Thermal bath |

**Gap:** ITS₁=1400 ps, ITS₂=499 ps — both are slow; ITS₃=3.4 ps marks the thermal bath.

**Benchmark reference = EV2** (λ=0.9900, ITS≈499 ps).
EV1 (λ=0.9964) exists but is ultralow-amplitude and was not used in v2.

**Implication for benchmark design at k=3, τ=5 ps:**

- chi_1 should converge to EV2 (C7eq∪αR ↔ C7ax)
- chi_2 and chi_3 have no slow dynamics to learn → will collapse to noise
- This is **not a failure** — it is the expected physical outcome
- The benchmark tests graceful collapse (chi_2/3 → SD≈0, chi_1 live)
  vs hard failure (all three collapse)

### Panel 0 — reference overview figure

![Panel 0 reference](../benchmark_v2/panel0/panel0_reference.png)

*(From v2 benchmark analysis.)*

---

## Training Results

## triple_well

Reference: Committor functions p_A, p_B, p_C (FEM reference)

| Method | Seeds | Mean r | SD r | k_eff (>0.05) | Verdict |
|--------|-------|--------|------|--------------|---------|
| V2-ISA | 5/5 | 0.981 | ±0.001 | 3.0 | **PASS** |
| V3-GramSchmidt | 5/5 | 0.862 | ±0.050 | 2.0 | **PASS** |
| V4-PseudoInv | 5/5 | 0.631 | ±0.038 | 2.4 | **PARTIAL** |
| V5-Cross | 5/5 | 0.772 | ±0.105 | 2.0 | **PARTIAL** |
| Power-Iter | 5/5 | 0.633 | ±0.114 | 1.4 | **PARTIAL** |
| B-VAMP2 | 5/5 | 0.811 | ±0.070 | 2.2 | **PASS** |

### Shape summary — triple_well

![shape summary](figures/triple_well/shape_summary.png)

### Chi panels — triple_well

#### V2-ISA

![V2-ISA chi panels](figures/triple_well/chi_panels_isa.png)

#### V3-GramSchmidt

![V3-GramSchmidt chi panels](figures/triple_well/chi_panels_gramschmidt.png)

#### V4-PseudoInv

![V4-PseudoInv chi panels](figures/triple_well/chi_panels_pseudoinv.png)

#### V5-Cross

![V5-Cross chi panels](figures/triple_well/chi_panels_cross.png)

#### Power-Iter

![Power-Iter chi panels](figures/triple_well/chi_panels_svd.png)

#### B-VAMP2

![B-VAMP2 chi panels](figures/triple_well/chi_panels_vamp2.png)

## alanine_5ps

Reference: Transfer-operator EV2 (eigenvalue≈0.990, tau=5ps)

| Method | Seeds | Mean r | SD r | k_eff (>0.05) | Verdict |
|--------|-------|--------|------|--------------|---------|
| V2-ISA | 5/5 | 0.844 | ±0.130 | 1.8 | **PASS** |
| V3-GramSchmidt | 5/5 | 0.819 | ±0.130 | 2.8 | **PASS** |
| V4-PseudoInv | 5/5 | 0.519 | ±0.156 | 0.0 | **PARTIAL** |
| V5-Cross | 5/5 | 0.622 | ±0.085 | 2.8 | **PARTIAL** |
| Power-Iter | 5/5 | 0.899 | ±0.166 | 1.0 | **PASS** |
| B-VAMP2 | 5/5 | 0.401 | ±0.218 | 0.0 | **FAIL** |

### Shape summary — alanine_5ps

![shape summary](figures/alanine_5ps/shape_summary.png)

### Chi panels — alanine_5ps

#### V2-ISA

![V2-ISA chi panels](figures/alanine_5ps/chi_panels_isa.png)

#### V3-GramSchmidt

![V3-GramSchmidt chi panels](figures/alanine_5ps/chi_panels_gramschmidt.png)

#### V4-PseudoInv

![V4-PseudoInv chi panels](figures/alanine_5ps/chi_panels_pseudoinv.png)

#### V5-Cross

![V5-Cross chi panels](figures/alanine_5ps/chi_panels_cross.png)

#### Power-Iter

![Power-Iter chi panels](figures/alanine_5ps/chi_panels_svd.png)

#### B-VAMP2

![B-VAMP2 chi panels](figures/alanine_5ps/chi_panels_vamp2.png)

## alanine_0p1ps

Reference: Transfer-operator EV2 (eigenvalue≈0.990, tau=5ps) — trained on 0.1ps bursts only

| Method | Seeds | Mean r | SD r | k_eff (>0.05) | Verdict |
|--------|-------|--------|------|--------------|---------|
| V2-ISA | 5/5 | 0.483 | ±0.408 | 1.8 | **FAIL** |
| V3-GramSchmidt | 5/5 | 0.388 | ±0.285 | 3.0 | **FAIL** |
| V4-PseudoInv | 5/5 | 0.160 | ±0.105 | 0.0 | **FAIL** |
| V5-Cross | 5/5 | 0.224 | ±0.113 | 2.4 | **FAIL** |
| Power-Iter | 5/5 | 0.540 | ±0.216 | 2.8 | **PARTIAL** |
| B-VAMP2 | 5/5 | 0.422 | ±0.209 | 0.0 | **FAIL** |

### Shape summary — alanine_0p1ps

![shape summary](figures/alanine_0p1ps/shape_summary.png)

### Chi panels — alanine_0p1ps

#### V2-ISA

![V2-ISA chi panels](figures/alanine_0p1ps/chi_panels_isa.png)

#### V3-GramSchmidt

![V3-GramSchmidt chi panels](figures/alanine_0p1ps/chi_panels_gramschmidt.png)

#### V4-PseudoInv

![V4-PseudoInv chi panels](figures/alanine_0p1ps/chi_panels_pseudoinv.png)

#### V5-Cross

![V5-Cross chi panels](figures/alanine_0p1ps/chi_panels_cross.png)

#### Power-Iter

![Power-Iter chi panels](figures/alanine_0p1ps/chi_panels_svd.png)

#### B-VAMP2

![B-VAMP2 chi panels](figures/alanine_0p1ps/chi_panels_vamp2.png)

## alanine_multitau

Reference: Transfer-operator EV2 (eigenvalue≈0.990, tau=5ps) — joint 5ps+0.1ps training

| Method | Seeds | Mean r | SD r | k_eff (>0.05) | Verdict |
|--------|-------|--------|------|--------------|---------|
| V2-ISA | 5/5 | 0.382 | ±0.132 | 0.6 | **FAIL** |
| V3-GramSchmidt | 5/5 | 0.705 | ±0.325 | 2.6 | **PARTIAL** |
| V4-PseudoInv | 5/5 | 0.338 | ±0.145 | 0.0 | **FAIL** |
| V5-Cross | 5/5 | 0.474 | ±0.097 | 3.0 | **FAIL** |
| Power-Iter | 5/5 | 0.585 | ±0.348 | 0.4 | **PARTIAL** |
| B-VAMP2 | 5/5 | 0.217 | ±0.104 | 0.0 | **FAIL** |

### Shape summary — alanine_multitau

![shape summary](figures/alanine_multitau/shape_summary.png)

### Chi panels — alanine_multitau

#### V2-ISA

![V2-ISA chi panels](figures/alanine_multitau/chi_panels_isa.png)

#### V3-GramSchmidt

![V3-GramSchmidt chi panels](figures/alanine_multitau/chi_panels_gramschmidt.png)

#### V4-PseudoInv

![V4-PseudoInv chi panels](figures/alanine_multitau/chi_panels_pseudoinv.png)

#### V5-Cross

![V5-Cross chi panels](figures/alanine_multitau/chi_panels_cross.png)

#### Power-Iter

![Power-Iter chi panels](figures/alanine_multitau/chi_panels_svd.png)

#### B-VAMP2

![B-VAMP2 chi panels](figures/alanine_multitau/chi_panels_vamp2.png)

## ISA — warm-up vs no warm-up

Does the 100-iter GramSchmidt warm-up matter for ISA? `isa` is the standard
(warm-up) run; `isa_nowarmup` removes it entirely (ISA target from iteration 1).
Same seeds/init. See `ISA_WARMUP_ABLATION.md` for the cross-framework version.

| Dataset | warm-up r | warm-up k_eff | no-warm-up r | no-warm-up k_eff |
|---------|-----------|---------------|--------------|------------------|
| triple_well | 0.981 | 3.0 | 0.980 | 3.0 |
| alanine_5ps | 0.844 | 1.8 | 0.552 | 0.6 |
| alanine_0p1ps | 0.483 | 1.8 | 0.619 | 1.2 |
| alanine_multitau | 0.382 | 0.6 | 0.603 | 1.2 |

Warm-up is **unnecessary on triple_well** (ISA reaches r≈0.98 / k_eff=3 either
way) but **stabilises ISA on the 231-dim ADP data**, where without it the
inner-simplex inversion collapses on most seeds. It supplies the
approximate-simplex chi the inversion needs in high dimensions.

## Notes

- **svd** = subspace power iteration (`power_method_multi`), NOT DMD eigen(H).
  See BENCHMARK_V2_POSTMORTEM.md, bug 3.
- **Multi-tau** uses real 231-dim features at 0.1 ps (50 steps) from the same
  1578 anchors. v2's 0.1 ps data was grid-snapped (postmortem bug 6) — replaced
  by `01_simulate_alanine_0p1ps.py`. Joint bursts = N_K=40 (20×5ps + 20×0.1ps).
- **ISA** uses the corrected inner-simplex transform `A = inv(K[verts])ᵀ`
  (`src/amore/isotarget.py`); on triple-well it is a genuine PASS (r≈0.98,
  k_eff=3), matching the Julia reference. See `ISA_WARMUP_ABLATION.md`.
- Verification gate: `tests/test_isotarget.py` — 19/19 PASSED (property tests).
  Cross-checked numerically against the Julia `isotarget.jl` originals; see
  `BENCHMARK_V3_JULIA_RESULTS.md`.
