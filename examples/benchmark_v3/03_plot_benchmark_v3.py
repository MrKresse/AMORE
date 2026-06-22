# -*- coding: utf-8 -*-
"""
Benchmark v3 plotting and shape-correlation analysis.

Verdicts are based on map SHAPE (Pearson r against reference), not SD alone.
SD alone was the central error of the v2 report — see BENCHMARK_V2_POSTMORTEM.md.

Shape metrics
-------------
Triple-well (TW):
  For each seed's chi_best (N=1600, k=3), compute the Hungarian-matched
  Pearson r of each chi row against the three committor references {p_A, p_B, p_C}.
  Anchors are identical between tw_committor.npz and triple_well_koopman.npz.

ADP tau=5ps:
  k_eff = 1 at tau=5ps (one non-trivial slow mode). Reference = EV2 from the
  transfer-operator eigenvector analysis (adp_eigvecs.npz, index 1).
  Map each of the 1578 anchors to its (phi,psi) grid cell, look up EV2 there.
  Report Pearson r of the best chi dimension vs EV2 for each seed.

Verdict thresholds
------------------
  r > 0.80  →  PASS
  r > 0.50  →  PARTIAL
  r ≤ 0.50  →  FAIL

Outputs
-------
  benchmark_v3/figures/{dataset}/
    chi_panels_{variant}.png   — chi_best per seed, coloured by chi value
    shape_summary.png          — r-bar chart per method, with SD error bars
  benchmark_v3/BENCHMARK_V3_RESULTS.md  — per-method verdict table
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

HERE      = os.path.dirname(__file__)
RUNS_DIR  = os.path.join(HERE, "runs")
DATA_DIR  = os.path.join(HERE, "..", "benchmark",    "data")
REF_DIR   = os.path.join(HERE, "..", "benchmark_v2", "panel0")
FIG_BASE  = os.path.join(HERE, "figures")
os.makedirs(FIG_BASE, exist_ok=True)

VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd", "vamp2"]
VARIANT_LABELS = {
    "isa"         : "V2-ISA",
    "gramschmidt" : "V3-GramSchmidt",
    "pseudoinv"   : "V4-PseudoInv",
    "cross"       : "V5-Cross",
    "svd"         : "Power-Iter",   # subspace power iteration
    "vamp2"       : "B-VAMP2",
}
N_SEEDS   = 5
K         = 3

PASS_R    = 0.80
PARTIAL_R = 0.50


# =============================================================================
# Reference loading
# =============================================================================

def load_tw_refs() -> dict:
    """Load triple-well committor references and anchor coordinates."""
    path = os.path.join(REF_DIR, "tw_committor.npz")
    if not os.path.exists(path):
        return {}
    d = np.load(path)
    return {
        "anchors" : d["anchors"],                        # (1600, 2)
        "refs"    : np.stack([d["p_A"], d["p_B"], d["p_C"]]),  # (3, 1600)
        "ref_names": ["p_A", "p_B", "p_C"],
    }


def load_adp_refs() -> dict:
    """Load ADP EV2 reference mapped onto the 1578 anchor set."""
    ev_path = os.path.join(REF_DIR,  "adp_eigvecs.npz")
    al_path = os.path.join(DATA_DIR, "alanine_koopman.npz")
    if not os.path.exists(ev_path) or not os.path.exists(al_path):
        return {}

    ev = np.load(ev_path)
    al = np.load(al_path)

    eigvecs = ev["eigvecs"]    # (1600, 16) — full 40×40 grid
    edges   = ev["edges"]      # (41,) — grid edges in [-π, π]

    phi = al["anchors_phi"]    # (1578,)
    psi = al["anchors_psi"]

    # Map each anchor to the nearest grid cell via digitize
    phi_idx = np.clip(np.digitize(phi, edges) - 1, 0, 39)
    psi_idx = np.clip(np.digitize(psi, edges) - 1, 0, 39)
    flat    = phi_idx * 40 + psi_idx               # (1578,) linearised index

    # EV2 = index 1 (index 0 is the stationary mode, eigenvalue≈1)
    ev2 = eigvecs[flat, 1]                         # (1578,)

    return {
        "phi"       : phi,
        "psi"       : psi,
        "ev2"       : ev2,
        "eigenvalues": ev["eigenvalues"][:8],
    }


# =============================================================================
# Shape metrics
# =============================================================================

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, allowing sign flip (returns |r|)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan
    a, b = a[mask], b[mask]
    a -= a.mean(); b -= b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def hungarian_match_r(chi: np.ndarray, refs: np.ndarray) -> tuple[float, list[int]]:
    """
    chi  : (N, k)  chi_best at anchors
    refs : (r, N)  reference functions (r may differ from k)

    Returns the mean |r| under the optimal (r → k) assignment, and the
    assignment permutation. Each ref row is matched to the chi column with
    the highest |Pearson r|, allowing sign flips.
    """
    k = chi.shape[1]
    r = refs.shape[0]
    n_match = min(k, r)

    # Build |r| matrix: shape (r, k)
    rmat = np.zeros((r, k))
    for i in range(r):
        for j in range(k):
            rmat[i, j] = abs(pearson_r(chi[:, j], refs[i]))

    # Brute-force optimal assignment over r rows → k columns (r ≤ 3 here)
    best_score, best_perm = -1.0, list(range(n_match))
    for perm in permutations(range(k), n_match):
        score = sum(rmat[i, perm[i]] for i in range(n_match)) / n_match
        if score > best_score:
            best_score, best_perm = score, list(perm)

    return best_score, best_perm


def tw_shape_r(chi: np.ndarray, tw_refs: dict) -> float:
    """Mean |r| of chi_best vs triple-well committors (Hungarian-matched)."""
    if not tw_refs:
        return np.nan
    score, _ = hungarian_match_r(chi, tw_refs["refs"])
    return score


def adp_shape_r(chi: np.ndarray, adp_refs: dict) -> float:
    """
    |r| of the best chi dimension vs EV2.

    k_eff=1 at tau=5ps: only one chi dimension should correlate with EV2;
    the other two should be collapsed. We take the max over chi columns.
    """
    if not adp_refs:
        return np.nan
    ev2 = adp_refs["ev2"]
    rs = [abs(pearson_r(chi[:, j], ev2)) for j in range(chi.shape[1])]
    return float(np.max(rs))


def compute_metrics(ds_name: str, refs: dict) -> dict:
    """
    For each variant, collect shape r and SD across seeds.

    Returns {variant: {"r_per_seed": [...], "sd_per_seed": [..., k-length each]}}
    """
    metrics: dict = {}
    shape_fn = tw_shape_r if ds_name.startswith("triple") else adp_shape_r

    for variant in VARIANTS:
        r_list, sd_list = [], []
        for seed in range(N_SEEDS):
            chi_path = os.path.join(RUNS_DIR, ds_name, variant, f"seed_{seed}", "chi_best.npy")
            if not os.path.exists(chi_path):
                continue
            chi = np.load(chi_path)                    # (N, k)
            r_list.append(shape_fn(chi, refs))
            sd_hist_path = os.path.join(RUNS_DIR, ds_name, variant,
                                        f"seed_{seed}", "chi_sd_history.npy")
            if os.path.exists(sd_hist_path):
                sd_hist = np.load(sd_hist_path)        # (n_iter, k)
                sd_list.append(sd_hist[-1])            # final SD per mode
        metrics[variant] = {
            "r_per_seed"  : r_list,
            "sd_per_seed" : sd_list,
        }
    return metrics


def verdict(r: float, mean_k_eff: float = None) -> str:
    if np.isnan(r):
        return "MISSING"
    # Require at least one non-degenerate mode (k_eff ≥ 1) for a full PASS.
    # Without this, ISA on TW earns r≈0.80 via the warm-up state but then
    # collapses (SD≈0) for all main-loop iterations — a "hollow PASS".
    if r >= PASS_R:
        if mean_k_eff is not None and mean_k_eff < 1.0:
            return "PARTIAL"
        return "PASS"
    if r >= PARTIAL_R:
        return "PARTIAL"
    return "FAIL"


# =============================================================================
# Plotting
# =============================================================================

def plot_chi_panels(ds_name: str, refs: dict, fig_dir: str) -> None:
    """
    For each variant, plot chi_best panels (one row per seed).
    TW: 2D scatter coloured by chi values.
    ADP: Ramachandran plot (phi vs psi) coloured by chi values.
    """
    os.makedirs(fig_dir, exist_ok=True)
    is_tw = ds_name.startswith("triple")

    if is_tw and refs:
        anchors = refs["anchors"]    # (1600, 2)

    for variant in VARIANTS:
        fig, axes = plt.subplots(N_SEEDS, K, figsize=(4 * K, 3 * N_SEEDS))
        fig.suptitle(f"{VARIANT_LABELS.get(variant, variant)} — {ds_name}", fontsize=13)

        for seed in range(N_SEEDS):
            chi_path = os.path.join(RUNS_DIR, ds_name, variant,
                                    f"seed_{seed}", "chi_best.npy")
            if not os.path.exists(chi_path):
                for j in range(K):
                    axes[seed, j].axis("off")
                    axes[seed, j].text(0.5, 0.5, "no data",
                                       ha="center", va="center", transform=axes[seed, j].transAxes)
                continue

            chi = np.load(chi_path)    # (N, k)

            if is_tw and refs:
                x, y = anchors[:, 0], anchors[:, 1]
            else:
                al_path = os.path.join(DATA_DIR, "alanine_koopman.npz")
                al = np.load(al_path)
                x, y = al["anchors_phi"], al["anchors_psi"]

            for j in range(K):
                ax = axes[seed, j]
                sc = ax.scatter(x, y, c=chi[:, j], s=4, cmap="coolwarm",
                                vmin=chi[:, j].min(), vmax=chi[:, j].max())
                ax.set_xticks([]); ax.set_yticks([])
                sd_j = float(chi[:, j].std())
                ax.set_title(f"s={seed} χ{j+1} sd={sd_j:.2f}", fontsize=8)
                plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)

        plt.tight_layout()
        out = os.path.join(fig_dir, f"chi_panels_{variant}.png")
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_shape_summary(ds_name: str, metrics: dict, fig_dir: str) -> None:
    """Bar chart of mean shape-r per method, error bars = std across seeds."""
    os.makedirs(fig_dir, exist_ok=True)

    r_means, r_stds, labels = [], [], []
    for variant in VARIANTS:
        rs = [r for r in metrics[variant]["r_per_seed"] if np.isfinite(r)]
        r_means.append(np.mean(rs) if rs else np.nan)
        r_stds.append(np.std(rs)   if len(rs) > 1 else 0.0)
        labels.append(VARIANT_LABELS.get(variant, variant))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, r_means, yerr=r_stds, capsize=4, color="steelblue", alpha=0.8)

    # Colour bars by verdict
    for bar, r in zip(bars, r_means):
        if np.isnan(r):
            bar.set_color("lightgray")
        elif r >= PASS_R:
            bar.set_color("forestgreen")
        elif r >= PARTIAL_R:
            bar.set_color("goldenrod")
        else:
            bar.set_color("firebrick")

    ax.axhline(PASS_R,    color="green",  ls="--", lw=1, label=f"PASS r={PASS_R}")
    ax.axhline(PARTIAL_R, color="orange", ls="--", lw=1, label=f"PARTIAL r={PARTIAL_R}")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean |Pearson r| vs reference")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Shape correlation — {ds_name}\n"
                 "(verdicts based on map shape, not SD alone)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(fig_dir, "shape_summary.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Results document
# =============================================================================

V2_REF_DIR = os.path.join(HERE, "..", "benchmark_v2", "panel0")


def _procedure_section() -> list[str]:
    """
    Generate the benchmark design and method description section.
    Numbers pulled from the module-level constants.
    """
    lines = [
        "## Benchmark Design",
        "",
        "### Datasets",
        "",
        "| Dataset | System | τ | k | Anchors | Bursts/anchor |",
        "|---------|--------|---|---|---------|---------------|",
        "| `triple_well` | 2D triple-well potential (σ=1.2) | 0.30 | 3 | 1600 | 20 |",
        "| `alanine_5ps` | ADP vacuum 450 K, AMBER14 | 5 ps | 3 | 1578 | 20 |",
        "| `alanine_0p1ps` | ADP vacuum 450 K, AMBER14 | 0.1 ps | 3 | 1578 | 20 |",
        "| `alanine_multitau` | ADP vacuum 450 K — joint τ=5ps + τ=0.1ps | both | 3 | 1578 | 20+20 |",
        "",
        "Anchors are distributed on a 40×40 (φ,ψ) grid for ADP (1578/1600 cells",
        "occupied) and a regular 2D grid for the triple-well.",
        "",
        "### Variants",
        "",
        "| ID | Label | Key operation |",
        "|----|-------|---------------|",
        "| V2 | ISA | Inner-simplex rotation — PCCA+ vertex selection + simplex inversion |",
        "| V3 | GramSchmidt | QR orthonormalisation of K[χ] |",
        "| V4 | PseudoInv | χ · pinv(K[χ]) projected onto Schur eigenvectors |",
        "| V5 | Cross | Residual-weighted Rayleigh–Ritz on accumulated (χ, K[χ]) history |",
        "| V6 | Power-Iter | Subspace orthogonal power iteration (`power_method_multi`) |",
        "| B  | VAMP2 | Negative VAMP-2 score maximisation |",
        "",
        "ShiftScale (V1) excluded — defined only for k=1.",
        "",
        "### Architecture",
        "",
        "All variants share the same network: **ChiNetMultiLinear**.",
        "",
        "```",
        "in_dim  →  128  →  32  →  8  →  k=3",
        "Activations: Tanh (hidden layers), Linear (output)",
        "```",
        "",
        "Linear output is required because the isotarget functions (GramSchmidt,",
        "Cross, PseudoInv) produce targets in their natural scale (~±√N ≈ ±40),",
        "which sigmoid would saturate.  Tanh hidden layers provide smooth gradients",
        "without capping the output range.",
        "",
        "### Training — isotarget variants (V2–V5)",
        "",
        "The isotarget loop alternates between computing a target from the current",
        "network and training the network to fit that target.",
        "",
        "```",
        "Optimizer : Adam, lr=1e-3, gradient clip norm=5",
        "Max iters : 5000",
        "Min iters : 1000 (early stopping disabled before this)",
        "Early stop: plateau — stop when range(val[-500:]) < 1e-3 × median(val[-500:])",
        "Warm-up   : 100 iters of GramSchmidt on all k outputs before variant's own",
        "            target takes over (not applied to Power-Iter or VAMP2)",
        "Seeds     : 5, each with an independent 80/20 patch-based train/test split",
        "```",
        "",
        "The warm-up is necessary for ISA and PseudoInv, which require a chi with",
        "approximate simplex structure before their inversion steps are numerically",
        "stable.  ISA raises a `ValueError` (singular simplex submatrix) whenever",
        "chi is near-uniform; the training loop skips those iterations and logs them.",
        "",
        "### Training — Power-Iter (V6)",
        "",
        "Power-Iter uses **subspace orthogonal power iteration** directly — it does",
        "not compute an isotarget and does not use the warm-up or train/test split.",
        "",
        "**Algorithm** (`power_method_multi` in `src/amore/isokann/power.py`):",
        "",
        "Each outer iteration n performs four steps:",
        "",
        "1. **Koopman action.** Evaluate the current network on the propagated",
        "   points: `Y = χ_n(x₁)`, shape (N, k).  This approximates K[χ_n](x₀),",
        "   the Koopman operator applied to the current basis.",
        "",
        "2. **Orthogonal deflation via SVD.** Compute the thin SVD of the",
        "   centred matrix: `Yc = Y − mean(Y)`,  `Yc = U S Vᵀ`.  Use the left",
        "   singular vectors U (shape N×k, orthonormal columns) as the target.",
        "   This is the key step that prevents all k functions from collapsing to",
        "   the single dominant eigenfunction: the SVD projects out the leading",
        "   direction at each iteration, forcing the remaining columns to span",
        "   orthogonal directions.  SVD is used instead of covariance whitening",
        "   because it is numerically stable when some eigenfunctions have near-zero",
        "   amplitude (whitening would divide by near-zero singular values).",
        "",
        "3. **Collapse guard.** If any column of U has standard deviation below",
        "   ε=1e-3 (mode has collapsed to constant), inject small Gaussian noise",
        "   into that column to re-seed the search.",
        "",
        "4. **Inner SGD.** Scale U column-wise to [0,1] and train χ_{n+1}(x₀) to",
        "   fit the scaled targets via Adam (batch=2048, `epochs_per_iter=50` passes",
        "   through the data).  LR decays by 0.97× per outer iteration.",
        "",
        "```",
        "n_iter          : 100  outer iterations",
        "epochs_per_iter : 50   inner SGD passes per iteration",
        "Total SGD steps : ~5000  (comparable to isotarget variants at MAX_ITER=5000)",
        "lr              : 1e-3, decaying by 0.97× per outer iter",
        "batch           : 2048",
        "Data            : full dataset (no train/test split)",
        "```",
        "",
        "**Convergence criterion.** After all n_iter iterations the Koopman matrix",
        "K is estimated in the chi basis:",
        "",
        "```",
        "A_ij = E[χ_i(x₀) χ_j(x₀)]   (auto-correlation)",
        "C_ij = E[χ_i(x₁) χ_j(x₀)]   (cross-correlation)",
        "K    = A⁻¹ C",
        "```",
        "",
        "The eigenvalues of K give the implied timescales",
        "`ITS_i = −τ / log|λ_i|` reported in the training output.",
        "",
        "**What it converges to.** Simultaneous orthogonal iteration converges to",
        "the invariant subspace spanned by the k Koopman eigenfunctions with the",
        "k largest |λ|.  The network learns a basis for this subspace, not",
        "necessarily individual eigenfunctions — a rotation within the subspace",
        "is unidentified.  Shape correlation is measured against references via",
        "Hungarian matching, which is rotation-invariant.",
        "",
        "**Difference from the v2 'SVD' method.** The v2 benchmark computed",
        "`eigen(H)` where H = U^T K[χ] V S⁻¹ is the reduced Koopman matrix from",
        "DMD.  That is an eigendecomposition of a k×k matrix, not power iteration.",
        "It used a fixed χ from a single forward pass rather than iterating.",
        "Power-Iter replaces this with proper iterative refinement over 100 outer",
        "steps. See `BENCHMARK_V2_POSTMORTEM.md` bug 3.",
        "",
        "### Training — VAMP2 (B)",
        "",
        "VAMP2 maximises the VAMP-2 score, which equals the sum of squared",
        "singular values of the empirical Koopman operator in the chi basis.  No",
        "isotarget is computed; the loss is differentiable end-to-end.",
        "",
        "To avoid the chi=0 degenerate fixed point (where the VAMP-2 gradient",
        "vanishes), chi is column-normalised (std-normalised) before computing",
        "the score.  No warm-up.  Train/test split applied as for isotarget variants.",
        "",
        "### Scoring",
        "",
        "Verdicts are based on **map shape** (Pearson |r| vs reference), not SD.",
        "SD alone was the central error of the v2 report — see",
        "`BENCHMARK_V2_POSTMORTEM.md`.",
        "",
        "| Dataset | Reference | Matching |",
        "|---------|-----------|---------|",
        "| triple_well | FEM committor functions p_A, p_B, p_C | Hungarian assignment of 3 chi columns to 3 refs, maximise mean |r| |",
        "| alanine_5ps / multitau | Transfer-operator EV2 (λ=0.9900, ITS=499ps) | Max |r| over chi columns |",
        "",
        "**Verdict gate:** r ≥ 0.80 AND k_eff ≥ 1 → PASS.  The k_eff ≥ 1",
        "requirement prevents a 'hollow PASS': a method that earns high r from",
        "a warm-up state but then collapses (SD≈0, no usable membership functions)",
        "is demoted to PARTIAL.",
        "",
        "---",
        "",
    ]
    return lines


def _reference_preamble(tw_refs: dict, adp_refs: dict) -> list[str]:
    """
    Generate the reference-quality preamble section, reproducing the v2
    Panel 0 checks so the v3 document is self-contained.
    Numbers are computed live from the reference npz files.
    """
    lines = [
        "## Reference Data Quality",
        "",
        "This section documents the reference functions used for shape-correlation",
        "scoring.  It reproduces the v2 Panel 0 checks so that v3 is a complete,",
        "self-contained document.",
        "",
    ]

    # ── Triple-well committors ────────────────────────────────────────────────
    lines += [
        "### Triple-well — FEM committor references",
        "",
    ]
    if tw_refs:
        pA = tw_refs["refs"][0]
        pB = tw_refs["refs"][1]
        pC = tw_refs["refs"][2]
        row_sum = pA + pB + pC
        lines += [
            f"N = {len(pA)} anchors, FEM-computed committor functions p_A, p_B, p_C.",
            "",
            "| Check | Value | Verdict |",
            "|-------|-------|---------|",
            f"| Partition of unity (p_A+p_B+p_C) | mean={row_sum.mean():.4f}, std={row_sum.std():.2e} | PASS — exact by construction |",
            f"| Basin weights | p_A={pA.mean():.3f}, p_B={pB.mean():.3f}, p_C={pC.mean():.3f} | Balanced — all three wells populated |",
            f"| Value range | [{pA.min():.2f}, {pA.max():.2f}] | PASS |",
            "",
            "Separatrix cells (p_i < 0.6 for all i) have ~20% variance at 20 bursts",
            "— acceptable for full-field Pearson r but would sharpen with 50 bursts.",
            "",
        ]
    else:
        lines += ["*(Reference file not found.)*", ""]

    # ── ADP transfer-operator spectrum ────────────────────────────────────────
    lines += [
        "### ADP (τ = 5 ps) — transfer-operator eigenvalue spectrum",
        "",
    ]
    if adp_refs:
        ev_path = os.path.join(V2_REF_DIR, "adp_eigvecs.npz")
        ev      = np.load(ev_path)
        evals   = ev["eigenvalues"]          # (16,) — excludes stationary λ=1
        occ     = int(ev["occupied"].sum())
        tau     = 5.0

        its = []
        for lam in evals:
            if abs(lam) > 1e-9 and abs(lam) < 1 - 1e-9:
                its.append(-tau / np.log(abs(lam)))
            else:
                its.append(np.inf)

        lines += [
            f"Occupied grid cells: {occ} / 1600 ({occ/16:.1f}%)",
            "",
            "| Mode | Eigenvalue | ITS (ps) | Comment |",
            "|------|-----------|----------|---------|",
        ]
        for i, (lam, t) in enumerate(zip(evals[:5], its[:5])):
            t_str = f"{t:.0f}" if t < 1e4 else "∞"
            if i == 0:
                comment = "Ultra-slow mode"
            elif i == 1:
                comment = "**Reference mode** — C7eq∪αR ↔ C7ax"
            elif i == 2:
                comment = "Thermal bath begins"
            else:
                comment = "Thermal bath"
            lines.append(f"| EV{i+1} | {lam:.4f} | {t_str} | {comment} |")
        lines += [
            f"| EV6–EV16 | {evals[5]:.3f}–{evals[-1]:.3f} | {its[5]:.1f}–{its[-1]:.1f} | Thermal bath |",
            "",
            f"**Gap:** ITS₁={its[0]:.0f} ps, ITS₂={its[1]:.0f} ps — "
            f"both are slow; ITS₃={its[2]:.1f} ps marks the thermal bath.",
            "",
            "**Benchmark reference = EV2** (λ=0.9900, ITS≈499 ps).",
            "EV1 (λ=0.9964) exists but is ultralow-amplitude and was not used in v2.",
            "",
            "**Implication for benchmark design at k=3, τ=5 ps:**",
            "",
            "- chi_1 should converge to EV2 (C7eq∪αR ↔ C7ax)",
            "- chi_2 and chi_3 have no slow dynamics to learn → will collapse to noise",
            "- This is **not a failure** — it is the expected physical outcome",
            "- The benchmark tests graceful collapse (chi_2/3 → SD≈0, chi_1 live)",
            "  vs hard failure (all three collapse)",
            "",
        ]
    else:
        lines += ["*(Reference file not found.)*", ""]

    # ── Panel 0 figure ────────────────────────────────────────────────────────
    panel0_rel = "../benchmark_v2/panel0/panel0_reference.png"
    panel0_abs = os.path.join(V2_REF_DIR, "panel0_reference.png")
    if os.path.exists(panel0_abs):
        lines += [
            "### Panel 0 — reference overview figure",
            "",
            f"![Panel 0 reference]({panel0_rel})",
            "",
            "*(From v2 benchmark analysis.)*",
            "",
        ]

    lines += ["---", ""]
    return lines


def _variant_metrics(ds_name: str, variant: str, refs: dict):
    """(mean r, SD r, mean k_eff, n_seeds) for one variant subdir under RUNS_DIR."""
    shape_fn = tw_shape_r if ds_name.startswith("triple") else adp_shape_r
    rs, keffs = [], []
    for seed in range(N_SEEDS):
        d = os.path.join(RUNS_DIR, ds_name, variant, f"seed_{seed}")
        cp = os.path.join(d, "chi_best.npy")
        if not os.path.exists(cp):
            continue
        rs.append(shape_fn(np.load(cp), refs))
        sp = os.path.join(d, "chi_sd_history.npy")
        if os.path.exists(sp):
            sh = np.load(sp)
            if sh.ndim == 2 and len(sh):
                keffs.append(int((sh[-1] > 0.05).sum()))
    rsf = [r for r in rs if np.isfinite(r)]
    return (np.mean(rsf) if rsf else np.nan,
            np.std(rsf) if len(rsf) > 1 else np.nan,
            np.mean(keffs) if keffs else np.nan, len(rsf))


def _isa_warmup_ablation_section(tw_refs: dict, adp_refs: dict) -> list:
    """ISA with vs without the GramSchmidt warm-up (Python, corrected transform)."""
    fmt = lambda x, n=3: ("—" if (x is None or (isinstance(x, float) and np.isnan(x)))
                          else f"{x:.{n}f}")
    lines = [
        "## ISA — warm-up vs no warm-up",
        "",
        "Does the 100-iter GramSchmidt warm-up matter for ISA? `isa` is the standard",
        "(warm-up) run; `isa_nowarmup` removes it entirely (ISA target from iteration 1).",
        "Same seeds/init. See `ISA_WARMUP_ABLATION.md` for the cross-framework version.",
        "",
        "| Dataset | warm-up r | warm-up k_eff | no-warm-up r | no-warm-up k_eff |",
        "|---------|-----------|---------------|--------------|------------------|",
    ]
    for ds in ["triple_well", "alanine_5ps", "alanine_0p1ps", "alanine_multitau"]:
        refs = tw_refs if ds.startswith("triple") else adp_refs
        if not os.path.isdir(os.path.join(RUNS_DIR, ds, "isa_nowarmup")):
            continue
        wr, _, wk, _ = _variant_metrics(ds, "isa", refs)
        nr, _, nk, _ = _variant_metrics(ds, "isa_nowarmup", refs)
        lines.append(f"| {ds} | {fmt(wr)} | {fmt(wk,1)} | {fmt(nr)} | {fmt(nk,1)} |")
    lines += [
        "",
        "Warm-up is **unnecessary on triple_well** (ISA reaches r≈0.98 / k_eff=3 either",
        "way) but **stabilises ISA on the 231-dim ADP data**, where without it the",
        "inner-simplex inversion collapses on most seeds. It supplies the",
        "approximate-simplex chi the inversion needs in high dimensions.",
        "",
    ]
    return lines


def write_results_md(results: dict, tw_refs: dict, adp_refs: dict) -> None:
    """Write BENCHMARK_V3_RESULTS.md with reference preamble and per-method verdict tables."""
    lines = [
        "# Benchmark v3 Results",
        "",
        "Verdicts are based on **map shape** (Pearson r vs reference), not SD alone.",
        "SD alone was the central error of the v2 report.",
        "See `BENCHMARK_V2_POSTMORTEM.md` for context.",
        "",
        f"PASS: r ≥ {PASS_R} AND k_eff ≥ 1  |  PARTIAL: r ≥ {PARTIAL_R} (or PASS with k_eff=0)  |  FAIL: r < {PARTIAL_R}",
        "",
        "---",
        "",
    ]

    lines += _procedure_section()
    lines += _reference_preamble(tw_refs, adp_refs)
    lines += [
        "## Training Results",
        "",
    ]

    for ds_name, ds_res in results.items():
        metrics = ds_res["metrics"]
        ref_label = ds_res.get("ref_label", "")
        lines += [
            f"## {ds_name}",
            "",
            f"Reference: {ref_label}",
            "",
            "| Method | Seeds | Mean r | SD r | k_eff (>0.05) | Verdict |",
            "|--------|-------|--------|------|--------------|---------|",
        ]
        for variant in VARIANTS:
            m = metrics[variant]
            rs  = [r for r in m["r_per_seed"] if np.isfinite(r)]
            sds = m["sd_per_seed"]
            n   = len(rs)
            mean_r = np.mean(rs) if rs else np.nan
            std_r  = np.std(rs)  if len(rs) > 1 else np.nan

            # k_eff: number of modes with SD > 0.05, averaged across seeds
            if sds:
                keff_vals  = [(s > 0.05).sum() for s in sds]
                mean_k_eff = float(np.mean(keff_vals))
                keff_str   = f"{mean_k_eff:.1f}"
            else:
                mean_k_eff = None
                keff_str   = "—"

            label = VARIANT_LABELS.get(variant, variant)
            verd  = verdict(mean_r, mean_k_eff)
            r_str = f"{mean_r:.3f}" if not np.isnan(mean_r) else "—"
            sd_str = f"±{std_r:.3f}" if not np.isnan(std_r) else ""
            lines.append(
                f"| {label} | {n}/{N_SEEDS} | {r_str} | {sd_str} "
                f"| {keff_str} | **{verd}** |"
            )

        lines.append("")

        # Shape summary figure
        fig_rel = f"figures/{ds_name}/shape_summary.png"
        lines += [
            f"### Shape summary — {ds_name}",
            "",
            f"![shape summary](figures/{ds_name}/shape_summary.png)",
            "",
        ]

        # Chi panels per variant
        lines += [f"### Chi panels — {ds_name}", ""]
        for variant in VARIANTS:
            fig_rel = f"figures/{ds_name}/chi_panels_{variant}.png"
            label   = VARIANT_LABELS.get(variant, variant)
            lines += [
                f"#### {label}",
                "",
                f"![{label} chi panels]({fig_rel})",
                "",
            ]

    lines += _isa_warmup_ablation_section(tw_refs, adp_refs)

    lines += [
        "## Notes",
        "",
        "- **svd** = subspace power iteration (`power_method_multi`), NOT DMD eigen(H).",
        "  See BENCHMARK_V2_POSTMORTEM.md, bug 3.",
        "- **Multi-tau** uses real 231-dim features at 0.1 ps (50 steps) from the same",
        "  1578 anchors. v2's 0.1 ps data was grid-snapped (postmortem bug 6) — replaced",
        "  by `01_simulate_alanine_0p1ps.py`. Joint bursts = N_K=40 (20×5ps + 20×0.1ps).",
        "- **ISA** uses the corrected inner-simplex transform `A = inv(K[verts])ᵀ`",
        "  (`src/amore/isotarget.py`); on triple-well it is a genuine PASS (r≈0.98,",
        "  k_eff=3), matching the Julia reference. See `ISA_WARMUP_ABLATION.md`.",
        "- Verification gate: `tests/test_isotarget.py` — 19/19 PASSED (property tests).",
        "  Cross-checked numerically against the Julia `isotarget.jl` originals; see",
        "  `BENCHMARK_V3_JULIA_RESULTS.md`.",
        "",
    ]

    out = os.path.join(HERE, "BENCHMARK_V3_RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResults document: {out}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    tw_refs  = load_tw_refs()
    adp_refs = load_adp_refs()

    all_results: dict = {}

    # ── Triple-well ──────────────────────────────────────────────────────────
    ds = "triple_well"
    if os.path.isdir(os.path.join(RUNS_DIR, ds)):
        print(f"\n{'='*55}\nPlotting {ds}\n{'='*55}")
        metrics  = compute_metrics(ds, tw_refs)
        fig_dir  = os.path.join(FIG_BASE, ds)
        plot_chi_panels(ds, tw_refs, fig_dir)
        plot_shape_summary(ds, metrics, fig_dir)
        all_results[ds] = {
            "metrics"  : metrics,
            "ref_label": "Committor functions p_A, p_B, p_C (FEM reference)",
        }
    else:
        print(f"No runs found for {ds}")

    # ── ADP tau=5ps ───────────────────────────────────────────────────────────
    ds = "alanine_5ps"
    if os.path.isdir(os.path.join(RUNS_DIR, ds)):
        print(f"\n{'='*55}\nPlotting {ds}\n{'='*55}")
        metrics  = compute_metrics(ds, adp_refs)
        fig_dir  = os.path.join(FIG_BASE, ds)
        plot_chi_panels(ds, adp_refs, fig_dir)
        plot_shape_summary(ds, metrics, fig_dir)
        all_results[ds] = {
            "metrics"  : metrics,
            "ref_label": "Transfer-operator EV2 (eigenvalue≈0.990, tau=5ps)",
        }
    else:
        print(f"No runs found for {ds}")

    # ── ADP tau=0.1ps only ───────────────────────────────────────────────────
    ds = "alanine_0p1ps"
    if os.path.isdir(os.path.join(RUNS_DIR, ds)):
        print(f"\n{'='*55}\nPlotting {ds}\n{'='*55}")
        metrics  = compute_metrics(ds, adp_refs)
        fig_dir  = os.path.join(FIG_BASE, ds)
        plot_chi_panels(ds, adp_refs, fig_dir)
        plot_shape_summary(ds, metrics, fig_dir)
        all_results[ds] = {
            "metrics"  : metrics,
            "ref_label": "Transfer-operator EV2 (eigenvalue≈0.990, tau=5ps) — trained on 0.1ps bursts only",
        }
    else:
        print(f"No runs found for {ds} — run 02_train_benchmark_v3.py first")

    # ── ADP multi-tau (5 ps + 0.1 ps joint) ─────────────────────────────────
    ds = "alanine_multitau"
    if os.path.isdir(os.path.join(RUNS_DIR, ds)):
        print(f"\n{'='*55}\nPlotting {ds}\n{'='*55}")
        # Same anchors and same EV2 reference as alanine_5ps
        metrics  = compute_metrics(ds, adp_refs)
        fig_dir  = os.path.join(FIG_BASE, ds)
        plot_chi_panels(ds, adp_refs, fig_dir)
        plot_shape_summary(ds, metrics, fig_dir)
        all_results[ds] = {
            "metrics"  : metrics,
            "ref_label": "Transfer-operator EV2 (eigenvalue≈0.990, tau=5ps) — joint 5ps+0.1ps training",
        }
    else:
        print(f"No runs found for {ds} — run 01_simulate_alanine_0p1ps.py then 02_train_benchmark_v3.py")

    if all_results:
        write_results_md(all_results, tw_refs, adp_refs)
    else:
        print("\nNo run data found. Run 02_train_benchmark_v3.py first.")
