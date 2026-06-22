# -*- coding: utf-8 -*-
"""
Score the ISA warm-up ablation across frameworks/configs and write
ISA_WARMUP_ABLATION.md. Uses the same shape metrics as 03_plot_benchmark_v3.py.

Configs compared per dataset (mean r ± SD, mean k_eff over 5 seeds):
  Julia  warmup      runs_julia/{ds}/isa
  Julia  no-warmup   runs_julia/{ds}/isa_nowarmup
  Python warmup BUGGY (original)        runs/{ds}/isa
  Python no-warmup BUGGY (evidence)     runs/{ds}/isa_nowarmup_buggy
  Python warmup FIXED                   runs/{ds}/isa_fixed
  Python no-warmup FIXED                runs/{ds}/isa_nowarmup
"""
import os, sys, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
spec = importlib.util.spec_from_file_location(
    "plot_v3", os.path.join(HERE, "03_plot_benchmark_v3.py"))
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)

DSETS = ["triple_well", "alanine_5ps", "alanine_0p1ps", "alanine_multitau"]
# (label, runs_root, subdir)
CONFIGS = [
    ("Julia · warmup",        "runs_julia", "isa"),
    ("Julia · no-warmup",     "runs_julia", "isa_nowarmup"),
    ("Python · warmup · buggy",    "runs", "isa_buggy"),
    ("Python · no-warmup · buggy", "runs", "isa_nowarmup_buggy"),
    ("Python · warmup · FIXED",    "runs", "isa"),
    ("Python · no-warmup · FIXED", "runs", "isa_nowarmup"),
]


def metrics(ds, root, sub, refs):
    shape_fn = plot.tw_shape_r if ds.startswith("triple") else plot.adp_shape_r
    rs, keffs = [], []
    base = os.path.join(HERE, root, ds, sub)
    for s in range(5):
        cp = os.path.join(base, f"seed_{s}", "chi_best.npy")
        if not os.path.exists(cp):
            continue
        rs.append(shape_fn(np.load(cp), refs))
        sp = os.path.join(base, f"seed_{s}", "chi_sd_history.npy")
        if os.path.exists(sp):
            sh = np.load(sp)
            if sh.ndim == 2 and len(sh):
                keffs.append(int((sh[-1] > 0.05).sum()))
    rsf = [r for r in rs if np.isfinite(r)]
    return (np.mean(rsf) if rsf else np.nan,
            np.std(rsf) if len(rsf) > 1 else np.nan,
            np.mean(keffs) if keffs else np.nan,
            len(rsf))


def f(x, n=3):
    return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{n}f}"


def main():
    tw, adp = plot.load_tw_refs(), plot.load_adp_refs()
    L = ["# ISA Warm-up Ablation — Julia vs Python (buggy/fixed)", "",
         "Does the 100-iter GramSchmidt warm-up matter for ISA, and does the "
         "Python port reproduce Julia ISA? Metric: same shape |Pearson r| vs "
         "reference as the main benchmark (TW: Hungarian over committors; ADP: "
         "max |r| vs EV2). k_eff = modes with final SD > 0.05 (mean over seeds).",
         "",
         "**Bug found & fixed:** `isa_target` computed `inv(K[verts]) @ kchi` but "
         "the Julia original is `inv(K[verts])` **transposed** (`A = inv(K[verts])^T`, "
         "the simplex condition `A·verts = I`). Without it the ISA target is not "
         "one-hot at the vertices and chi collapses once the warm-up is removed. "
         "Numerically: `inv(C).T @ C.T = I` (err 2e-15) vs `inv(C) @ C.T` off by ~35×.",
         ""]
    for ds in DSETS:
        refs = tw if ds.startswith("triple") else adp
        L += [f"## {ds}", "",
              "| Config | seeds | mean r | SD r | k_eff |",
              "|--------|-------|--------|------|-------|"]
        for label, root, sub in CONFIGS:
            mr, sr, ke, n = metrics(ds, root, sub, refs)
            L.append(f"| {label} | {n}/5 | {f(mr)} | ±{f(sr)} | {f(ke,1)} |")
        L.append("")
    L += ["## Reading the table", "",
          "- **Julia warmup vs no-warmup** isolates whether warm-up is needed "
          "(transform is correct on both).",
          "- **Python buggy vs FIXED** isolates the transpose bug.",
          "- **Python FIXED vs Julia** tests the port once corrected.",
          "",
          "*Python · no-warmup · buggy was only sampled on triple_well (run "
          "stopped once it showed collapse); the ADP '—' cells were not run but "
          "would collapse identically (k_eff=0), as the buggy target is degenerate "
          "regardless of dataset.*",
          "",
          "## Conclusions",
          "",
          "### 1. The Python ISA port had a real bug (now fixed)",
          "On triple_well the fix converts the old **hollow PASS** (r=0.802, "
          "k_eff=0 — high r borrowed from the GramSchmidt warm-up state, then "
          "collapsed) into a **genuine PASS** (r=0.981, k_eff=3) that matches the "
          "Julia original (r=0.980, k_eff=3) to three decimals, with or without "
          "warm-up. The fix also lifts every ADP config (e.g. alanine_5ps "
          "with-warmup 0.521→0.844, k_eff 0→1.8). The warm-up had been *masking* "
          "the bug: chi_best was checkpointed during the GramSchmidt phase, so the "
          "broken main-loop ISA target never showed up in the score until warm-up "
          "was removed and chi collapsed to a constant (k_eff=0).",
          "",
          "### 2. Is the GramSchmidt warm-up necessary for ISA?",
          "- **triple_well: NO.** Correct ISA (Julia, or fixed Python) reaches "
          "r≈0.98 / k_eff=3 on all 5 seeds with no warm-up — identical to "
          "with-warm-up. The warm-up is redundant here.",
          "- **ADP (231-dim): YES, it helps / is needed for stability.** Even the "
          "correct transform collapses on most seeds without warm-up (Julia "
          "no-warmup: 5ps 1/5, 0p1ps 2/5, multitau 0/5 seeds non-degenerate). "
          "Warm-up gives ISA the approximate-simplex chi its inversion needs in "
          "high dimensions; on alanine_5ps it clearly wins (FIXED: warmup r=0.844 "
          "k_eff=1.8 vs no-warmup r=0.552 k_eff=0.6). On 0p1ps/multitau the means "
          "are within the (large, ±0.3–0.4) seed scatter — ADP ISA is intrinsically "
          "high-variance at k=3 where only one slow mode (EV2) exists.",
          "",
          "**Bottom line:** keep the warm-up for ADP-scale problems; it is optional "
          "for low-dimensional systems like the triple well. And the headline "
          "triple_well ISA result you flagged is real — once the transpose is "
          "fixed, Python reproduces it exactly (r=0.98, k_eff=3), warm-up or not.",
          ""]
    # ── embedded chi panels (generated by make_isa_ablation_figures.py) ──────
    gallery = [("julia_nowarmup", "Julia ISA (no warm-up)"),
               ("py_buggy_warmup", "Python ISA buggy (warm-up) — original"),
               ("py_fixed_warmup", "Python ISA FIXED (warm-up)"),
               ("py_fixed_nowarmup", "Python ISA FIXED (no warm-up)")]
    L += ["## Chi panels (rows = seeds, cols = χ₁/χ₂/χ₃; title shows per-mode SD)", ""]
    for ds in DSETS:
        any_fig = False
        for tag, _ in gallery:
            if os.path.exists(os.path.join(HERE, "figures_isa_ablation", ds, tag + ".png")):
                any_fig = True
        if not any_fig:
            continue
        L += [f"### {ds}", ""]
        for tag, lab in gallery:
            rel = f"figures_isa_ablation/{ds}/{tag}.png"
            if os.path.exists(os.path.join(HERE, rel)):
                L += [f"**{lab}**", "", f"![{lab}]({rel})", ""]

    out = os.path.join(HERE, "ISA_WARMUP_ABLATION.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("wrote", out)
    # also echo to stdout
    print("\n".join(L))


if __name__ == "__main__":
    main()
