# -*- coding: utf-8 -*-
"""
Render the full chi panels + shape-summary figures for the Julia runs
(runs_julia/), reusing the exact plotting code from 03_plot_benchmark_v3.py
with the variant list/labels and RUNS_DIR redirected. Figures go to
figures_julia/{dataset}/.
"""
import os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "plot_v3", os.path.join(HERE, "03_plot_benchmark_v3.py"))
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)

# Redirect to the Julia outputs and adapt the variant set (svd_power, no vamp2)
plot.RUNS_DIR = os.path.join(HERE, "runs_julia")
plot.FIG_BASE = os.path.join(HERE, "figures_julia")
os.makedirs(plot.FIG_BASE, exist_ok=True)
plot.VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd_power"]
plot.VARIANT_LABELS = {
    "isa": "V2-ISA", "gramschmidt": "V3-GramSchmidt", "pseudoinv": "V4-PseudoInv",
    "cross": "V5-Cross", "svd_power": "Power-Iter (Julia)",
}

tw_refs = plot.load_tw_refs()
adp_refs = plot.load_adp_refs()

DS = [("triple_well", tw_refs), ("alanine_5ps", adp_refs),
      ("alanine_0p1ps", adp_refs), ("alanine_multitau", adp_refs)]

for ds, refs in DS:
    if not os.path.isdir(os.path.join(plot.RUNS_DIR, ds)):
        print("skip (no runs):", ds); continue
    print("=" * 55, "\n", ds, "\n", "=" * 55)
    fig_dir = os.path.join(plot.FIG_BASE, ds)
    metrics = plot.compute_metrics(ds, refs)
    plot.plot_chi_panels(ds, refs, fig_dir)
    plot.plot_shape_summary(ds, metrics, fig_dir)

print("\nFigures written under:", plot.FIG_BASE)
