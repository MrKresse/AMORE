# -*- coding: utf-8 -*-
"""Chi panels for the ISA warm-up ablation configs, reusing 03_plot's plotter.
Figures -> figures_isa_ablation/{ds}/{tag}.png"""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "plot_v3", os.path.join(HERE, "03_plot_benchmark_v3.py"))
plot = importlib.util.module_from_spec(spec); spec.loader.exec_module(plot)

FIG = os.path.join(HERE, "figures_isa_ablation")
os.makedirs(FIG, exist_ok=True)
tw, adp = plot.load_tw_refs(), plot.load_adp_refs()
DS = [("triple_well", tw), ("alanine_5ps", adp),
      ("alanine_0p1ps", adp), ("alanine_multitau", adp)]
# (tag, runs_root, subdir, label)
CFG = [
    ("julia_nowarmup",  "runs_julia", "isa_nowarmup",       "Julia ISA (no warm-up)"),
    ("py_buggy_warmup", "runs",       "isa_buggy",          "Python ISA buggy (warm-up)"),
    ("py_fixed_warmup", "runs",       "isa",                "Python ISA FIXED (warm-up)"),
    ("py_fixed_nowarmup","runs",      "isa_nowarmup",       "Python ISA FIXED (no warm-up)"),
]
for ds, refs in DS:
    for tag, root, sub, label in CFG:
        if not os.path.isdir(os.path.join(HERE, root, ds, sub)):
            continue
        plot.RUNS_DIR = os.path.join(HERE, root)
        plot.VARIANTS = [sub]
        plot.VARIANT_LABELS = {sub: f"{label} — {ds}"}
        fig_dir = os.path.join(FIG, ds)
        plot.plot_chi_panels(ds, refs, fig_dir)
        src = os.path.join(fig_dir, f"chi_panels_{sub}.png")
        dst = os.path.join(fig_dir, f"{tag}.png")
        if os.path.exists(src) and src != dst:
            os.replace(src, dst)
        print("saved", os.path.relpath(dst, HERE))
print("done")
