# -*- coding: utf-8 -*-
"""Emit examples/2cm2_full/spectral_gap_pocket.ipynb: re-validate "how many metastable
states" on the POCKET feature set (data.build_features_pocket: whole-protein side-chain
COM-COM distances + all-atom 5A ligand<->protein contacts, PBC) -- the feature set this
session's actual chi-MEP/edge/heatmap work has used throughout (via
comfeat.load_trained_model_pocket), assumed k=3 without ever re-running this specific
overparametrization/spectral-gap check on it.

Why this needs re-asking rather than assumed: `build_spectral_gap.py`'s own re-validation
of the LIGAND-INCLUSIVE feature set (data.build_features_lig) found k=4, not the k=3 that
held on the original protein-only features -- two independently-trained overparametrized
models (k_trained=5,6) agreed on a spectral-gap elbow at process 4. The pocket feature set
is a THIRD, separately-engineered feature set (built later, to fix H-atom relaxation-lag
artifacts in build_features_lig's COM-only ligand term) that has never had this check
applied at all -- the k=3 pocket model in active use could just as easily be
underparametrized.

Mirrors build_spectral_gap.py's structure/methodology exactly (same amore.inverse_pcca
spectral-gap machinery, same eval_chi-from-fresh-forward-pass convention avoiding any
possibly-stale cached chi, same reversible=True-with-Schur-fallback), swapping only the
feature-set/model-loading calls (pocket=True instead of include_ligand=True).

Run:  python build_spectral_gap_pocket.py
Then: jupyter nbconvert --to notebook --execute --inplace spectral_gap_pocket.ipynb
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# 2cm2 (full trajectory) — how many metastable states, POCKET features?

`build_spectral_gap.py`'s ligand-inclusive re-validation found k=4 (not the protein-only
run's k=3) -- two independently-trained overparametrized models (k=5,6) agreeing on a
spectral-gap elbow at process 4. The POCKET feature set (whole-protein side-chain COM-COM
+ all-atom 5A ligand-protein contacts, `data.build_features_pocket`) is a third,
separately-engineered feature set -- built later this session to fix H-atom
relaxation-lag artifacts the ligand-inclusive COM-only ligand term caused in chi-MEP work
-- that has been used throughout this session's actual pocket-model work assuming k=3,
without ever running this same check on it. This notebook does that: same
`amore.inverse_pcca` overparametrization/spectral-gap methodology, same lag=20 full
trajectory, bracketing k=3 on both sides (k=3,4,5,6).
""")

code(r"""import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

SRC = os.path.abspath(os.path.join("..", "..", "src"))
LIB = os.path.abspath(os.path.join("..", "2cm2", "lib"))
sys.path.insert(0, SRC)
sys.path.insert(0, LIB)
import data, train, run, analysis as A
from amore.isokann import ChiNetMulti
from amore.inverse_pcca import (
    inverse_pcca, group_conjugate_pairs, find_spectral_gap, plot_complex_plane, rate_matrix,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 1.0   # arbitrary consistent unit; the gap is about ratios, not absolute time
plt.rcParams["figure.dpi"] = 120
print("device:", DEVICE)""")

md(r"""## Full trajectory, lag=20, pocket features

Same dynamic NSTART/NEND/LAG repointing technique as the ligand-inclusive re-validation.
`pocket=True` (takes precedence over `include_ligand`) gives every cache path its own
`_pocket` tag, so this never touches or collides with either the protein-only or
ligand-inclusive full-trajectory caches under the same `$CM2_SCRATCH`.
""")

code(r"""data.NSTART = 0
data.NEND = 2979     # NEND + LAG must stay < 3000 (the trajectory's frame count)
data.LAG = 20
KVALS = [3, 4, 5, 6]   # brackets k=3 (this session's working assumption) both sides,
                       # same narrower re-validation sweep as the ligand-inclusive check
print(f"anchors: [{data.NSTART}, {data.NEND})  lag={data.LAG}  pocket=True  "
      f"-> {data.NEND - data.NSTART} anchors")

feats = data.build_features_pocket(use_pbc=True, verbose=True)
D0n, Dtn, mu, sd = train.normalise(feats["D0"], feats["Dt"])
D0n = D0n.to(DEVICE); Dtn = Dtn.to(DEVICE)
N, F = D0n.shape; Kb = Dtn.shape[1]
print(f"N anchors={N}  F features={F} ({len(feats['res_pairs'])} sidechain-COM-COM + "
      f"{feats['n_contact']} ligand-protein all-atom contact)  Kb(lag replicas)={Kb}")""")

code(r"""def load_net(k, F=F):
    # run.get_model trains fresh (cached to scratch) if this (k, NSTART, NEND, LAG,
    # pocket) combination hasn't been trained before.
    m = run.get_model(k, pocket=True, verbose=True)
    net = ChiNetMulti(F, k, hidden=m["hidden"]).to(DEVICE)
    net.load_state_dict(m["net_state"]); net.eval()
    return net, m

def eval_chi(net, D0n=D0n):
    with torch.no_grad():
        return net(D0n).cpu().numpy()

def make_propagate(net, Dtn=Dtn, N=N, F=F, Kb=Kb):
    def propagate():
        with torch.no_grad():
            flat = Dtn.reshape(N * Kb, F)
            return net(flat).reshape(N, Kb, -1).mean(1).cpu().numpy()
    return propagate

results = {}
for k in KVALS:
    print(f"=== training/loading k={k} (full trajectory, lag={data.LAG}, pocket) ===")
    net, m = load_net(k)
    chi = eval_chi(net)
    keff = int((chi.std(0) > 0.05).sum())
    print(f"  chi std/output: {chi.std(0).round(3)}  k_eff={keff}")
    if keff < 2:
        print(f"  collapsed (k_eff={keff}), skipping inverse_pcca for this k")
        results[k] = dict(m=m, net=net, chi=chi, result=None)
        continue
    try:
        result = inverse_pcca(chi, make_propagate(net), TAU, reversible=True)
        rev = True
    except ValueError as e:
        print(f"  reversible=True failed ({e})\n  -> falling back to reversible=False")
        result = inverse_pcca(chi, make_propagate(net), TAU, reversible=False)
        rev = False
    processes = group_conjugate_pairs(result.lam)
    gap = find_spectral_gap(processes) if len(processes) >= 2 else None
    results[k] = dict(m=m, net=net, chi=chi, result=result, reversible=rev,
                      processes=processes, gap=gap)
    print(f"  reversible={rev}  residual(RMS/entry)={result.residual/np.sqrt(chi.size):.4f}")
    print(f"  distinct processes ({len(processes)}):")
    for p in processes:
        print(f"    |lambda|={p.modulus:.4f}  {p.kind}  lam={p.lam:.4f}")
    if gap is not None:
        print(f"  find_spectral_gap: k={gap.k} processes above it "
              f"({gap.modulus_above:.4f} -> {gap.modulus_below:.4f}, drop={gap.gap:.4f})")
    print()""")

md(r"""## Loss curves

`train` (MSE to the ISA regression target) and `val` (held-out Gram-Schmidt residual) per
outer ISA power iteration.
""")

code(r"""fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, k in zip(axes.ravel(), KVALS):
    A.plot_loss(results[k]["m"], ax=ax)
fig.tight_layout(); plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight"); plt.show()""")

code(r"""fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, k in zip(axes.ravel(), KVALS):
    if results[k]["result"] is None:
        ax.set_title(f"k={k} (collapsed)")
        continue
    plot_complex_plane(ax, results[k]["result"].lam)
    ax.set_title(f"k={k} (full traj, lag={data.LAG}, pocket)")
fig.tight_layout(); plt.savefig("spectral_gap_complex_plane.png", dpi=150, bbox_inches="tight"); plt.show()""")

code(r"""fig, ax = plt.subplots(figsize=(8, 5))
for k in KVALS:
    if results[k]["result"] is None:
        continue
    moduli = [p.modulus for p in results[k]["processes"]]
    ax.plot(np.arange(1, len(moduli) + 1), moduli, "o-", label=f"k={k}")
ax.axhline(1.0, color="gray", lw=1, ls=":")
ax.set_xlabel("process index (complex-conjugate pairs counted once)")
ax.set_ylabel("|lambda|")
ax.set_title(f"2cm2 full trajectory (lag={data.LAG}, pocket features): |lambda| by distinct process")
ax.legend()
fig.tight_layout(); plt.savefig("spectral_gap_lambda.png", dpi=150, bbox_inches="tight"); plt.show()""")

nb["cells"] = cells
path = os.path.join(HERE, "spectral_gap_pocket.ipynb")
with open(path, "w") as f:
    nbf.write(nb, f)
print("wrote", path)
