# -*- coding: utf-8 -*-
"""Builder for reversible.ipynb (run: python build_reversible.py)."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

C = []
def md(s): C.append(new_markdown_cell(s))
def co(s): C.append(new_code_cell(s))

md(r"""# Reversible multi-D ISOKANN benchmark

Benchmarks the multi-dimensional ISOKANN target transforms in `src/amore` against
**numerical ground-truth** reference solutions on two reversible systems. Self-contained:
running top-to-bottom regenerates every table and figure from the cached training runs.

## Two method families (each with its NATURAL representation)
A system with `S=3` metastable states has a 3-D slow subspace = **constant + 2 non-trivial
eigenfunctions**, equivalently **3 memberships** (a partition of unity). So the two families
output different but equivalent things:

| family | head | output | k | methods |
|---|---|---|---|---|
| **membership** | softmax `ChiNetMulti` | 3 memberships (χ≥0, Σχ=1) | 3 | ISA, VAMP (=VAMPnets), [Schur-ISA, GPCCA] |
| **basis** | linear `ChiNetMultiLinear` (constant-deflated) | 2 non-trivial eigenfunctions | 2 | GramSchmidt, PseudoInv, Cross, SVD-Power |

The two are linked by **PCCA+**: an eigenfunction basis is rotated into memberships by the
inner-simplex map (§2). The softmax head enforces the simplex architecturally; deflating the
Koopman target makes the linear basis learn the 2 non-trivial eigenfunctions (not the constant).

**No warm-up is used.** The softmax membership head removes the amplitude-collapse / mode-selection
that the *linear* ISA suffered on 231-D ADP — softmax-ISA recovers both slow modes from scratch
(§3 documents why, against the old linear-ISA + 1-D warm-up band-aid).

## Scoring (both computed per method)
- **eig |r|** — eigenfunction-subspace recovery: Hungarian |Pearson r| of the 2 non-trivial
  eigenfunctions vs the true transfer-operator EV₂/EV₃ (uniform metric, both systems).
  Membership outputs are converted to eigenfunctions by mean-center + SVD.
- **memb |r|** — Hungarian |r| of the 3 memberships vs the committors (triple-well only, where
  committor ground truth exists). Basis outputs are converted to memberships by PCCA+.

## Held constant vs varied
Constant per system: data/anchors/bursts, trunk arch in→[128,32,8] (Tanh hidden), Adam lr=1e-3,
grad-clip 5, plateau stop + best-on-held-out checkpoint, 80/20 split. Varied: the method
(family = head+target), the seed (5), the system. Large data/runs live on scratch (`paths.py`);
rebuild with `python generate_data.py all` and `python run_ensemble.py all`.""")

co("""import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("lib"))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
import paths, systems, ground_truth as gt, harness, plotting
print(paths.summary())

SEEDS = list(range(5))
MEMBERSHIP = ["isa", "vamp"]            # softmax, k=3  (reversible set)
BASIS = ["gramschmidt", "pseudoinv", "cross", "svd_power"]   # linear, k=2
REV_VARIANTS = ["isa", "gramschmidt", "pseudoinv", "cross", "svd_power", "vamp"]

tw  = systems.load_triple_well()
adp = systems.load_adp_300k_0p1(max_anchors=25000)
SYS = {"triple_well": tw, "adp_300k_0p1": adp}
print("triple_well :", tw["feat"].shape, "| adp_300k_0p1:", adp["feat"].shape)
for v in REV_VARIANTS:
    print(f"  {v:12s} family={'membership(softmax,k=3)' if harness.is_membership(v) else 'basis(linear,k=2)'}")""")

md("""## 1. Numerical ground truth

Computed directly from the simulation data (no neural net): committors for the triple-well,
transfer-operator eigenvectors for both systems.""")

co("""# triple-well basin committors (the membership reference) and its EV2/EV3 (the eigenfunction reference)
fig = plotting.plot_references(tw); fig.suptitle("triple_well — basin committors p_A,p_B,p_C (0..1)", y=1.02); plt.show()
ev_refs, ev_vals, _ = gt.tw_eigvecs()
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for j,(nm) in enumerate(["EV2","EV3"]):
    m=np.abs(tw["ev_refs"][j]).max()
    s=ax[j].scatter(tw["coords"][:,0], tw["coords"][:,1], c=tw["ev_refs"][j], s=8, cmap="coolwarm", vmin=-m, vmax=m)
    ax[j].set_title(f"triple_well {nm} (λ={ev_vals[j+1]:.3f})"); ax[j].set_xlabel("x"); ax[j].set_ylabel("y"); plt.colorbar(s,ax=ax[j],fraction=.046)
fig.suptitle("triple_well — transfer-operator eigenfunctions (signed)", y=1.02); plt.tight_layout(); plt.show()""")

co("""# alanine dipeptide: stationary distribution (log) + dominant non-trivial eigenvectors (0.1 ps)
edges, evals, R, occ = gt.adp_transfer_operator("T300_0p1")
its = np.array([-0.1/np.log(abs(l)) if 0 < abs(l) < 1 else np.inf for l in evals])
print("ADP top eigenvalues:", np.round(evals[:6], 4)); print("ITS (ps):", np.round(its[:6], 2))
NB = 40; cc = 0.5*(edges[:-1]+edges[1:]); occ_idx = np.where(occ)[0]; ii, jj = np.divmod(occ_idx, NB)
_, pi, _ = gt.adp_stationary("T300_0p1"); pe = pi[occ_idx]; pe = np.clip(pe, pe[pe>0].min(), None)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
s0 = ax[0].scatter(cc[ii]*180/np.pi, cc[jj]*180/np.pi, c=pe, s=14, cmap="magma", norm=LogNorm())
ax[0].set_title("EV1 stationary π (log)"); plt.colorbar(s0, ax=ax[0], fraction=.046)
for kk, idx, ttl in [(1,1,f"EV2 φ-flip (λ={evals[1]:.3f}, ITS≈{its[1]:.0f}ps)"),
                     (2,2,f"EV3 ψ (λ={evals[2]:.3f}, ITS≈{its[2]:.1f}ps)")]:
    v=R[occ_idx,idx]; m=np.abs(v).max()
    s=ax[kk].scatter(cc[ii]*180/np.pi, cc[jj]*180/np.pi, c=v, s=14, cmap="RdBu_r", vmin=-m, vmax=m)
    ax[kk].set_title(ttl); plt.colorbar(s, ax=ax[kk], fraction=.046)
for a in ax: a.set_xlabel("φ"); a.set_ylabel("ψ")
fig.suptitle("adp_300k_0p1 — 0.1 ps transfer operator (EV1 π log; EV2/EV3 right eigenvectors)", y=1.02)
plt.tight_layout(); plt.show()""")

md(r"""## 2. Eigenfunctions ↔ memberships: the PCCA+ bridge (once)

A **basis** method (here GramSchmidt) outputs the **2 non-trivial eigenfunctions** (signed). The
**3 memberships** are a *rotation* of {constant, EV₂, EV₃} into the probability simplex — the
inner-simplex map of **PCCA+**. We show this once on the triple-well; afterwards every method is
displayed as **3 memberships** (basis methods via PCCA+, membership methods natively), so the
committor correlation is a common metric.

The diagnostic below also makes the "constant is present but not a panel" point concrete: the
constant lies *in the span* of the outputs (low projection residual) but as a combination, and
mean-centering removes exactly one dimension (the near-zero 3rd singular value).""")

co("""g = harness.train_chi(tw, "gramschmidt", 0)["chi_best"]   # basis: 2 eigenfunctions
xy = tw["coords"]; evr = tw["ev_refs"]; comm = tw["committor_refs"]; names = ["p_A","p_B","p_C"]
E = plotting.align_columns(harness.eigfns(g), evr, sign=True)   # sign-aligned to EV2, EV3
M = plotting.align_columns(harness.to_memberships(g), comm)     # basin-aligned to p_A, p_B, p_C

# row 1: GramSchmidt's 2 eigenfunctions vs true EV2/EV3 (sign-aligned, RdBu_r: red=+, blue=-)
fig, ax = plt.subplots(2, 2, figsize=(9, 8))
for j in range(2):
    for row,(arr,ttl) in enumerate([(E[:,j], f"GramSchmidt EV{j+2}"), (evr[j], f"true EV{j+2}")]):
        m=np.abs(arr).max(); s=ax[row,j].scatter(xy[:,0],xy[:,1],c=arr,s=7,cmap="RdBu_r",vmin=-m,vmax=m)
        ax[row,j].set_title(f"{ttl} (|r|={abs(harness.pearson_r(E[:,j],evr[j])):.2f})",fontsize=9); plt.colorbar(s,ax=ax[row,j],fraction=.046)
fig.suptitle("GramSchmidt learns the 2 non-trivial eigenfunctions (top) ≈ true EV2/EV3 (bottom)",y=1.0)
plt.tight_layout(); plt.show()

# diagnostic: constant in span, centered singular spectrum
Q,_=np.linalg.qr(g); o=np.ones(len(g)); resid=np.linalg.norm(o-Q@(Q.T@o))/np.linalg.norm(o)
sv=np.linalg.svd(g-g.mean(0),compute_uv=False)
print(f"constant-in-span residual of the 2 outputs: {resid:.3f} (low = constant present as a combination)")
print(f"mean-centered singular values: {np.round(sv/sv[0],3)} (2 non-trivial dims; 3rd ≈0 = the constant)")

# row 2: PCCA+ memberships vs committors (basin-aligned columns, red=1 / blue=0)
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for j in range(3):
    for row,(arr,ttl) in enumerate([(M[:,j], f"PCCA+ χ ({names[j]})"), (comm[j], f"committor {names[j]}")]):
        s=ax[row,j].scatter(xy[:,0],xy[:,1],c=arr,s=7,cmap="RdBu_r",vmin=0,vmax=1)
        ax[row,j].set_title(ttl,fontsize=9); plt.colorbar(s,ax=ax[row,j],fraction=.046)
mr,_=harness.shape_r(M, comm)
fig.suptitle(f"PCCA+ rotates the eigenfunctions into 3 memberships (top) ≈ committors (bottom)  |r|={mr:.3f}",y=1.0)
plt.tight_layout(); plt.show()""")

md(r"""## 3. Why a softmax head for the membership family

The linear ISA target collapses on 231-D ADP (amplitude → 0 and it mode-selects onto φ, missing
ψ); the historical fix was a converged 1-D ShiftScale warm-up — a band-aid. The **softmax head**
makes ISA produce a proper membership that *cannot* amplitude-collapse, recovering **both** φ and ψ
from scratch. Below: ADP ISA in three forms across seeds.""")

co("""forms = [("linear (no warm-up)", dict(head="linear", warmup=False)),
         ("linear + 1-D warm-up",  dict(head="linear", warmup=True)),
         ("softmax (default)",      dict(head=None,     warmup=False))]
form_runs = {}; rows = []
for name, kw in forms:
    runs = {s: harness.train_chi(adp, "isa", s, verbose=False, **kw) for s in SEEDS}
    form_runs[name] = runs
    er = [harness.eig_r(runs[s]["chi_best"], adp["ev_refs"])[0] for s in SEEDS]
    ke = [harness.k_eff(harness.to_memberships(runs[s]["chi_best"])) for s in SEEDS]
    row = {f"seed{s}": f"{er[s]:.2f}/k{ke[s]}" for s in SEEDS}
    row["mean±sd eig_r"] = f"{np.mean(er):.3f}±{np.std(er):.3f}"
    rows.append(dict(form=name, **row))
print("full 5-seed comparison — cell = eig_r / k_eff per seed:")
display(pd.DataFrame(rows).set_index("form"))""")

co("""# the 1-D warm-up convergence (the band-aid the softmax head removes)
fig = plotting.plot_warmup(adp, SEEDS); plt.show()
# ALL 5 seeds per form (portrays the failure modes the table reports: linear's high variance,
# warm-up's φ-only mode-selection, softmax's consistent φ+ψ). Columns basin-aligned, red=1/blue=0.
for name in [f[0] for f in forms]:
    plotting.chi_maps_grid(adp, form_runs[name], name, view="membership"); plt.show()""")

md("""## 4. Train / load the ensemble

Cached per `(system, variant, seed)`; loads instantly if present, else trains. Parallel build:
`python run_ensemble.py all`.""")

co("""RUNS = {}
for tag, sysd in SYS.items():
    for v in REV_VARIANTS:
        RUNS[(tag, v)] = {s: harness.train_chi(sysd, v, s, verbose=False) for s in SEEDS}
print("loaded", len(RUNS), "method×system run-sets")""")

md("""## 5. Benchmark summary — eigenfunction recovery + membership/committor recovery, over seeds

`eig_r` = the 2 non-trivial eigenfunctions vs EV₂/EV₃ (uniform metric, both systems). `memb_r` =
the 3 memberships vs committors (triple-well). `k_eff` = live memberships. mean ± SD over 5 seeds.

**`eig_r` is computed on the eigenfunctions directly** (mean-center + SVD of the output), with **no
PCCA+ involved** — so the ADP `eig_r` isolates the *method's* eigenfunction quality from the
membership conversion. The basis methods' lower ADP `eig_r` is therefore the method itself (they
recover φ but struggle with the fast ψ at the correct k=2), not a PCCA+ artifact.""")

co("""for tag, sysd in SYS.items():
    print(f"\\n=== {tag} ===")
    display(plotting.score_frame(sysd, RUNS, REV_VARIANTS))""")

md("""## 6. Membership maps for every method and seed

All methods shown as the **3 memberships** (basis methods via PCCA+, membership methods natively).
triple-well on (x,y), alanine dipeptide on (φ,ψ). Compare to the committors / EV maps in §1.""")

for tag in ["triple_well", "adp_300k_0p1"]:
    md(f"### {tag}")
    co(f"""for v in REV_VARIANTS:
    fig = plotting.chi_maps_grid(SYS["{tag}"], RUNS[("{tag}", v)], v, view="membership"); plt.show()""")

md("""## 7. Loss curves — held-out vs train

Method-agnostic ISOKANN self-consistency residual ‖χ − GramSchmidt(E[χ(x_τ)])‖² on train (blue)
and held-out (red), all seeds overlaid. (SVD-Power shows the power-iteration MSE, full-data, no
held-out; VAMP shows −VAMP-2.)""")

for tag in ["triple_well", "adp_300k_0p1"]:
    md(f"### {tag}")
    co(f"""fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, v in zip(axes.ravel(), REV_VARIANTS):
    plotting.loss_curves(RUNS[("{tag}", v)], v, ax=ax)
fig.suptitle("{tag} — train (blue) vs held-out (red) loss", y=1.01); plt.tight_layout(); plt.show()""")

md(r"""## 8. Reading the results

- **Ground truth (§1)** sets the bar: TW committors, ADP φ-flip (EV₂) / ψ (EV₃).
- **Bridge (§2)**: a basis method gives the 2 eigenfunctions; PCCA+ rotates {const,EV₂,EV₃} into
  the 3 memberships ≈ committors. The constant is in the span (low residual) but isn't a panel.
- **Why softmax (§3)**: softmax-ISA recovers φ *and* ψ from scratch; linear-ISA collapses and even
  the warm-up band-aid only mode-selects φ. This is why the membership family uses a softmax head
  and no warm-up.
- **Tables (§5)**: `eig_r` (subspace recovery, both systems) with small SD = accurate *and* stable;
  `memb_r` (TW committors) is the membership-quality metric.
- **Maps (§6)** show the learned memberships per method/seed; **loss curves (§7)** show held-out
  generalisation of the learned invariance.""")

nb = new_notebook(); nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reversible.ipynb")
nbf.write(nb, out); print("wrote", out, "with", len(C), "cells")
