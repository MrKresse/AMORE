# -*- coding: utf-8 -*-
"""Emit the 2cm2 ISOKANN notebook for dim3, at CM2_LAG=20.

Run:  python build.py            # writes dim3/2cm2_isokann_dim3.ipynb
Then: jupyter nbconvert --to notebook --execute --inplace dim3/2cm2_isokann_dim3.ipynb

Only dim3 is kept: `spectral_gap/spectral_gap.ipynb`'s inverse-PCCA+ analysis found no
clean spectral gap at the original lag=1 (every eigenvalue compressed near 1, unstable
across k=3..6), but at lag=20 the leading cluster separates into distinct
Perron/complex-pair/real processes matching a k=3 real invariant structure (Perron + one
complex-conjugate pair needs exactly 3 real dimensions, not bisected -- see that
notebook's corrected conclusion). dim4/dim5/dim6 (all lag=1) were dropped as superseded
by that finding.

Heavy machinery lives in lib/ (data.py, train.py, analysis.py, run.py); the notebook is a
thin driver.
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = [3]
LAG = 20


def build_one(k: int):
    nb = nbf.v4.new_notebook()
    cells = []
    def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))

    md(rf"""# 2cm2 — {k}-dimensional softmax-ISA ISOKANN (lag={LAG} frames)

A **k={k}** ISOKANN model maps each configuration to the membership simplex
$\Delta^{{{k-1}}}$: $\chi_i\ge0,\ \sum_i\chi_i=1$.  The {k} vertices are the metastable
states and the $\binom{{{k}}}{{2}}={k*(k-1)//2}$ **edges** are the pairwise interconversions.

**Lag={LAG}, not the original lag=1**: `../spectral_gap/spectral_gap.ipynb` found lag=1
too short to separate any process from noise (`inverse_pcca`'s recovered spectrum sat
compressed in `[0.985, 1.0]` for every k, with an unstable leading timescale across k). At
lag={LAG} the spectrum resolves into a Perron mode plus a genuine complex-conjugate pair
-- a real invariant structure needing exactly 3 real dimensions (Perron + 2 for the pair),
i.e. k=3 without bisecting the pair. That's the k trained below.

**Pipeline**
1. Features — per-residue centre-of-mass pairwise distances (PBC / minimum image), computed
   from the trajectory as in `compute_coords_350_2000.ipynb`, paired along the trajectory
   with lag={LAG} frames (see `data.LAG`, printed below).
2. Training — {k} softmax memberships via the **ISA isotarget, no warm-up**
   (`amore.isokann.ChiNetMulti`), with the `ptb1b_isokann_500_2` hyperparameters
   (hidden `[4096, 512, 64]`, lr 5e-4, weight_decay 1e-8, batch 128).
3. Diagnostics — loss curves and the memberships $\chi_i$ along the trajectory.
4. Simplex edges — per-edge transition population and detection of **rare** edges.
5. Zeroth-order pathways — for every relevant edge, the on-edge frames **ordered by the edge
   coordinate** $s_{{ij}}=\tfrac12(\chi_i-\chi_j+1)$, exported as DCD (+ PDB topology).
6. Metastable-state renders — one PyMOL PNG per state (representative conformation + ligand,
   with an aligned ensemble overlay).
7. Per-edge transition sessions — a playable PyMOL `.pse` per relevant edge (the ordered
   pathway + ligand transition sweep).""")

    code(f"""import os, sys
import numpy as np
import matplotlib.pyplot as plt

LIB = os.path.abspath(os.path.join("..", "lib"))
sys.path.insert(0, LIB)
import data, analysis as A, run

data.LAG = {LAG}   # data.py / run.py both read data.LAG dynamically at call time, so this
                   # repoints build_features()'s and model_cache()'s cache paths together.
K = {k}
plt.rcParams["figure.dpi"] = 120
print("selection:", data.SELECTION, "| frames", data.NSTART, "..", data.NEND,
      "| lag", data.LAG)""")

    md("## 1–2. Features + training (cached per k)\n"
       "Features are shared across all dimensions; training is cached to scratch, so this "
       "cell is instant on re-run.  Delete the cache file to retrain.")
    code("""m = run.get_model(K, verbose=True)
chi = m["chi"]                      # (N, k) memberships on the anchor frames
NSTART = m["nstart"]
print("chi", chi.shape, "| per-membership std:", chi.std(0).round(3),
      "| k_eff:", int((chi.std(0) > 0.05).sum()))""")

    md("## 3. Loss curves and memberships along the trajectory")
    code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
A.plot_loss(m, ax=ax[0])
A.plot_chi_trajectory(chi, NSTART, ax=ax[1])
plt.tight_layout(); plt.savefig("loss_and_chi.png", dpi=150, bbox_inches="tight"); plt.show()""")

    if k == 3:
        md("The k=3 memberships visualise directly as the barycentric **simplex triangle** "
           "(each vertex = one metastable state).")
        code("""fig, ax = plt.subplots(figsize=(5, 5))
A.plot_simplex_triangle(chi, ax=ax)
plt.savefig("simplex_triangle.png", dpi=150, bbox_inches="tight"); plt.show()""")
    else:
        md("For k>3 the simplex cannot be drawn as a triangle, so we lay the memberships out "
           "with **IMAP** (Invariant Manifold Approximation and Projection) — "
           "`amore.scrna.plotting.imap_sgd`: direct SGD on the cross-entropy between the "
           "Bhattacharyya chi-affinity and the low-dim UMAP kernel, with **no kNN graph and no "
           "bandwidth calibration** (fewer hyperparameters than UMAP/CUMAP, and it keeps the "
           "states connected rather than fragmenting into islands).  The first panel colours "
           "the embedding by dominant state; the rest colour it by each membership $\\chi_i$.")
        code("""fig, emb = A.plot_chi_umap(chi, seed=0)
plt.savefig("chi_umap.png", dpi=150, bbox_inches="tight"); plt.show()""")

    md("## 4. Simplex edges — transition population and rare edges\n"
       "For each edge $(i,j)$: `on_edge` counts frames with $\\chi_i+\\chi_j\\ge0.8$, "
       "`transition` counts the subset with intermediate $s_{ij}\\in[0.15,0.85]$ (genuine "
       "transition-state frames).  An edge is **relevant** when both endpoint states are "
       "visited, and **rare** when its transition count is far below the median relevant edge.")
    code("""rows = A.edge_table(chi)
rare = A.rare_edges(rows)
print(A.format_edge_table(rows, rare))
print("\\nrare simplex edges:", rare if rare else "none")

fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(rows)), 4))
A.plot_edge_populations(rows, rare, ax=ax)
plt.tight_layout(); plt.savefig("edge_populations.png", dpi=150, bbox_inches="tight"); plt.show()""")

    md("## 4b. Coarse-grained rate matrix (a principled cross-check on the edge table)\n"
       "The edge table above flags rare edges by **counting on-edge frames** -- a kinematic "
       "heuristic. `amore.inverse_pcca.rate_matrix` gives a principled alternative: "
       "`Q = logm(Lambda_S) / tau`, the continuous-time generator of the coarse propagator "
       "`Lambda_S` (`inverse_pcca.py`'s own `G_hat^{-1} C_hat` Rayleigh-Ritz projection). "
       "Off-diagonal `Q[i,j]` is the coarse transition RATE `i -> j`; unlike frame-counting "
       "it comes with a timescale unit and needs no arbitrary on-edge/transition thresholds. "
       "See `examples/isokann_benchmark/inverse_pcca.ipynb`'s own rate-matrix section for "
       "the validation of this against a numerical reference on the (synthetic) benchmark "
       "systems -- 2cm2 has no such ground truth, so this is a diagnostic, not a validation.")
    code("""import torch
SRC = os.path.abspath(os.path.join("..", "..", "..", "src"))
sys.path.insert(0, SRC)
import train
from amore.isokann import ChiNetMulti
from amore.inverse_pcca import inverse_pcca, rate_matrix, plot_rate_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feats_q = data.build_features(use_pbc=True, verbose=False)
D0n, Dtn, _, _ = train.normalise(feats_q["D0"], feats_q["Dt"])
D0n, Dtn = D0n.to(DEVICE), Dtn.to(DEVICE)
Nq, Fq = D0n.shape; Kbq = Dtn.shape[1]

net_q = ChiNetMulti(Fq, K, hidden=m["hidden"]).to(DEVICE)
net_q.load_state_dict(m["net_state"]); net_q.eval()

def propagate_q():
    with torch.no_grad():
        flat = Dtn.reshape(Nq * Kbq, Fq)
        return net_q(flat).reshape(Nq, Kbq, -1).mean(1).cpu().numpy()

try:
    rate_result = inverse_pcca(chi, propagate_q, float(data.LAG), reversible=True)
except ValueError as e:
    print(f"reversible=True failed ({e})\\n-> falling back to reversible=False (Schur route): "
          "a single un-resampled trajectory (N_REP=1) need not satisfy empirical detailed "
          "balance -- rate_matrix itself needs no eigendecomposition either way, only Lambda_S.")
    rate_result = inverse_pcca(chi, propagate_q, float(data.LAG), reversible=False)
Q = rate_matrix(rate_result.Lambda_S, float(data.LAG))
print(f"Lambda_S row sums: {rate_result.Lambda_S.sum(1).round(4)}")
print(f"invariance residual (RMS/entry): {rate_result.residual/np.sqrt(chi.size):.4f}")
print("coarse rate matrix Q (per frame):")
print(Q.round(5))
print("row sums (should be 0):", Q.sum(1).round(8))

edge_rank_frames = sorted(((r["transition"], r["edge"]) for r in rows if r["relevant"]), reverse=True)
edge_rank_rate = sorted(((abs(Q[i, j]) + abs(Q[j, i]), (i, j))
                         for i in range(K) for j in range(i + 1, K)), reverse=True)
print("\\nedge ranking by transition-frame count:", [e for _, e in edge_rank_frames])
print("edge ranking by |rate|:              ", [e for _, e in edge_rank_rate])""")

    code("""fig, ax = plt.subplots(figsize=(4.5, 4.5))
plot_rate_matrix(ax, Q, title=f"2cm2 dim{K}: coarse rate matrix Q")
plt.tight_layout(); plt.savefig("rate_matrix.png", dpi=150, bbox_inches="tight"); plt.show()""")

    md("## 5. Zeroth-order transition pathways\n"
       "For every **relevant** edge, take the on-edge frames, order them by the edge "
       "coordinate $s_{ij}$ (vertex $j\\to$ vertex $i$), and write that reordered frame "
       "sequence to a DCD (+ PDB topology) under `pathways/`.  This is a zeroth-order "
       "approximation of the transition pathway — a reordering of existing frames, no "
       "interpolation.  Rare edges are exported too (flagged in the printout).")
    code("""pdb, dcd = data.pdb_path(), data.dcd_path()
outdir = os.path.abspath("pathways")
os.makedirs(outdir, exist_ok=True)
summary = []
for r in rows:
    if not r["relevant"]:
        continue
    i, j = r["edge"]
    order, s = A.pathway_frames(chi, i, j)
    if len(order) == 0:
        continue
    stem = os.path.join(outdir, f"pathway_edge_{i}-{j}")
    n = A.export_pathway(order, NSTART, pdb, dcd, stem + ".dcd", stem + ".pdb")
    tag = "RARE" if (i, j) in rare else ""
    summary.append((f"{i}-{j}", n, round(float(s.min()), 3), round(float(s.max()), 3), tag))
    print(f"edge {i}-{j}: {n:4d} frames  s in [{s.min():.3f}, {s.max():.3f}]  "
          f"-> {os.path.basename(stem)}.dcd  {tag}")
print("\\nwrote", len(summary), "pathway DCDs to", outdir)""")

    md("Overlay of the exported pathways in the edge coordinate — each curve is one edge's "
       "on-edge frames sorted by $s_{ij}$, coloured by whether it is a rare edge.")
    code("""fig, ax = plt.subplots(figsize=(7, 4))
for r in rows:
    if not r["relevant"]:
        continue
    i, j = r["edge"]
    order, s = A.pathway_frames(chi, i, j)
    if len(order) == 0:
        continue
    c = "crimson" if (i, j) in rare else "steelblue"
    ax.plot(np.linspace(0, 1, len(s)), s, color=c, alpha=0.8,
            label=f"{i}-{j}" + (" (rare)" if (i, j) in rare else ""))
ax.set_xlabel("ordered pathway position"); ax.set_ylabel("edge coordinate $s_{ij}$")
ax.set_title("zeroth-order pathways (frames ordered along each edge)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("pathways_edgecoord.png", dpi=150, bbox_inches="tight"); plt.show()""")

    md("## 6. Metastable-state renders (PyMOL)\n"
       "One **4K PNG per state**, zoomed on the ligand.  The representative pose (frame with "
       "$\\chi_i$ closest to 0.95) is drawn as **one fat orange KB8**, with the other "
       "$\\chi_i\\in[0.9,1.0]$ poses — superposed **on the ligand** — as a faint pale-orange "
       "**halo** showing its conformational spread.  Protein residues within **5 Å** of the "
       "ligand are thin grey sticks, on a **see-through grey cartoon** (chopped, so nothing "
       "occludes the ligand).  Water/ions removed, white background.  Each state also gets an "
       "editable **`.pse`** you can open in a local PyMOL and rotate.  Rendered headless via "
       "the open-source PyMOL micromamba env; cached under `states/` (delete to re-render).")
    code("""import render
from IPython.display import Image, display
# render.py's own cache check only looks at file existence, not content provenance -- if
# you change data.LAG/K and states/edges/ still holds a PREVIOUS config's renders, pass
# force=True here once to refresh them (see build.py's own note where this is set).
infos, state_pngs = render.render_model_states(K, verbose=True)
for i, p in enumerate(state_pngs):
    print(f"state {i}: {os.path.basename(p)}  (+ {os.path.basename(p)[:-4]}.pse)")
    display(Image(filename=p, width=560))""")

    md("## 7. Per-edge transition-pathway sessions (PyMOL `.pse`)\n"
       "For every **relevant** simplex edge, an editable **`.pse`** (+ 4K preview PNG) under "
       "`edges/`, in the same aesthetic as the state renders.  Each holds the zeroth-order "
       "pathway (on-edge frames ordered by $s_{ij}$, superposed on protein C$\\alpha$) as a "
       "**playable multi-state object** — open the `.pse` in a local PyMOL and scrub the states "
       "to walk the transition $j\\to i$ — plus the whole ligand **transition sweep** as a "
       "faint pale-orange halo over the see-through cartoon and 5 Å pocket sticks.")
    code("""edge_infos = render.build_edge_pses(K, verbose=True)
for info in edge_infos:
    i, j = info["edge"]
    png = os.path.join(os.path.dirname(info["pse"]), f"edge_{i}-{j}.png")
    tag = " (RARE)" if info["rare"] else ""
    print(f"edge {i}-{j}{tag}: {info['n']} frames, s {info['s_lo']:.2f}->{info['s_hi']:.2f}"
          f"  -> {os.path.basename(info['pse'])}")
    if os.path.exists(png):
        display(Image(filename=png, width=560))""")

    md(f"""### Summary

- **k = {k}** softmax-ISA memberships trained on the 2cm2 trajectory anchor frames
  (per-residue COM pairwise-distance features, PBC).
- The edge table above lists every simplex edge; **rare** edges (few transition frames
  between two visited states) are flagged in red.
- Each relevant edge's zeroth-order pathway is saved under `pathways/` as a DCD + PDB pair,
  with frames ordered along the edge coordinate.""")

    nb["cells"] = cells
    return nb


def main():
    for k in DIMS:
        d = os.path.join(HERE, f"dim{k}")
        os.makedirs(d, exist_ok=True)
        nb = build_one(k)
        path = os.path.join(d, f"2cm2_isokann_dim{k}.ipynb")
        with open(path, "w") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
