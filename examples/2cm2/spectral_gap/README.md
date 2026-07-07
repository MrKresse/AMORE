# 2cm2 — multi-dimensional softmax-ISA ISOKANN

Multi-dimensional ISOKANN / AMORE on the **2cm2** protein–ligand MD trajectory.  One
directory and one notebook per membership dimension **k = 3, 4, 5, 6**; each trains a
`k`-way softmax-ISA membership model, inspects the membership **simplex** and its edges,
and exports zeroth-order transition pathways.

## Layout

```
2cm2/
├── lib/
│   ├── data.py       features: per-residue COM pairwise distances (PBC), trajectory-lag pairs
│   ├── train.py      k-dim softmax-ISA training (ChiNetMulti + ISA isotarget)
│   ├── analysis.py   simplex edges, rare-edge detection, pathway export, plotting, chi-UMAP
│   └── run.py        cache-backed get_model(k)
├── build.py          regenerates the four notebooks
├── dim3/  2cm2_isokann_dim3.ipynb   (+ pathways/, figures)
├── dim4/  2cm2_isokann_dim4.ipynb
├── dim5/  2cm2_isokann_dim5.ipynb
└── dim6/  2cm2_isokann_dim6.ipynb
```

Heavy artefacts (COMs, features, trained models) are cached under
`/scratch/htc/$USER/2cm2/` (override with `CM2_SCRATCH`).  Notebooks load the cache, so
re-execution is fast; delete the relevant cache file to recompute.

## Pipeline

1. **Features** (`lib/data.py`) — reproduces Fazil's `compute_coords_350_2000.ipynb`:
   `select_atoms("protein and not element H")`, centre of mass **per residue**
   (285 residues), then pairwise distances over all C(285,2)=40470 residue pairs with the
   **minimum-image convention** (periodic boundary conditions, static 81 Å cubic box).
   Koopman pairs are formed **along the trajectory**: anchors = frames [350, 2000), lag
   images = frames shifted by `CM2_LAG` (default 1).

   > **PBC note.** The protein is whole (max COM-pair distance 64 Å < box 81 Å) but larger
   > than half the box (40.5 Å), so MIC wraps ~8% of the pairs.  `use_pbc=True` reproduces
   > the reference pipeline; `build_features(use_pbc=False)` gives raw distances (raw COMs
   > are cached separately, so switching is a 1-second recompute).

2. **Training** (`lib/train.py`) — `k` softmax memberships via the **ISA isotarget, no
   warm-up** (`amore.isokann.ChiNetMulti`), with the `ptb1b_isokann_500_2` hyperparameters:
   hidden `[4096, 512, 64]`, lr 5e-4, weight_decay 1e-8, batch 128.  Features are z-scored
   with the anchor statistics.  Each outer power iteration recomputes the ISA target and
   runs a short minibatch-SGD inner loop; the best validation (Gram-Schmidt residual) chi is
   kept.

3. **Diagnostics** — loss curves and the memberships χ_i along the trajectory.
   k=3 is drawn as the barycentric **simplex triangle**; k>3 is laid out with **CUMAP**
   (`amore.scrna.plotting.run_chi_umap`, cross-entropy layout on the chi-simplex affinity).

4. **Simplex edges** (`lib/analysis.py`) — the edge coordinate
   s_ij = ½(χ_i − χ_j + 1).  Per edge we count on-edge frames (χ_i+χ_j ≥ 0.8) and genuine
   **transition** frames (intermediate s_ij).  Edges between two visited states with few
   transition frames are the **rare** simplex edges.

5. **Zeroth-order pathways** — for every relevant edge, the on-edge frames ordered by s_ij
   (vertex j → vertex i) are written to `dimK/pathways/pathway_edge_i-j.dcd` (+ `.pdb`
   topology).  No interpolation — a reordering of existing frames.

## Reproduce

```bash
python build.py                                   # (re)write the four notebooks
jupyter nbconvert --to notebook --execute --inplace dim3/2cm2_isokann_dim3.ipynb
# ... dim4, dim5, dim6
```

Training all four models directly: `python lib/run.py` is not a CLI, but
`python -c "import sys; sys.path.insert(0,'lib'); import run; [run.get_model(k) for k in (3,4,5,6)]"`.
