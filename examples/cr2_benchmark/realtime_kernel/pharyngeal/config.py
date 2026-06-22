"""
Shared, single-source-of-truth configuration for the ISOKANN+AMORE vs CellRank2
benchmark on the **pharyngeal endoderm** dataset (CellRank 2, Nature Methods 2024,
Figure 2 — the "subsetted data" RealTimeKernel analysis).

Every value below that is marked PINNED is read verbatim from the CR2
reproducibility repo:
    cellrank2_reproducibility/scripts/realtime_kernel/pharyngeal_endoderm/
        realtimekernel_subsetted_data.py

Values marked RUNTIME are data-dependent (e.g. the number of HVGs selected by
scanpy's default dispersion cutoffs, or the cell count after subsetting). They
are NOT guessed here: they are written into `artifacts/cr2_config_resolved.json`
by 01_cr2_reproduce.py after the actual CR2 run, and both arms read them back
from there. This guarantees the ISOKANN arm and the CR2 arm consume identical
preprocessing.

Why a single config: the task requires the plug-in arm (Arm K) to use *exactly*
the kernel/features CR2 uses. Centralising the pinned values here and forcing
both pipelines to import them is the mechanism that enforces "do not guess,
read from repo".
"""

from __future__ import annotations
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
# cellrank2_reproducibility is shared by both realtime_kernel benchmarks; it is kept
# two levels up at cr2_benchmark/ (this config now lives in realtime_kernel/pharyngeal/).
CR2_REPO     = os.path.join(HERE, "..", "..", "cellrank2_reproducibility")
CR2_DATA     = os.path.join(CR2_REPO, "data")
PHARYNX_RAW  = os.path.join(CR2_DATA, "pharyngeal_endoderm", "raw", "adata_pharynx.h5ad")
TMAPS_DIR    = os.path.join(CR2_DATA, "pharyngeal_endoderm", "tmaps_subsetted_data")
ARTIFACTS    = os.path.join(HERE, "artifacts")
FIGURES      = os.path.join(HERE, "figures")
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Resolved (runtime) config written by 01_cr2_reproduce.py and read by everyone.
RESOLVED_JSON = os.path.join(ARTIFACTS, "cr2_config_resolved.json")

# =============================================================================
# PINNED — subset definition
# =============================================================================
# adata = adata[adata.obs["cluster_fine"].isin([...])]
# cluster_fine = cluster_data.csv["res.1"]  (fine Seurat clustering).
SUBSET_CLUSTER_KEY    = "cluster_fine"
SUBSET_CLUSTERS       = ["2", "4", "9", "12", "25", "26"]   # PINNED

# =============================================================================
# PINNED — preprocessing (scanpy defaults exactly as called in the notebook)
# =============================================================================
# sc.pp.highly_variable_genes(adata)        -> flavor='seurat', dispersion cutoffs
#                                              (min_mean=0.0125, max_mean=3, min_disp=0.5),
#                                              n_top_genes=None. HVG COUNT is RUNTIME.
# sc.tl.pca(adata)                          -> n_comps=50 (scanpy default), on HVGs
# sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
HVG_FLAVOR            = "seurat"     # PINNED (scanpy default)
HVG_KWARGS            = {}           # PINNED: highly_variable_genes called with NO args
PCA_N_COMPS           = 50           # PINNED (scanpy sc.tl.pca default)
NEIGHBORS_N_PCS       = 30           # PINNED
NEIGHBORS_N_NEIGHBORS = 30           # PINNED

# Features fed to ISOKANN: the 50 PCs on HVGs (== adata.obsm["X_pca"][:, :50]).
ISOKANN_N_PCS         = 50           # task: "50 PCs computed on HVGs"

# =============================================================================
# PINNED — RealTimeKernel (the EXACT kernel CR2 uses; Arm K plug-in expectation)
# =============================================================================
TIME_KEY              = "day"        # PINNED (adata.obs["day"] from day_str)
# wot.ot.OTModel(adata).compute_all_transport_maps(...)  -> WOT defaults
# rtk = RealTimeKernel.from_wot(adata, path=TMAPS_DIR, time_key="day")
RTK_GROWTH_ITERS      = 3            # PINNED
RTK_GROWTH_RATE_KEY   = "growth_rate_init"   # PINNED (must exist in adata.obs)
RTK_SELF_TRANSITIONS  = "all"        # PINNED
RTK_CONN_WEIGHT        = 0.1         # PINNED

# =============================================================================
# PINNED — GPCCA estimator
# =============================================================================
SCHUR_N_COMPONENTS    = 5            # PINNED  estimator.compute_schur(n_components=5)
N_MACROSTATES         = 4            # PINNED  estimator.compute_macrostates(n_states=4)
MACROSTATE_CLUSTER_KEY = "cluster_name"      # PINNED
TERMINAL_STATES       = ["parathyroid", "cTEC", "mTEC", "ubb"]  # PINNED set_terminal_states(...)
# TSI: estimator.tsi(n_macrostates=10, terminal_states=TERMINAL_STATES, cluster_key="cluster_name")
TSI_N_MACROSTATES     = 10           # PINNED
# Fate probabilities: estimator.compute_fate_probabilities()  (use_petsc handled at runtime)

# ISOKANN membership dimension == CR2 macrostate count (task: "k = CR2 macrostate count").
K_CHI                 = N_MACROSTATES

# =============================================================================
# PINNED — driver / TF-recovery benchmark
# =============================================================================
# estimator.compute_lineage_drivers(lineages=["mTEC"], clusters=["2","9"],
#                                   cluster_key="cluster_fine")
DRIVER_LINEAGE        = "mTEC"       # PINNED
DRIVER_CLUSTERS       = ["2", "9"]   # PINNED (cluster_fine progenitor clusters of mTEC)
DRIVER_CLUSTER_KEY    = "cluster_fine"

# Curated mTEC marker genes (ground-truth driver list) — PINNED verbatim.
MTEC_GENES = [
    "Cldn3", "Cldn4", "Notch1", "Krt5", "H2-Aa", "H2-Ab1", "H2-Eb1",
    "Grhl3", "Grhl1", "Elf5", "Irf6", "Sox9", "Upk2", "Ovol1", "Hes1",
    "Rhov", "Pvrl4", "Klf5", "Egr1", "Sfn", "Perp", "Fxyd3", "Hspb1",
    "Krt5", "S100a11",
]

# Genes excluded from driver ranking — PINNED.
GENE_EXCLUDE_PREFIXES = ("mt.", "Rpl", "Rps", "^Hb[^(p)]")   # str.startswith patterns
MOUSE_TFS_TSV         = os.path.join(CR2_DATA, "generic", "mouse_tfs.tsv")  # CR2's own (absent on figshare)
# Canonical mouse TF list (Aerts-lab cisTarget) used as the TF universe. Applied
# IDENTICALLY to both the CR2 and ISOKANN rankings, so the head-to-head is fair
# regardless of any difference from CR2's exact unpublished mouse_tfs.tsv.
MOUSE_TFS_LOCAL       = os.path.join(CR2_DATA, "generic", "allTFs_mm.txt")

# Cell-cycle gene lists used to filter drivers — PINNED verbatim (Tirosh et al.).
S_GENES = [
    "Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1", "Ung", "Gins2",
    "Mcm6", "Cdca7", "Dtl", "Prim1", "Uhrf1", "Mlf1ip", "Hells", "Rfc2",
    "Rpa2", "Nasp", "Rad51ap1", "Gmnn", "Wdr76", "Slbp", "Ccne2", "Ubr7",
    "Pold3", "Msh2", "Atad2", "Rad51", "Rrm2", "Cdc45", "Cdc6", "Exo1",
    "Tipin", "Dscc1", "Blm", "Casp8ap2", "Usp1", "Clspn", "Pola1", "Chaf1b",
    "Brip1", "E2f8",
]
G2M_GENES = [
    "Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80",
    "Cks2", "Nuf2", "Cks1b", "Mki67", "Tmpo", "Cenpf", "Tacc3", "Fam64a",
    "Smc4", "Ccnb2", "Ckap2l", "Ckap2", "Aurkb", "Bub1", "Kif11", "Anp32e",
    "Tubb4b", "Gtse1", "Kif20b", "Hjurp", "Cdca3", "Hn1", "Cdc20", "Ttk",
    "Cdc25c", "Kif2c", "Rangap1", "Ncapd2", "Dlgap5", "Cdca2", "Cdca8",
    "Ect2", "Kif23", "Hmmr", "Aurka", "Psrc1", "Anln", "Lbr", "Ckap5",
    "Cenpe", "Ctcf", "Nek2", "G2e3", "Gas2l3", "Cbx5", "Cenpa",
]

# Driver-recovery scoring: count curated/TF genes within top-`threshold` of the
# correlation ranking (CR2 uses threshold=100 in get_var_ranks). We report
# overlap@k for several k for a directly comparable curve.
DRIVER_TOPK           = [10, 20, 50, 100]   # overlap@k cutoffs

# =============================================================================
# Gradient-based attribution (the ISOKANN-native driver method) — benchmark spec
# =============================================================================
# Per-lineage, per-cell, signed attribution of gene -> membership:
#     d chi_i / d gene  =  (d chi_i / d PC) @ pca_loadings        (exact autograd)
# computed independently for EACH chi_i (i.e. per cell line), so direction is
# intrinsic: sign>0 = gene pushes a cell toward lineage i, sign<0 = away.
ATTR_METHOD           = "dchi/dgene via chain rule through PCA loadings (per cell, signed)"
# Ranking magnitude: MAX over cells of |d chi_i/d gene| (peak at the commitment
# boundary; mean dilutes it). Direction for the driver list: sign of the mean
# signed gradient (equivalently sign of corr(gene, chi_i)).
ATTR_RANK_AGG         = "max_over_cells |dchi/dgene|"
ATTR_DIRECTION        = "sign(mean_cell dchi/dgene)  ==  sign(corr(gene, chi_i))"
ATTR_GENE_UNIVERSE    = "HVGs minus cell-cycle(S/G2M) and mt./Rpl/Rps/Hb (CR2 filter)"
ATTR_HEATMAP_GRID     = 100        # chi grid points (0->1), locally Gaussian-averaged
ATTR_HEATMAP_BW       = 0.03       # gaussian bandwidth in chi units

# =============================================================================
# CBC (decision-boundary) benchmark
# =============================================================================
# CR2 (cbc_ptk_vs_vk.py): kernel.cbc(source, target, cluster_key, rep);
# kernels compared via log((cbc_A+1)/(cbc_B+1)), one-sided t-test vs 0.
# Boundaries here: progenitors -> each terminal state (RUNTIME-checked that the
# progenitor label exists in cluster_name). Aggregation matches CR2 (per-cell
# CBC distribution per boundary).
CBC_CLUSTER_KEY       = "cluster_name"
CBC_REP               = "X_pca"
# STATE_TRANSITIONS resolved at runtime from the macrostate->cluster mapping.

# =============================================================================
# ISOKANN training (Arm K) — schedule fixed by the task
# =============================================================================
ISOKANN_HIDDEN        = [128, 64, 32]   # ChiNetMultiLinear hidden widths (Tanh, linear out)
WARMUP_TARGET         = "shiftscale"    # task: 1D warm-up with ShiftScale isotarget
MAIN_TARGET           = "isa"           # task: switch to ISA for k-D simplex memberships
WARMUP_ITERS          = 150             # outer iters of 1D ShiftScale warm-up
MAIN_ITERS            = 600             # outer iters of k-D ISA main loop
EPOCHS_PER_ITER       = 50              # inner SGD epochs per outer iter
LR                    = 1e-3
LR_DECAY              = 0.999
GRAD_CLIP             = 5.0
BATCH                 = 4096
PAIR_MONITOR_FRAC     = 0.10            # monitoring-only holdout of transition entries
SEED                  = 0

# SD live-mode threshold (feedback memory: SD<0.05 borderline, >0.05 live).
SD_LIVE_THRESHOLD     = 0.05
