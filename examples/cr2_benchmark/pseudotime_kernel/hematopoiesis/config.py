"""
Shared, single-source-of-truth configuration for the ISOKANN+AMORE vs CellRank2
benchmark on the **NeurIPS 2021 human bone-marrow hematopoiesis** dataset
(Luecken et al. 2021; CellRank 2, Nature Methods 2024 — the PseudotimeKernel
"hematopoiesis" analysis).

Every value below marked PINNED is read verbatim from the CR2 reproducibility repo:
    cellrank2_reproducibility/scripts/pseudotime_kernel/hematopoiesis/
        dpt.py            (cell-type subset, DPT root, PseudotimeKernel, GPCCA,
                           macrostates / terminal states / TSI / fate / drivers)

Values marked RUNTIME are data-dependent and are written into
`artifacts/cr2_config_resolved.json` by 01_cr2_reproduce.py after the actual CR2
run; both arms read them back from there so the ISOKANN arm and the CR2 arm consume
identical preprocessing.

This mirrors realtime_kernel/mef/config.py one-for-one; only the dataset, the
kernel (PseudotimeKernel on DPT instead of the WOT RealTimeKernel) and the pinned
CR2 settings differ. The ISOKANN training schedule, attribution method, CBC,
heatmap and SD-threshold blocks are kept identical to the MEF / pharyngeal
benchmarks so the three analyses share the established ISOKANN procedure exactly.

How the ISOKANN arm consumes this data (deliberate, documented choices):
  * The CR2 PseudotimeKernel builds its kNN graph on the MultiVI integrated latent
    (`sc.pp.neighbors(use_rep="MultiVI_latent")`) and a DPT pseudotime. That graph
    transition matrix T is THE operator both arms share (kchi = T @ chi). It is
    exactly the matrix CR2's GPCCA consumes.
  * ISOKANN's chi network is parametrised by the **50 PCs computed on the dataset's
    own shipped HVG mask** (`var["hvg_multiVI"]`, 4000 genes) — the direct analogue
    of the MEF benchmark (PCA on the shipped HVGs). Features != kernel-construction
    space is already the case in the MEF/pharyngeal benchmarks (T from WOT, features
    from PCA), so this is the same, established setup. The PCs give the linear
    loadings needed for the gradient->gene chain rule in driver attribution.

One honest deviation, forced by CR2's hematopoiesis pipeline and flagged here:
  * CR2's hematopoiesis analysis provides no curated lineage-driver gene list (it
    shows two example pDC genes, RUNX2 + TCF4). To run the SAME driver panel as the
    MEF / pharyngeal notebooks we supply a canonical literature pDC transcription-
    factor / marker panel as the ground-truth driver list for the focal pDC lineage
    (the hematopoiesis analogue of pharyngeal mTEC / MEF IPS), applied IDENTICALLY to
    the GPCCA and ISOKANN rankings.
"""

from __future__ import annotations
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
# cellrank2_reproducibility is shared by all benchmarks; it is kept two levels up at
# cr2_benchmark/ (this config lives in pseudotime_kernel/hematopoiesis/).
CR2_REPO     = os.path.join(HERE, "..", "..", "cellrank2_reproducibility")
CR2_DATA     = os.path.join(CR2_REPO, "data")
HEMATO_RAW   = os.path.join(CR2_DATA, "hematopoiesis", "processed", "gex_preprocessed.h5ad")
ARTIFACTS    = os.path.join(HERE, "artifacts")
FIGURES      = os.path.join(HERE, "figures")
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Resolved (runtime) config written by 01_cr2_reproduce.py and read by everyone.
RESOLVED_JSON = os.path.join(ARTIFACTS, "cr2_config_resolved.json")

# =============================================================================
# PINNED — dataset definition (dpt.py)
# =============================================================================
CELLTYPE_KEY = "l2_cell_type"   # PINNED  cluster labels used everywhere
# adata = adata[adata.obs["l2_cell_type"].isin(CELLTYPES_TO_KEEP)]
CELLTYPES_TO_KEEP = [
    "HSC", "MK/E prog", "Proerythroblast", "Erythroblast", "Normoblast",
    "cDC2", "pDC", "G/M prog", "CD14+ Mono",
]   # PINNED (dpt.py) -> 24 440 cells

# =============================================================================
# PINNED — neighbour graph / DPT (dpt.py)
# =============================================================================
# sc.pp.neighbors(adata, use_rep="MultiVI_latent")    (n_neighbors=15 default)
NEIGHBORS_USE_REP     = "MultiVI_latent"   # PINNED
# sc.tl.diffmap(adata, n_comps=15); root = HSC cell argmax X_diffmap[:,5];
# adata.uns["iroot"] = root; sc.tl.dpt(adata, n_dcs=6)
DIFFMAP_N_COMPS       = 15           # PINNED
DIFFMAP_ROOT_COMP     = 5            # PINNED  (component used to pick the HSC root)
DPT_N_DCS             = 6            # PINNED
ROOT_CLUSTER          = "HSC"        # PINNED  (root chosen among HSC cells)
TIME_KEY              = "dpt_pseudotime"

# =============================================================================
# PINNED — PseudotimeKernel (dpt.py)
# =============================================================================
# ptk = PseudotimeKernel(adata, time_key="dpt_pseudotime")
#         .compute_transition_matrix(threshold_scheme="soft")
PTK_THRESHOLD_SCHEME  = "soft"       # PINNED

# =============================================================================
# PINNED — GPCCA estimator (dpt.py)
# =============================================================================
SCHUR_N_COMPONENTS    = 20           # PINNED  estimator.compute_schur(n_components=20)
N_MACROSTATES         = 6            # PINNED  estimator.compute_macrostates(6, ...)
MACROSTATE_CLUSTER_KEY = "l2_cell_type"   # PINNED
# estimator.set_terminal_states(["pDC","cDC2","CD14+ Mono","Normoblast"])
TERMINAL_STATES       = ["pDC", "cDC2", "CD14+ Mono", "Normoblast"]  # PINNED
# TSI: estimator.tsi(n_macrostates=7, terminal_states=["CD14+ Mono","Normoblast","cDC2","pDC"], ...)
TSI_N_MACROSTATES     = 7            # PINNED
TSI_TERMINAL_STATES   = ["CD14+ Mono", "Normoblast", "cDC2", "pDC"]  # PINNED (order as in script)
# Fate probabilities: estimator.compute_fate_probabilities(tol=1e-7)
FATE_TOL              = 1e-7         # PINNED

# ISOKANN membership dimension. NOT tied to CR2's N_MACROSTATES=6 (which GPCCA uses
# only to surface the 4 terminal states it keeps): the number of slow ISOKANN modes
# is set by the spectral gap of the PseudotimeKernel operator T. On this data there
# are 4 terminal lineages (pDC, cDC2, CD14+ Mono, Normoblast); GPCCA needed 6
# macrostates to surface them, and the Koopman spectrum's largest gap sits after ~5
# eigenvalues, so k in {4,5,6} are all defensible. We treat k as the load-bearing
# choice (a Koopman-spectrum panel in the notebook is the ISOKANN analogue of GPCCA's
# TSI for choosing #states), default to k=4 (the number of terminal lineages), and
# provide k=5 / k=6 alternates (ARMK_SUFFIX + K_CHI_OVERRIDE) for the comparison.
K_CHI                 = int(os.environ.get("K_CHI_OVERRIDE", 4))   # primary=4

# Source / progenitor population (the hematopoiesis analogue of pharyngeal
# "progenitors" / MEF "MEF/other"): the HSC root of the differentiation tree.
PROGENITOR_LABEL      = "HSC"

# =============================================================================
# PINNED — preprocessing for the ISOKANN features
# =============================================================================
# PCA on the dataset's OWN shipped HVG mask (var["hvg_multiVI"], 4000 genes), giving
# the 50 PCs fed to ISOKANN + the loadings for the gradient->gene chain rule. The
# mask is stored as a categorical "True"/"False" column in this h5ad.
HVG_VAR_KEY           = "hvg_multiVI"
HVG_VAR_TRUE          = "True"       # the "selected" category value
PCA_N_COMPS           = 50
ISOKANN_N_PCS         = 50

# =============================================================================
# Driver / TF-recovery benchmark (focal lineage = pDC, the hematopoiesis analogue
# of pharyngeal mTEC / MEF IPS — CR2's dpt.py runs its driver example on pDC)
# =============================================================================
DRIVER_LINEAGE        = "pDC"
DRIVER_CLUSTER_KEY    = "l2_cell_type"

# Curated pDC transcription-factor / marker panel (ground-truth driver list for pDC)
# — a canonical literature panel: the master pDC regulators (TCF4/E2-2, IRF8, IRF7,
# SPIB, RUNX2, BCL11A, …) plus the pDC surface/effector markers used to define the
# state (IL3RA/CD123, CLEC4C/BDCA2, LILRA4, GZMB, JCHAIN, MZB1, …). Applied
# IDENTICALLY to the GPCCA and ISOKANN rankings; the notebook intersects this with
# the post-filter gene universe (exactly as MEF does for IPS markers). RUNX2 + TCF4
# are the two genes CR2's own dpt.py highlights for pDC.
PDC_GENES = [
    # master / lineage-determining TFs
    "TCF4", "IRF8", "IRF7", "IRF4", "SPIB", "RUNX2", "BCL11A", "ZEB2",
    "POU2F2", "NFIL3", "TCF3", "BCL6",
    # pDC surface / effector / secretory markers
    "IL3RA", "CLEC4C", "LILRA4", "GZMB", "JCHAIN", "MZB1", "PLD4", "SERPINF1",
    "TCL1A", "ITM2C", "PTGDS", "DERL3", "SCT", "PACSIN1", "SMPD3", "UGCG",
    "CCDC50", "PTPRS", "MAP1A", "SLC15A4", "CIITA",
]

# Genes excluded from driver ranking — PINNED (same CR2-style filter as MEF, human
# gene symbols: mitochondrial MT-, ribosomal RPL/RPS, haemoglobin HB*).
GENE_EXCLUDE_PREFIXES = ("MT-", "RPL", "RPS", "^HB[^(P)]")   # str patterns
MOUSE_TFS_LOCAL       = os.path.join(CR2_DATA, "generic", "allTFs_hg38.txt")  # human TFs

# Cell-cycle gene lists used to filter drivers — Tirosh et al. (HUMAN symbols).
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2",
    "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1",
    "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
    "BRIP1", "E2F8",
]
G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]

DRIVER_TOPK           = [10, 20, 50, 100]   # overlap@k cutoffs

# =============================================================================
# Gradient-based attribution (the ISOKANN-native driver method) — identical spec
# =============================================================================
ATTR_METHOD           = "dchi/dgene via chain rule through PCA loadings (per cell, signed)"
ATTR_RANK_AGG         = "max_over_cells |dchi/dgene|"
ATTR_DIRECTION        = "sign(mean_cell dchi/dgene)  ==  sign(corr(gene, chi_i))"
ATTR_GENE_UNIVERSE    = "HVGs minus cell-cycle(S/G2M) and MT-/RPL/RPS/HB (CR2-style filter)"
ATTR_HEATMAP_GRID     = 100
ATTR_HEATMAP_BW       = 0.03

# =============================================================================
# CBC (decision-boundary) benchmark
# =============================================================================
CBC_CLUSTER_KEY       = "l2_cell_type"
CBC_REP               = "MultiVI_latent"
# Boundaries resolved at runtime: HSC -> each terminal state.

# =============================================================================
# ISOKANN training (Arm K) — identical schedule to the MEF / pharyngeal benchmarks
# =============================================================================
ISOKANN_HIDDEN        = [128, 64, 32]
WARMUP_TARGET         = "shiftscale"
MAIN_TARGET           = "isa"
WARMUP_ITERS          = 150
MAIN_ITERS            = 600
EPOCHS_PER_ITER       = 50
LR                    = 1e-3
LR_DECAY              = 0.999
GRAD_CLIP             = 5.0
BATCH                 = 4096
PAIR_MONITOR_FRAC     = 0.10
SEED                  = 0

SD_LIVE_THRESHOLD     = 0.05
