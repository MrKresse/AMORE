"""
Shared, single-source-of-truth configuration for the ISOKANN+AMORE vs CellRank2
benchmark on the **mouse-embryonic-fibroblast (MEF) reprogramming** dataset
(Schiebinger et al. 2019; CellRank 2, Nature Methods 2024 — the RealTimeKernel
"mef" analysis).

Every value below marked PINNED is read verbatim from the CR2 reproducibility repo:
    cellrank2_reproducibility/scripts/realtime_kernel/mef/
        realtime_informed_pseudotime.py   (terminal states / fate / TSI / macrostates)
        wot.py                            (data loading + terminal-state set)

Values marked RUNTIME are data-dependent and are written into
`artifacts/cr2_config_resolved.json` by 01_cr2_reproduce.py after the actual CR2
run; both arms read them back from there so the ISOKANN arm and the CR2 arm consume
identical preprocessing.

This mirrors realtime_kernel/pharyngeal/config.py one-for-one; only the dataset and
its pinned CR2 settings differ. The training schedule, attribution method, CBC,
heatmap and SD-threshold blocks are kept identical to the pharyngeal benchmark so
the two analyses share the established ISOKANN procedure exactly.

The HVG universe is the dataset's OWN published `var["highly_variable"]` mask (1479
genes, shipped with the serum subset) — not a recomputation. scanpy's `sc.pp.pca`
auto-uses that mask (`use_highly_variable=True` when the column exists), so CR2's
`sc.pp.pca(adata)` already runs PCA on those 1479 HVGs, exactly mirroring the
pharyngeal benchmark (PCA on HVGs). No preprocessing deviation.

One honest deviation, forced by CR2's MEF pipeline and flagged here:
  * CR2's MEF analysis provides no curated lineage-driver list (it benchmarks fate &
    pseudotime, not drivers). To run the SAME driver panel as the pharyngeal notebook
    we supply a canonical literature pluripotency/iPSC marker panel as the ground-truth
    driver list for the IPS lineage (the MEF analogue of pharyngeal's mTEC markers),
    applied IDENTICALLY to the GPCCA and ISOKANN rankings.
"""

from __future__ import annotations
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
# cellrank2_reproducibility is shared by both realtime_kernel benchmarks; it is kept
# two levels up at cr2_benchmark/ (this config lives in realtime_kernel/mef/).
CR2_REPO     = os.path.join(HERE, "..", "..", "cellrank2_reproducibility")
CR2_DATA     = os.path.join(CR2_REPO, "data")
# serum subset h5ad ships the precomputed CR2 RealTimeKernel transition matrix in
# obsp["transition_matrix"] (row-stochastic), so no WOT recomputation is needed.
MEF_RAW      = os.path.join(CR2_DATA, "mef", "reprogramming_schiebinger_serum.h5ad")
ARTIFACTS    = os.path.join(HERE, "artifacts")
FIGURES      = os.path.join(HERE, "figures")
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Resolved (runtime) config written by 01_cr2_reproduce.py and read by everyone.
RESOLVED_JSON = os.path.join(ARTIFACTS, "cr2_config_resolved.json")

# =============================================================================
# PINNED — dataset definition
# =============================================================================
# adata = cr.datasets.reprogramming_schiebinger(...); adata[adata.obs.serum == "True"]
# The serum subset is exactly this selection and additionally carries the published
# transition matrix.
SUBSET_SERUM_ONLY     = True          # PINNED  (serum == "True")
CELLTYPE_KEY          = "cell_sets"   # PINNED  cluster labels used everywhere

# =============================================================================
# PINNED — preprocessing (scanpy defaults exactly as called in the CR2 mef scripts)
# =============================================================================
# sc.pp.pca(adata)                       -> n_comps=50 (scanpy default); auto-uses the
#                                           shipped var["highly_variable"] mask (1479 genes)
# sc.pp.neighbors(adata, random_state=0) -> n_neighbors=15, n_pcs=50 (defaults)
PCA_N_COMPS           = 50            # PINNED (scanpy sc.pp.pca default)
NEIGHBORS_RANDOM_STATE = 0           # PINNED
NEIGHBORS_N_NEIGHBORS = 15           # PINNED (scanpy default)
NEIGHBORS_N_PCS       = 50           # uses the 50 PCs above

# Features fed to ISOKANN: the 50 PCs (== adata.obsm["X_pca"][:, :50]).
ISOKANN_N_PCS         = 50

# HVG universe = the dataset's OWN published mask (var["highly_variable"], 1479 genes),
# the same mask sc.pp.pca consumes. Used for the ISOKANN attribution universe / HVG model.
HVG_VAR_KEY           = "highly_variable"

# =============================================================================
# PINNED — RealTimeKernel (published transition matrix shipped with the serum subset)
# =============================================================================
TIME_KEY              = "day"         # PINNED (adata.obs["day"])
# rtk = RealTimeKernel.from_wot(adata, path=wot_tmaps, time_key="day");
# rtk.transition_matrix = load_npz(all_connectivities.npz)   (forward, row-stochastic)
# The serum subset stores exactly this matrix in obsp["transition_matrix"].
RTK_OBSP_KEY          = "transition_matrix"

# =============================================================================
# PINNED — GPCCA estimator (realtime_informed_pseudotime.py)
# =============================================================================
SCHUR_N_COMPONENTS    = 10           # PINNED  estimator.compute_schur(n_components=10)
N_MACROSTATES         = 4            # PINNED  estimator.compute_macrostates(n_states=4)
MACROSTATE_CLUSTER_KEY = "cell_sets" # PINNED
# set_terminal_states(states=["IPS","Neural","Trophoblast","Stromal"])
TERMINAL_STATES       = ["IPS", "Neural", "Trophoblast", "Stromal"]  # PINNED
# TSI: estimator.tsi(n_macrostates=10, terminal_states=["Neural","IPS","Trophoblast","Stromal"], ...)
TSI_N_MACROSTATES     = 10           # PINNED
TSI_TERMINAL_STATES   = ["Neural", "IPS", "Trophoblast", "Stromal"]  # PINNED (order as in script)
# Fate probabilities: estimator.compute_fate_probabilities()

# ISOKANN membership dimension. NOT tied to CR2's pinned N_MACROSTATES=4: GPCCA was
# told to keep exactly the 4 TERMINAL states, but the Koopman operator of this system
# has MORE metastable sets (a huge non-metastable MEF/other source + an intermediate).
# The number of slow ISOKANN modes is set by the spectral gap of T, which on this data
# is unambiguous: eigenvalues [1.000, 0.998, 0.996, 0.995, 0.991, 0.988] then a 0.0245
# gap (the largest in the spectrum) before 0.963 -> SIX metastable states. With k=4 the
# source/intermediates get no mode of their own and contaminate the terminal committors
# (the "IPS" column was actually the MEF source: its top cells were 100% MEF/other),
# which also spikes the ISA loss. With k=6 each terminal gets a clean committor and the
# 4 terminal states are mapped to their columns by Hungarian assignment (the 2 extra
# columns = source + intermediate, not used in the terminal head-to-head). See the
# Koopman-spectrum panel in the notebook (the ISOKANN analogue of GPCCA's TSI/spectrum
# for choosing the number of states).
K_CHI                 = int(os.environ.get("K_CHI_OVERRIDE", 6))   # primary=6; set K_CHI_OVERRIDE=4 for the k=4 comparison run

# Source / progenitor population (the MEF analogue of pharyngeal "progenitors").
PROGENITOR_LABEL      = "MEF/other"

# =============================================================================
# Driver / TF-recovery benchmark (focal lineage = IPS, the MEF analogue of mTEC)
# =============================================================================
DRIVER_LINEAGE        = "IPS"
DRIVER_CLUSTER_KEY    = "cell_sets"

# Curated iPSC / pluripotency marker genes (ground-truth driver list for IPS) —
# a canonical literature panel (Takahashi-Yamanaka core factors + naive-pluripotency
# and 2-cell/totipotency markers used by Schiebinger et al. to score the IPS state).
# Applied IDENTICALLY to the GPCCA and ISOKANN rankings. The notebook intersects this
# with the post-filter gene universe (exactly as pharyngeal does for mTEC markers).
IPS_GENES = [
    "Pou5f1", "Sox2", "Nanog", "Klf4", "Klf2", "Zfp42", "Esrrb", "Sall4",
    "Lin28a", "Lin28b", "Dppa1", "Dppa2", "Dppa3", "Dppa4", "Dppa5a", "Utf1",
    "Tdgf1", "Fbxo15", "Nr0b1", "Tcl1", "Tcl1b1", "Tex19.1", "Obox6", "Sox15",
    "Prdm14", "Tfcp2l1", "Fgf4", "Zfp296", "Nr5a2", "Tbx3", "Gdf3", "Pim2",
    "Morc1", "Trh", "Spp1", "Apoa1",
]

# Genes excluded from driver ranking — PINNED (same CR2 filter as pharyngeal).
GENE_EXCLUDE_PREFIXES = ("mt.", "Rpl", "Rps", "^Hb[^(p)]")   # str.startswith patterns
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

DRIVER_TOPK           = [10, 20, 50, 100]   # overlap@k cutoffs

# =============================================================================
# Gradient-based attribution (the ISOKANN-native driver method) — identical spec
# =============================================================================
ATTR_METHOD           = "dchi/dgene via chain rule through PCA loadings (per cell, signed)"
ATTR_RANK_AGG         = "max_over_cells |dchi/dgene|"
ATTR_DIRECTION        = "sign(mean_cell dchi/dgene)  ==  sign(corr(gene, chi_i))"
ATTR_GENE_UNIVERSE    = "HVGs minus cell-cycle(S/G2M) and mt./Rpl/Rps/Hb (CR2 filter)"
ATTR_HEATMAP_GRID     = 100
ATTR_HEATMAP_BW       = 0.03

# =============================================================================
# CBC (decision-boundary) benchmark
# =============================================================================
CBC_CLUSTER_KEY       = "cell_sets"
CBC_REP               = "X_pca"
# Boundaries resolved at runtime: MEF/other -> each terminal state.

# =============================================================================
# ISOKANN training (Arm K) — identical schedule to the pharyngeal benchmark
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
