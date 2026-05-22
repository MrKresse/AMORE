"""
Extended validation metrics for LARRY ISOKANN.

Replaces Spearman with:
  - Mutual information (sklearn)
  - AUC-ROC (fate-prediction as a classification problem)
  - Hypergeometric enrichment test for known TFs in the top-k sensitivity genes

The TF enrichment test answers the key question:
  "Does the chi sensitivity preferentially pick up transcription factors compared
   to randomly selecting the same number of genes from the expressed universe?"

Hypergeometric null:
  N = total expressed genes
  K = TF genes in N (AnimalTFDB3 / Lambert 2018 curated list)
  n = 100 (top-sensitivity genes examined)
  k = TF genes in top-n
  p-value = P(X >= k | Hypergeometric(N, K, n))
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, pointbiserialr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Mouse TF gene list ─────────────────────────────────────────────────────────
# Curated from AnimalTFDB3 (Zhang et al. 2019) and Lambert et al. 2018 Nat. Rev.
# Covers the major TF families expressed in mouse hematopoiesis.
# Symbols match standard MGI gene nomenclature used in 10x/cospar data.

MOUSE_TFS = {
    # ── bHLH ──────────────────────────────────────────────────────────────────
    "Tal1","Scl","Lyl1","Tal2",
    "Myc","Mycn","Mycl","Max","Mxi1","Mlx","Mnt",
    "Hes1","Hes2","Hes5","Hey1","Hey2","Heyl",
    "Id1","Id2","Id3","Id4",
    "Tcf3","Tcf4","Tcf12","Heb",
    "Atoh1","Neurod1","Neurod2","Neurog1","Neurog2",
    "Arnt","Hif1a","Epas1","Ahr",
    # ── bZIP ──────────────────────────────────────────────────────────────────
    "Jun","Junb","Jund","Fos","Fosb","Fosl1","Fosl2",
    "Atf1","Atf2","Atf3","Atf4","Atf5","Atf6","Atf7",
    "Cebpa","Cebpb","Cebpd","Cebpe","Cebpg","Cebpz",
    "Nfe2","Nfe2l1","Nfe2l2","Nfe2l3",
    "Bach1","Bach2","Maf","Maff","Mafb","Mafg","Mafk",
    "Nrl","Xbp1","Ddit3","Dbp","Tef","Hlf","Nfil3",
    # ── ETS ───────────────────────────────────────────────────────────────────
    "Spi1","Spib","Spic",
    "Fli1","Fli1b","Erg","Flii",
    "Ets1","Ets2","Etv1","Etv2","Etv4","Etv5","Etv6","Etv7",
    "Gfi1","Gfi1b",
    "Elf1","Elf2","Elf4","Elf5",
    "Elk1","Elk3","Elk4","Srf",
    "Gabpa","Gabpb1",
    "Ets21c","Ehf","Ese3",
    # ── GATA ──────────────────────────────────────────────────────────────────
    "Gata1","Gata2","Gata3","Gata4","Gata5","Gata6",
    "Zfpm1","Zfpm2",  # FOG co-factors
    # ── KLF / SP ──────────────────────────────────────────────────────────────
    "Klf1","Klf2","Klf3","Klf4","Klf5","Klf6","Klf7","Klf8",
    "Klf9","Klf10","Klf11","Klf12","Klf13","Klf14","Klf15","Klf16",
    "Sp1","Sp2","Sp3","Sp4","Sp5","Sp6","Sp7","Sp8","Sp9",
    # ── RUNX ──────────────────────────────────────────────────────────────────
    "Runx1","Runx2","Runx3","Cbfb",
    # ── IRF ───────────────────────────────────────────────────────────────────
    "Irf1","Irf2","Irf3","Irf4","Irf5","Irf7","Irf8","Irf9",
    # ── STAT ──────────────────────────────────────────────────────────────────
    "Stat1","Stat2","Stat3","Stat4","Stat5a","Stat5b","Stat6",
    # ── NF-kB ─────────────────────────────────────────────────────────────────
    "Nfkb1","Nfkb2","Rela","Relb","Rel",
    # ── Ikaros / Helios / Aiolos ──────────────────────────────────────────────
    "Ikzf1","Ikzf2","Ikzf3","Ikzf4","Ikzf5",
    # ── Zinc-finger ───────────────────────────────────────────────────────────
    "Trp53","Trp63","Trp73",
    "Zeb1","Zeb2","Snai1","Snai2","Snai3","Twist1","Twist2",
    "Bcl11a","Bcl11b","Lmo2","Lmo4",
    "Tox","Tox2","Tox3","Tox4",
    "Hhex","Prox1","Prox2",
    # ── Nuclear receptors ─────────────────────────────────────────────────────
    "Nr3c1","Nr3c2","Nr1h2","Nr1h3","Nr1h4",
    "Ppara","Ppard","Pparg",
    "Rara","Rarb","Rarg","Rxra","Rxrb","Rxrg",
    "Vdr","Lhx2","Lhx4",
    # ── Homeobox ──────────────────────────────────────────────────────────────
    "Hoxa1","Hoxa2","Hoxa3","Hoxa4","Hoxa5","Hoxa7","Hoxa9","Hoxa10","Hoxa11",
    "Hoxb1","Hoxb3","Hoxb4","Hoxb5","Hoxb6","Hoxb7","Hoxb8",
    "Meis1","Meis2","Pbx1","Pbx2","Pbx3",
    "Nkx2-3","Nkx2-5","Nkx3-1",
    # ── PAX ───────────────────────────────────────────────────────────────────
    "Pax5","Pax3","Pax4","Pax6","Pax7","Pax8","Pax9",
    # ── EBF ───────────────────────────────────────────────────────────────────
    "Ebf1","Ebf2","Ebf3","Ebf4",
    # ── T-box ─────────────────────────────────────────────────────────────────
    "Tbx1","Tbx2","Tbx3","Tbx4","Tbx5","Tbx6","Tbx10","Tbx21",
    "Eomes","Tbet",
    # ── Forkhead ──────────────────────────────────────────────────────────────
    "Foxo1","Foxo3","Foxo4","Foxo6",
    "Foxp1","Foxp2","Foxp3","Foxp4",
    "Foxn1","Foxm1","Foxa1","Foxa2","Foxa3",
    # ── MEF2 ──────────────────────────────────────────────────────────────────
    "Mef2a","Mef2b","Mef2c","Mef2d",
    # ── SMAD ──────────────────────────────────────────────────────────────────
    "Smad1","Smad2","Smad3","Smad4","Smad5","Smad6","Smad7","Smad9",
    # ── E2F ───────────────────────────────────────────────────────────────────
    "E2f1","E2f2","E2f3","E2f4","E2f5","E2f6","E2f7","E2f8",
    # ── MYB ───────────────────────────────────────────────────────────────────
    "Myb","Mybl1","Mybl2",
    # ── MITF / TFEB ───────────────────────────────────────────────────────────
    "Mitf","Tfeb","Tfec","Tfe3",
    # ── RORC / ROR ────────────────────────────────────────────────────────────
    "Rorc","Rora","Rorb",
    # ── Erg family ────────────────────────────────────────────────────────────
    "Mecom","Erg","Bcl6","Bcl6b",
    # ── Additional hematopoiesis ──────────────────────────────────────────────
    "Dntt","Rag1","Rag2",
    "Tcf7","Lef1","Tcf7l1","Tcf7l2",
    "Nfatc1","Nfatc2","Nfatc3","Nfatc4",
    "Epc1","Epc2",
    "Hif1a",
    "Lhx2",
    # ── Lymphoid specific ─────────────────────────────────────────────────────
    "Gata3","Rorgt",
}

MOUSE_TFS_LOWER = {g.lower() for g in MOUSE_TFS}


def load_data():
    adata    = sc.read_h5ad(os.path.join(DATA_DIR, "larry_processed.h5ad"))
    X_pca    = np.load(os.path.join(DATA_DIR, "larry_pca.npy"))
    chi_vals = np.load(os.path.join(OUT_DIR,  "larry_chi_vals.npy"))
    dchi_dpc = np.load(os.path.join(OUT_DIR,  "larry_dchi_dx.npy"))
    src      = np.load(os.path.join(DATA_DIR, "larry_src.npy"))
    dst      = np.load(os.path.join(DATA_DIR, "larry_dst.npy"))

    X_expr = adata.X
    if hasattr(X_expr, "toarray"):
        X_expr = X_expr.toarray()
    X_expr = np.array(X_expr, dtype=np.float32)

    return adata, X_pca, chi_vals, dchi_dpc, src, dst, X_expr


def gene_sensitivity_ridge(X_expr, X_pca, dchi_dpc, n_sub=8000):
    """
    Recover gene sensitivities via dual-form Ridge regression.

    Solves  P_c = X_c @ W  for W (n_genes, N_PCS) using the dual form:
        W = X_c.T @ (X_c X_c.T + alpha I)^{-1} P_c

    The dual form inverts an (n_sub × n_sub) matrix rather than (n_genes × n_genes),
    making it tractable when n_genes >> n_sub.
    """
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(X_expr), n_sub, replace=False)
    X_c   = (X_expr[idx] - X_expr[idx].mean(0)).astype(np.float64)
    P_c   = (X_pca[idx]  - X_pca[idx].mean(0)).astype(np.float64)
    alpha = 1.0
    # Gram matrix: (n_sub, n_sub)
    K     = X_c @ X_c.T + alpha * np.eye(n_sub)
    # Dual solution: (n_sub, N_PCS)
    dual  = np.linalg.solve(K, P_c)
    # Primal weights: W = X_c.T @ dual  →  (n_genes, N_PCS)
    W     = X_c.T @ dual
    mean_dpc  = np.abs(dchi_dpc).mean(0)
    gene_sens = np.abs(W) @ mean_dpc
    return gene_sens.astype(np.float32), W.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Better fate-bias metrics: MI + AUC-ROC
# ══════════════════════════════════════════════════════════════════════════════

def fate_bias_metrics(adata, chi_vals):
    """Compute MI and AUC-ROC between chi and all fate-bias columns."""
    obs      = adata.obs
    times    = obs["time_info"].astype(str).values
    day2     = times == "2"
    chi_d2   = chi_vals[day2].reshape(-1, 1)
    chi_all  = chi_vals.reshape(-1, 1)

    fate_cols = [c for c in obs.columns if c.startswith("progenitor_")]
    results = {}

    print("\n-- Fate-bias metrics (day-2 progenitors) --")
    print(f"  {'Lineage':<22s}  {'MI':>8}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'n_pos':>6}")
    print(f"  {'-'*60}")

    for col in fate_cols + ["NeuMon_fate_bias"]:
        if col not in obs.columns:
            continue
        bias = pd.to_numeric(obs[col], errors="coerce").values.astype(float)
        lineage = col.replace("progenitor_", "").replace("_fate_bias", "")

        # Day-2 cells only
        mask_d2 = day2 & ~np.isnan(bias)
        if mask_d2.sum() < 20:
            continue

        chi_sub  = chi_vals[mask_d2]
        bias_sub = bias[mask_d2]

        # Mutual information
        mi = mutual_info_regression(chi_sub.reshape(-1, 1), bias_sub,
                                    random_state=42)[0]

        # Binary classification: fate_bias > 0.5 vs rest
        y_bin = (bias_sub > 0.5).astype(int)
        n_pos = y_bin.sum()
        if n_pos >= 5 and n_pos < len(y_bin) - 5:
            try:
                auc_roc = roc_auc_score(y_bin, chi_sub)
                auc_pr  = average_precision_score(y_bin, chi_sub)
            except Exception:
                auc_roc = auc_pr = float("nan")
        else:
            auc_roc = auc_pr = float("nan")

        results[lineage] = dict(mi=mi, auc_roc=auc_roc, auc_pr=auc_pr, n_pos=n_pos)
        print(f"  {lineage:<22s}  {mi:>8.4f}  {auc_roc:>8.4f}  {auc_pr:>8.4f}  {n_pos:>6}")

    # Plot MI and AUC-ROC side by side
    df = pd.DataFrame(results).T.dropna(subset=["auc_roc"])
    if len(df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df_sorted_mi  = df.sort_values("mi", ascending=False)
        df_sorted_auc = df.sort_values("auc_roc", ascending=False)

        for ax, col, title in [
            (axes[0], "mi",      "Mutual information (chi, fate_bias)"),
            (axes[1], "auc_roc", "AUC-ROC (fate committed vs rest)"),
        ]:
            vals = df_sorted_mi[col] if col == "mi" else df_sorted_auc[col]
            idx_  = df_sorted_mi.index if col == "mi" else df_sorted_auc.index
            bars = ax.barh(range(len(vals)), vals.values,
                           color=["crimson" if v > 0.5 else "steelblue" for v in vals])
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(idx_, fontsize=8)
            ax.set_xlabel(col.upper())
            ax.set_title(title)
            if col == "auc_roc":
                ax.axvline(0.5, ls="--", c="gray", lw=1, label="random")
                ax.legend(fontsize=7)

        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "metrics_fatebias.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: metrics_fatebias.png")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TF enrichment: hypergeometric test
# ══════════════════════════════════════════════════════════════════════════════

def tf_enrichment_test(gene_names, gene_sens, top_n_list=(50, 100, 200)):
    """
    Hypergeometric enrichment of known TFs in the top-n sensitivity genes.

    H0: the top-n genes are a random sample from the expressed universe.
    H1: the top-n genes are enriched for TFs.
    """
    gene_lower = {g.lower(): g for g in gene_names}

    # TFs present in the expressed gene set
    tfs_in_universe = {g for g in gene_names if g.lower() in MOUSE_TFS_LOWER}
    N = len(gene_names)            # universe size
    K = len(tfs_in_universe)       # TFs in universe
    print(f"\n-- TF enrichment test --")
    print(f"  Universe:      {N:,} genes")
    print(f"  TFs in set:    {K:,}  ({100*K/N:.1f}% of universe)")
    print(f"\n  {'top-n':>6}  {'TFs found':>10}  {'expected':>9}  {'p-value':>12}  {'fold':>6}")
    print(f"  {'-'*55}")

    top_idx = np.argsort(gene_sens)[::-1]
    results = {}

    for n in top_n_list:
        top_genes = gene_names[top_idx[:n]]
        k = sum(1 for g in top_genes if g.lower() in MOUSE_TFS_LOWER)
        expected = n * K / N
        # P(X >= k) under hypergeometric(N, K, n)
        pval = hypergeom.sf(k - 1, N, K, n)
        fold = k / expected if expected > 0 else float("nan")
        print(f"  {n:>6}  {k:>10}  {expected:>9.1f}  {pval:>12.2e}  {fold:>6.2f}x")

        top_tfs = [g for g in top_genes if g.lower() in MOUSE_TFS_LOWER]
        results[n] = dict(k=k, expected=expected, pval=pval, fold=fold, tfs=top_tfs)

    # Also label ALL known TFs in the top-200 with their rank
    top200 = gene_names[top_idx[:200]]
    print(f"\n  Known TFs in top-200 (by sensitivity rank):")
    for rank, g in enumerate(top200, 1):
        if g.lower() in MOUSE_TFS_LOWER:
            print(f"    #{rank:>3d}  {g}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Sensitivity bar chart with all TFs labelled
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity_plot_with_tfs(gene_names, gene_sens, top_n=60):
    """Bar plot of top-n genes with all known TFs highlighted and labelled."""
    top_idx   = np.argsort(gene_sens)[::-1]
    top_genes = gene_names[top_idx[:top_n]]
    top_vals  = gene_sens[top_idx[:top_n]]
    is_tf     = [g.lower() in MOUSE_TFS_LOWER for g in top_genes]

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ["crimson" if tf else "lightsteelblue" for tf in is_tf]
    bars   = ax.bar(range(top_n), top_vals, color=colors, edgecolor="none")

    # Label TF bars
    for i, (g, tf) in enumerate(zip(top_genes, is_tf)):
        if tf:
            ax.text(i, top_vals[i] + 0.001, g, rotation=90, fontsize=6,
                    ha="center", va="bottom", color="crimson", fontweight="bold")
        else:
            ax.text(i, top_vals[i] + 0.001, g, rotation=90, fontsize=5,
                    ha="center", va="bottom", color="gray")

    ax.set_xticks([])
    ax.set_ylabel("|dchi/dgene|  (mean over cells)")
    ax.set_title(f"Top-{top_n} sensitivity genes  (red = known TF, labels on TFs)")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="crimson", label="Known TF"),
                        Patch(facecolor="lightsteelblue", label="Other gene")],
              fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "metrics_sensitivity_tfs.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: metrics_sensitivity_tfs.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data ...")
    adata, X_pca, chi_vals, dchi_dpc, src, dst, X_expr = load_data()
    gene_names = np.array(adata.var_names)
    N_PCS = X_pca.shape[1]
    print(f"  {len(chi_vals):,} cells | {len(gene_names):,} genes | {N_PCS} PCs")

    # 1. Gene sensitivity
    print("\nRecovering gene sensitivities via Ridge regression ...")
    gene_sens, W = gene_sensitivity_ridge(X_expr, X_pca, dchi_dpc)

    # 2. Fate-bias metrics
    fate_results = fate_bias_metrics(adata, chi_vals)

    # 3. TF enrichment
    tf_results = tf_enrichment_test(gene_names, gene_sens, top_n_list=[50, 100, 200, 500])

    # 4. Labelled sensitivity bar chart
    sensitivity_plot_with_tfs(gene_names, gene_sens, top_n=60)

    # 5. Save ranked gene table with TF annotation
    is_tf_arr = np.array([g.lower() in MOUSE_TFS_LOWER for g in gene_names])
    pd.DataFrame({
        "gene":          gene_names,
        "sensitivity":   gene_sens,
        "rank":          np.argsort(np.argsort(-gene_sens)),
        "is_known_tf":   is_tf_arr,
    }).sort_values("sensitivity", ascending=False).to_csv(
        os.path.join(OUT_DIR, "larry_gene_sensitivity_annotated.csv"), index=False)
    print(f"\n  Saved: larry_gene_sensitivity_annotated.csv")
    print(f"\nDone. All outputs -> {OUT_DIR}/")
