# -*- coding: utf-8 -*-
"""
Benchmark v2 plotting script.

Panels
------
A  Convergence diagnostics: val loss + chi-SD per mode vs iteration (all seeds)
B  Headline metric grid: val loss / reference correlation / min chi-SD (boxplots)
C  Mode-resolved chi-SD bar chart
D  Qualitative chi maps:
     triple_well   : chi_1/2/3 filled contour on (x,y)
     alanine_5ps   : chi scatter on (phi,psi) + (chi_1,chi_2) scatter colored by k-means state
     alanine_multi : same as alanine_5ps — check for 3-state separation

Reference data loaded from panel0/ outputs.

Run from: AMORE/examples/benchmark_v2/
"""

import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment

HERE     = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
P0_DIR   = os.path.join(HERE, "panel0")
DATA_DIR = os.path.join(HERE, "..", "benchmark", "data")
FIG_DIR  = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

VARIANTS = ["isa", "gramschmidt", "pseudoinv", "svd", "cross", "vamp2"]
VAR_LABELS = {
    "isa":         "ISA",
    "gramschmidt": "GramSchmidt",
    "pseudoinv":   "PseudoInv",
    "svd":         "SVD",
    "cross":       "Cross",
    "vamp2":       "VAMP2",
}
N_SEEDS  = 5
SD_COLLAPSE = 0.01    # below this = collapsed mode
SD_LIVE     = 0.05    # above this = live mode
COLORS = plt.cm.tab10(np.linspace(0, 1, len(VARIANTS)))

DATASETS = ["triple_well", "alanine_5ps", "alanine_multi_tau"]
DS_LABELS = {
    "triple_well":       "Triple-well",
    "alanine_5ps":       "ADP tau=5ps",
    "alanine_multi_tau": "ADP tau=5ps+0.1ps",
}


# ── Data loading helpers ────────────────────────────────────────────────────────

def load_runs(ds_name):
    """
    Returns dict: variant -> list of per-seed dicts
    Each seed dict: val_loss (T,), chi_sd_history (T,k), chi_best (N,k), chi_atstop (N,k)
    """
    out = {}
    for v in VARIANTS:
        seeds = []
        for s in range(N_SEEDS):
            d = os.path.join(RUNS_DIR, ds_name, v, f"seed_{s}")
            if not os.path.isdir(d):
                continue
            seed_dict = {}
            for key in ["val_loss", "chi_sd_history", "chi_atstop", "chi_best"]:
                fp = os.path.join(d, f"{key}.npy")
                if os.path.exists(fp):
                    seed_dict[key] = np.load(fp)
            if seed_dict:
                seeds.append(seed_dict)
        if seeds:
            out[v] = seeds
    return out


def best_seed(runs_v):
    """Return index of seed with lowest final val_loss."""
    finals = []
    for sd in runs_v:
        vl = sd.get("val_loss", np.array([np.inf]))
        finals.append(float(np.nanmin(vl)))
    return int(np.argmin(finals))


# ── Reference correlation ───────────────────────────────────────────────────────

def pearson_r(a, b):
    a  = a - a.mean(); b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def hungarian_corr(chi, refs):
    """
    Best Pearson r between columns of chi (N,k) and refs (N,m) via Hungarian assignment.
    Returns mean of assigned correlations (absolute value, sign flip allowed).
    """
    k = chi.shape[1]; m = refs.shape[1]
    cost = np.zeros((k, m))
    for i in range(k):
        for j in range(m):
            cost[i, j] = -abs(pearson_r(chi[:, i], refs[:, j]))
    row, col = linear_sum_assignment(cost)
    return float(-cost[row, col].mean())


def load_tw_reference():
    """Triple-well: empirical committor p_A, p_B, p_C."""
    fp = os.path.join(P0_DIR, "tw_committor.npz")
    if not os.path.exists(fp):
        return None
    d = np.load(fp)
    return np.stack([d["p_A"], d["p_B"], d["p_C"]], axis=1)   # (N_ANC, 3)


def load_adp_reference():
    """ADP: EV2 (the one slow eigenfunction) on occupied cells."""
    fp = os.path.join(P0_DIR, "adp_eigvecs.npz")
    if not os.path.exists(fp):
        return None, None
    d    = np.load(fp)
    ev2  = d["eigvecs"][:, 1]   # (N_CELLS,) second eigenvector
    occ  = d["occupied"]        # (N_CELLS,) bool
    return ev2, occ


# ── Panel A: convergence ───────────────────────────────────────────────────────

def plot_panel_a(ds_name, runs):
    n_v = len(VARIANTS)
    fig, axes = plt.subplots(n_v, 2, figsize=(12, 2.5 * n_v))
    fig.suptitle(f"Panel A — Convergence: {DS_LABELS.get(ds_name, ds_name)}", fontsize=11)

    for vi, v in enumerate(VARIANTS):
        if v not in runs:
            continue
        ax_val = axes[vi, 0]; ax_sd = axes[vi, 1]
        label  = VAR_LABELS[v]

        for si, sd_dict in enumerate(runs[v]):
            vl  = sd_dict.get("val_loss", np.array([]))
            sdh = sd_dict.get("chi_sd_history", np.zeros((0, 3)))
            xs  = np.arange(len(vl))
            ax_val.plot(xs, vl, alpha=0.5, lw=0.8, color=f"C{si}")
            for ki in range(sdh.shape[1]):
                ax_sd.plot(np.arange(len(sdh)), sdh[:, ki],
                           alpha=0.5, lw=0.7, color=f"C{ki}",
                           label=f"chi_{ki+1}" if si == 0 else "_")

        ax_val.set_title(f"{label} — val loss", fontsize=8)
        ax_val.set_yscale("log")
        ax_val.set_xlabel("Iteration"); ax_val.set_ylabel("Val loss")

        ax_sd.axhline(SD_COLLAPSE, color="red",    ls="--", lw=0.8, label="collapse")
        ax_sd.axhline(SD_LIVE,     color="orange", ls="--", lw=0.8, label="live")
        ax_sd.set_title(f"{label} — chi SD per mode", fontsize=8)
        ax_sd.set_xlabel("Iteration"); ax_sd.set_ylabel("SD")
        if vi == 0:
            ax_sd.legend(fontsize=7, ncol=3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"panelA_{ds_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Panel B: headline metric grid ─────────────────────────────────────────────

def plot_panel_b(ds_name, runs, ref_chi=None, adp_ev2=None, adp_occ=None,
                 al_npz=None):
    """
    3 columns: val loss | reference correlation | min chi-SD
    Boxplots over seeds. Collapsed runs as red X.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Panel B — Metrics: {DS_LABELS.get(ds_name, ds_name)}", fontsize=11)

    ax_val, ax_cor, ax_sd = axes

    for vi, v in enumerate(VARIANTS):
        if v not in runs:
            continue
        x = vi + 1
        label = VAR_LABELS[v]

        val_finals = []
        corr_vals  = []
        min_sds    = []
        collapsed_count = 0

        for si, sd_dict in enumerate(runs[v]):
            vl  = sd_dict.get("val_loss", np.array([]))
            sdh = sd_dict.get("chi_sd_history", None)
            chi = sd_dict.get("chi_best", sd_dict.get("chi_atstop", None))

            val_f = float(np.nanmin(vl)) if len(vl) else np.nan
            val_finals.append(val_f)

            # Check collapse
            if sdh is not None and len(sdh):
                sd_final = sdh[-1]
                min_sd   = float(sd_final.min())
            else:
                min_sd = np.nan
            min_sds.append(min_sd)

            if not np.isnan(min_sd) and min_sd < SD_COLLAPSE:
                collapsed_count += 1

            # Reference correlation
            if chi is not None and ref_chi is not None and "triple_well" in ds_name:
                # Subset chi to anchors in ref_chi
                n = min(chi.shape[0], ref_chi.shape[0])
                corr = hungarian_corr(chi[:n], ref_chi[:n])
                corr_vals.append(corr)
            elif chi is not None and adp_ev2 is not None and al_npz is not None:
                # Map chi to cells, correlate with EV2
                # chi: (N_ANC, k) — N_ANC = 1578 for alanine_5ps, N_ML for multi
                # EV2: (N_CELLS,) on occupied cells
                phi_a = al_npz["anchors_phi"]
                psi_a = al_npz["anchors_psi"]
                n     = min(len(phi_a), chi.shape[0])
                edges = np.linspace(-np.pi, np.pi, 41)
                def to_cell(p, q):
                    bi = np.clip(np.digitize(p, edges)-1, 0, 39)
                    bj = np.clip(np.digitize(q, edges)-1, 0, 39)
                    return bi*40+bj
                ci = to_cell(phi_a[:n], psi_a[:n])
                ev2_at_anchors = adp_ev2[ci]        # (N_ANC,)
                # Pearson corr of chi_best columns vs EV2 (best col by abs corr)
                best_c = max(abs(pearson_r(chi[:n, ki], ev2_at_anchors)) for ki in range(chi.shape[1]))
                corr_vals.append(best_c)

        # Box plots
        def _box(ax, vals, x, color):
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                return
            bp = ax.boxplot([vals], positions=[x], widths=0.6,
                            patch_artist=True,
                            boxprops=dict(facecolor=color, alpha=0.6),
                            medianprops=dict(color="black", lw=2),
                            whiskerprops=dict(lw=1),
                            capprops=dict(lw=1),
                            flierprops=dict(marker="o", ms=4))

        _box(ax_val, val_finals, x, COLORS[vi])
        _box(ax_cor, corr_vals,  x, COLORS[vi])
        _box(ax_sd,  min_sds,    x, COLORS[vi])

        if collapsed_count > 0:
            ax_sd.text(x, SD_COLLAPSE * 0.5, f"{collapsed_count}/{N_SEEDS}",
                       ha="center", va="center", fontsize=7, color="red")

    for ax in axes:
        ax.set_xticks(range(1, len(VARIANTS)+1))
        ax.set_xticklabels([VAR_LABELS.get(v, v) for v in VARIANTS],
                           rotation=30, ha="right", fontsize=8)
    ax_val.set_ylabel("Val loss (log)"); ax_val.set_yscale("log")
    ax_val.set_title("Val loss (within-method)", fontsize=9)
    ax_cor.set_ylabel("Pearson r (vs reference)"); ax_cor.set_ylim(0, 1.05)
    ax_cor.set_title("Reference correlation", fontsize=9)
    ax_sd.set_ylabel("Min chi-SD across modes"); ax_sd.set_yscale("log")
    ax_sd.axhline(SD_COLLAPSE, color="red", ls="--", lw=0.8, label="collapse")
    ax_sd.axhline(SD_LIVE,     color="orange", ls="--", lw=0.8, label="live")
    ax_sd.legend(fontsize=7); ax_sd.set_title("Min mode SD", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"panelB_{ds_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Panel C: mode-resolved chi-SD ──────────────────────────────────────────────

def plot_panel_c(ds_name, runs):
    n_v = len([v for v in VARIANTS if v in runs])
    fig, ax = plt.subplots(figsize=(max(8, n_v * 2), 4))
    fig.suptitle(f"Panel C — Chi-SD per mode: {DS_LABELS.get(ds_name, ds_name)}", fontsize=11)

    x0   = 0
    xticks = []; xlabels = []
    k    = 3
    bar_w = 0.6 / k
    mode_colors = ["steelblue", "tomato", "forestgreen"]

    for vi, v in enumerate(VARIANTS):
        if v not in runs:
            continue
        # Use best seed
        bi   = best_seed(runs[v])
        sdh  = runs[v][bi].get("chi_sd_history", None)
        if sdh is None or len(sdh) == 0:
            continue
        sd_final = sdh[-1]   # (k,)

        for ki in range(len(sd_final)):
            xpos = x0 + ki * bar_w - bar_w * (k-1) / 2
            ax.bar(xpos, sd_final[ki], width=bar_w * 0.9,
                   color=mode_colors[ki % len(mode_colors)], alpha=0.8,
                   label=f"chi_{ki+1}" if vi == 0 else "_")

        xticks.append(x0)
        xlabels.append(VAR_LABELS.get(v, v))
        x0 += 1.2

    ax.axhline(SD_COLLAPSE, color="red",    ls="--", lw=1.0, label=f"collapse (<{SD_COLLAPSE})")
    ax.axhline(SD_LIVE,     color="orange", ls="--", lw=1.0, label=f"live (>{SD_LIVE})")
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("chi SD"); ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=k+2)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"panelC_{ds_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Panel D helpers ────────────────────────────────────────────────────────────

def _ramachandran_scatter(ax, phi, psi, vals, cmap="RdBu_r", vabs=None, s=6, title=""):
    vabs  = vabs or np.abs(vals).max()
    sc = ax.scatter(phi, psi, c=vals, cmap=cmap, vmin=-vabs, vmax=vabs, s=s, rasterized=True)
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
    ticks  = [-np.pi, 0, np.pi]
    tlabs  = [r"$-\pi$", "0", r"$\pi$"]
    ax.set_xticks(ticks, tlabs, fontsize=6); ax.set_yticks(ticks, tlabs, fontsize=6)
    ax.set_xlabel(r"$\phi$", fontsize=7); ax.set_ylabel(r"$\psi$", fontsize=7)
    ax.set_title(title, fontsize=8)
    return sc


# ── Panel D: triple-well chi maps ─────────────────────────────────────────────

def plot_panel_d_tw(runs):
    tw   = np.load(os.path.join(DATA_DIR, "triple_well_koopman.npz"))
    xy   = tw["anchors"]      # (N, 2)
    k    = 3
    n_v  = len([v for v in VARIANTS if v in runs])

    fig, axes = plt.subplots(k, n_v, figsize=(3 * n_v, 3 * k))
    fig.suptitle("Panel D — TW chi maps (best-val-loss seed)", fontsize=11)

    for vi, v in enumerate(VARIANTS):
        if v not in runs:
            continue
        bi  = best_seed(runs[v])
        chi = runs[v][bi].get("chi_best", runs[v][bi].get("chi_atstop"))
        if chi is None:
            continue
        sd  = runs[v][bi].get("chi_sd_history", np.zeros((1,k)))[-1]

        for ki in range(k):
            ax = axes[ki, vi] if n_v > 1 else axes[ki]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=chi[:, ki],
                            cmap="RdBu_r", s=6, rasterized=True)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            title = f"{VAR_LABELS.get(v,v)}\nchi_{ki+1}  SD={sd[ki]:.3f}"
            if sd[ki] < SD_COLLAPSE:
                title += " [COLLAPSED]"
            ax.set_title(title, fontsize=7)
            ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "panelD_TW_maps.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Panel D: ADP Ramachandran + chi scatter ───────────────────────────────────

def plot_panel_d_adp(ds_name, runs, al_phi, al_psi, multi_phi=None, multi_psi=None):
    """
    4 rows × n_methods columns:
      Rows 0-2: chi_1, chi_2, chi_3 on (phi,psi)  — same layout as TW Panel D
      Row 3:    (best two live chi modes) scatter coloured by k-means
    """
    from sklearn.cluster import KMeans

    use_phi = al_phi if multi_phi is None else multi_phi
    use_psi = al_psi if multi_phi is None else multi_psi

    n_v = len([v for v in VARIANTS if v in runs])
    k   = 3
    NROWS = k + 1   # 3 Ramachandran rows + 1 scatter row

    fig, axes = plt.subplots(NROWS, n_v,
                             figsize=(3.2 * n_v, 3.0 * NROWS),
                             squeeze=False)
    fig.suptitle(f"Panel D — ADP chi maps + scatter  ({DS_LABELS.get(ds_name, ds_name)})",
                 fontsize=10)

    col = 0   # actual column counter (only increments for variants with data)
    for vi, v in enumerate(VARIANTS):
        if v not in runs:
            continue

        bi  = best_seed(runs[v])
        chi = runs[v][bi].get("chi_best", runs[v][bi].get("chi_atstop"))
        if chi is None:
            col += 1
            continue

        sd   = runs[v][bi].get("chi_sd_history", np.zeros((1, k)))[-1]
        n    = min(chi.shape[0], len(use_phi))
        phi_ = use_phi[:n]; psi_ = use_psi[:n]; chi_ = chi[:n]

        # Rows 0–2: one Ramachandran plot per chi mode
        for ki in range(k):
            ax  = axes[ki, col]
            val = chi_[:, ki]
            vabs = max(np.abs(val).max(), 1e-6)
            sc  = ax.scatter(phi_, psi_, c=val, cmap="RdBu_r",
                             vmin=-vabs, vmax=vabs, s=5, rasterized=True)
            plt.colorbar(sc, ax=ax, fraction=0.046)
            collapsed = "  [COLLAPSED]" if sd[ki] < SD_COLLAPSE else ""
            ax.set_title(f"{VAR_LABELS.get(v,v)}\nchi_{ki+1} SD={sd[ki]:.3f}{collapsed}",
                         fontsize=7)
            ax.set_xlabel(r"$\phi$", fontsize=6); ax.set_ylabel(r"$\psi$", fontsize=6)
            ticks = [-np.pi, 0, np.pi]
            ax.set_xticks(ticks, [r"$-\pi$","0",r"$\pi$"], fontsize=5)
            ax.set_yticks(ticks, [r"$-\pi$","0",r"$\pi$"], fontsize=5)
            ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)

        # Row 3: chi-space scatter (two live modes)
        ax1  = axes[k, col]
        live = sd > SD_COLLAPSE
        if live.sum() >= 2:
            live_idx = np.where(live)[0][:2]
        else:
            live_idx = np.array([0, 1])
        chi2d = chi_[:, live_idx]
        try:
            km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(chi2d)
            colours = km.labels_
        except Exception:
            colours = np.zeros(n, dtype=int)

        sc2 = ax1.scatter(chi2d[:, 0], chi2d[:, 1], c=colours,
                          cmap="Set1", vmin=-0.5, vmax=2.5, s=4, rasterized=True)
        plt.colorbar(sc2, ax=ax1, fraction=0.046, ticks=[0,1,2])
        k_eff = int((sd > SD_LIVE).sum())
        xi, yi = live_idx[0]+1, live_idx[1]+1
        ax1.set_title(f"chi-space scatter\nchi_{xi} vs chi_{yi}  k_eff={k_eff}/{k}",
                      fontsize=7)
        ax1.set_xlabel(f"chi_{xi}", fontsize=6)
        ax1.set_ylabel(f"chi_{yi}", fontsize=6)
        col += 1

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"panelD_{ds_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load reference data
    tw_ref    = load_tw_reference()
    adp_ev2, adp_occ = load_adp_reference()

    # Load anchor data
    al_npz    = None
    al_phi    = al_psi = None
    ml_phi    = ml_psi = None

    al_path   = os.path.join(DATA_DIR, "alanine_koopman.npz")
    if os.path.exists(al_path):
        al_npz = np.load(al_path)
        al_phi = al_npz["anchors_phi"]
        al_psi = al_npz["anchors_psi"]

    ml_path = os.path.join(DATA_DIR, "alanine_multilag.npz")
    if os.path.exists(ml_path):
        ml = np.load(ml_path)
        ml_phi = ml["anchors_phi"]
        ml_psi = ml["anchors_psi"]

    for ds_name in DATASETS:
        print(f"\n{'='*55}")
        print(f"Plotting: {DS_LABELS.get(ds_name, ds_name)}")
        print(f"{'='*55}")

        runs = load_runs(ds_name)
        if not runs:
            print(f"  No run data found in {RUNS_DIR}/{ds_name}/ — skipping.")
            continue

        print(f"  Variants with data: {list(runs)}")

        # Panel A
        plot_panel_a(ds_name, runs)

        # Panel B
        if "triple_well" in ds_name:
            plot_panel_b(ds_name, runs, ref_chi=tw_ref)
        else:
            plot_panel_b(ds_name, runs,
                         adp_ev2=adp_ev2, adp_occ=adp_occ, al_npz=al_npz)

        # Panel C
        plot_panel_c(ds_name, runs)

        # Panel D
        if "triple_well" in ds_name:
            plot_panel_d_tw(runs)
        elif "multi" in ds_name:
            plot_panel_d_adp(ds_name, runs, ml_phi, ml_psi,
                             multi_phi=ml_phi, multi_psi=ml_psi)
        else:
            plot_panel_d_adp(ds_name, runs, al_phi, al_psi)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
