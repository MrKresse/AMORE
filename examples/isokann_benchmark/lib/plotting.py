# -*- coding: utf-8 -*-
"""
plotting.py — shared figure/table helpers for both benchmark notebooks.

Conventions:
  * coords are (N,2): xy for triple_well/ring, (phi,psi) [rad] for ADP.
  * chi maps are scatter-coloured by chi value; references likewise.
  * scoring uses harness.shape_r (Hungarian |Pearson r|) so colours/columns line up
    with the printed tables.
All heavy arrays come from systems.py / scratch; nothing is recomputed here beyond
cheap correlations.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness


def _subsample(n, max_pts=30000, seed=0):
    return (np.random.default_rng(seed).choice(n, max_pts, replace=False)
            if n > max_pts else np.arange(n))


def memb_reference(system):
    """A fixed set of N_STATES reference memberships for column alignment + display:
    committors (TW) or PCCA+ of the true eigenvectors (ADP, ring)."""
    if system.get("committor_refs") is not None:
        return system["committor_refs"]                      # (3, N)
    return harness.to_memberships(system["ev_refs"].T).T      # PCCA+ of true eigenvectors -> (3, N)


def align_columns(cols, refs, sign=False):
    """Reorder the columns of `cols` (N,k) to best match references `refs` (R,N) by Hungarian
    |Pearson r| (so a fixed reference fixes the column = basin/mode order across methods/seeds).
    sign=True also flips each column's sign to the reference (removes arbitrary eigenvector sign)."""
    from scipy.optimize import linear_sum_assignment
    k, R = cols.shape[1], refs.shape[0]
    S = np.array([[harness.pearson_r(cols[:, j], refs[i]) for i in range(R)] for j in range(k)])
    ri, ci = linear_sum_assignment(-np.abs(np.nan_to_num(S)))
    out = np.zeros((len(cols), R))
    for j, i in zip(ri, ci):
        out[:, i] = cols[:, j] * (np.sign(S[j, i]) if sign else 1.0)
    return out


def _axkw(ax, coords, title):
    ax.set_title(title, fontsize=9)
    if coords.shape[1] == 2:
        pass
    ax.set_xlabel("x" )
    ax.set_ylabel("y")


def _scatter(ax, coords, c, cmap="coolwarm", s=4, vsym=False, deg=False, vmin=None, vmax=None):
    X = coords.copy()
    if deg:
        X = X * 180 / np.pi
    if vsym:
        m = np.nanmax(np.abs(c)) or 1.0
        vmin, vmax = -m, m
    sc = ax.scatter(X[:, 0], X[:, 1], c=c, s=s, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    return sc


def axis_labels(system):
    return (r"$\phi$", r"$\psi$") if system["tag"].startswith("adp") else ("x", "y")


def plot_references(system, deg=None, figsize=None):
    """Plot the numerical ground-truth reference shapes for a system."""
    coords = system["coords"]; refs = system["refs"]; names = system["ref_names"]
    deg = system["tag"].startswith("adp") if deg is None else deg
    R = refs.shape[0]
    # committors live in [0,1] (sequential 0->1 scale); eigenvectors are signed (symmetric)
    is_committor = all(str(n).lower().startswith("p") for n in names)
    ss = _subsample(len(coords))
    fig, ax = plt.subplots(1, R, figsize=figsize or (4.6 * R, 4))
    if R == 1:
        ax = [ax]
    xl, yl = axis_labels(system)
    for i in range(R):
        if is_committor:
            sc = _scatter(ax[i], coords[ss], refs[i][ss], cmap="RdBu_r",
                          vmin=0.0, vmax=1.0, deg=deg)
        else:
            sc = _scatter(ax[i], coords[ss], refs[i][ss], vsym=True, deg=deg)
        ax[i].set_title(names[i], fontsize=10); ax[i].set_xlabel(xl); ax[i].set_ylabel(yl)
        plt.colorbar(sc, ax=ax[i], fraction=0.046)
    fig.tight_layout()
    return fig


def chi_maps_grid(system, results_by_seed, method, view="membership", deg=None, ss=None, max_pts=None):
    """Grid of chi maps: rows = seeds, cols = the chosen representation.

    view='membership' -> N_STATES memberships (softmax native, basis via PCCA+), [0,1] colours,
                         scored vs committors (TW) else ev_refs.
    view='eigfn'      -> the N_STATES-1 non-trivial eigenfunctions, signed colours, vs ev_refs.
    """
    coords = system["coords"]
    deg = system["tag"].startswith("adp") if deg is None else deg
    seeds = sorted(results_by_seed)
    membership = (view == "membership")
    # columns aligned to a FIXED reference so column j = the same basin/mode across all
    # methods & seeds; consistent colour direction (red = high), no arbitrary eigenvector sign.
    if membership:
        refs = memb_reference(system); cmap = "RdBu_r"; vlim = (0.0, 1.0); vsym = False; lab = "χ"
    else:
        refs = system["ev_refs"]; cmap = "RdBu_r"; vlim = (None, None); vsym = True; lab = "EV"
    def rep(chi):
        cols = harness.to_memberships(chi) if membership else harness.eigfns(chi)
        return align_columns(cols, refs, sign=not membership)
    if ss is None:
        ss = _subsample(len(coords), max_pts or 30000)
    xl, yl = axis_labels(system)
    chis = {sd: rep(results_by_seed[sd]["chi_best"]) for sd in seeds}
    k = chis[seeds[0]].shape[1]
    fig, axes = plt.subplots(len(seeds), k, figsize=(3.4 * k, 3.0 * len(seeds)), squeeze=False)
    for r, sd in enumerate(seeds):
        chi = chis[sd]
        for j in range(k):
            sc = _scatter(axes[r, j], coords[ss], chi[ss, j], cmap=cmap, vsym=vsym,
                          vmin=vlim[0], vmax=vlim[1], deg=deg)
            axes[r, j].set_title(f"seed{sd} {lab}{j+1} (SD={chi[:, j].std():.2f})", fontsize=8)
            axes[r, j].set_xlabel(xl); axes[r, j].set_ylabel(yl)
            plt.colorbar(sc, ax=axes[r, j], fraction=0.046)
        axes[r, 0].set_ylabel(f"seed{sd}\n{yl}", fontsize=8)
    kind = "memberships" if membership else "eigenfunctions"
    fig.suptitle(f"{harness.LABELS.get(method, method)} — {kind}  "
                 f"(Hungarian |r| seed0={harness.shape_r(chis[seeds[0]], refs)[0]:.3f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def loss_curves(results_by_seed, method, ax=None):
    """Train vs held-out loss (GramSchmidt self-consistency residual) over seeds."""
    seeds = sorted(results_by_seed)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    allv = []
    for sd in seeds:
        r = results_by_seed[sd]
        lt = r.get("loss_train"); lv = r.get("loss_val")
        if lt is not None and np.isfinite(lt).any():
            ax.plot(lt, color="tab:blue", alpha=0.5, lw=1, label="train" if sd == seeds[0] else None)
            allv.append(lt[np.isfinite(lt)])
        if lv is not None and np.isfinite(lv).any():
            ax.plot(lv, color="tab:red", alpha=0.5, lw=1, label="held-out" if sd == seeds[0] else None)
            allv.append(lv[np.isfinite(lv)])
    # VAMP's loss is −VAMP-2 (negative) → log axis would drop it; use linear when non-positive
    pos = len(allv) and np.concatenate(allv).min() > 0
    ax.set_yscale("log" if pos else "linear")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss" if pos else "loss (−VAMP-2 for VAMP)")
    ax.set_title(harness.LABELS.get(method, method), fontsize=9); ax.legend(fontsize=7)
    return fig


def plot_warmup(system, seeds, deg=None, max_pts=None):
    """Verify the shared 1-D ShiftScale committor warm-up converged: per-seed χ map
    (top row) + train/held-out warm-up loss (bottom). Every ISOKANN method on this
    system is initialised from this χ, so it must have converged to the dominant
    slow coordinate."""
    coords = system["coords"]; tag = system["tag"]
    deg = tag.startswith("adp") if deg is None else deg
    xl, yl = axis_labels(system)
    ss = _subsample(len(coords), max_pts or 30000)
    ns = len(seeds)
    fig, axes = plt.subplots(2, ns, figsize=(3.3 * ns, 6.2), squeeze=False)
    for c, sd in enumerate(seeds):
        chi1d, loss = harness.load_warmup_artifacts(tag, sd)
        axc, axl = axes[0, c], axes[1, c]
        if chi1d is not None:
            X = coords[ss] * (180/np.pi if deg else 1.0)
            s = axc.scatter(X[:, 0], X[:, 1], c=chi1d[ss], s=4, cmap="coolwarm", rasterized=True)
            axc.set_title(f"seed{sd}  1-D χ (SD={chi1d.std():.2f})", fontsize=8)
            plt.colorbar(s, ax=axc, fraction=0.046)
            axl.plot(loss[0], color="tab:blue", lw=1, label="train")
            axl.plot(loss[1], color="tab:red", lw=1, label="held-out")
            axl.set_yscale("log"); axl.set_xlabel("warm-up iter"); axl.set_ylabel("ShiftScale MSE")
            if c == 0:
                axl.legend(fontsize=7)
        axc.set_xlabel(xl); axc.set_ylabel(yl)
    fig.suptitle(f"{tag} — shared 1-D ShiftScale committor warm-up (init for every method)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def score_frame(system, runs, variants):
    """Per-method scores (mean±SD over seeds):
      eig|r|  = eigenfunction-subspace recovery (the N_STATES-1 eigenfunctions vs ev_refs)
                — the uniform metric for both systems.
      memb|r| = N_STATES memberships vs committors (only where committor_refs exists, i.e. TW).
      k_eff   = live χ of the membership representation (chi_best for membership, PCCA+ for basis).
    """
    import pandas as pd
    ev = system["ev_refs"]; comm = system.get("committor_refs")
    rows = []
    for v in variants:
        rb = runs.get((system["tag"], v), {})
        if not rb:
            continue
        seeds = sorted(rb)
        eigr = [harness.eig_r(rb[s]["chi_best"], ev)[0] for s in seeds]
        kes = [harness.k_eff(harness.to_memberships(rb[s]["chi_best"])) for s in seeds]
        row = dict(method=harness.LABELS.get(v, v), seeds=len(seeds),
                   eig_r_mean=np.mean(eigr), eig_r_sd=np.std(eigr) if len(eigr) > 1 else np.nan)
        if comm is not None:
            mr = [harness.membership_r(rb[s]["chi_best"], comm)[0] for s in seeds]
            row["memb_r_mean"] = np.mean(mr); row["memb_r_sd"] = np.std(mr) if len(mr) > 1 else np.nan
        row["k_eff_mean"] = np.mean(kes)
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").round(3)
