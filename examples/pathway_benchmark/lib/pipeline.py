# -*- coding: utf-8 -*-
"""
pipeline.py — helpers for the AMORE pathway benchmark notebook.

Thin orchestration over src/amore: ISOKANN training, seed preparation
(minimise + orthogonal-equilibration, with the optional edge-activity hold),
parallel pathway ensembles via the src simplex entry points, path-metric
clustering, robust PMF, sensitivity, and plotting.  No reimplementation of the
level-set integrators or constraints — those live in amore.mep.
"""
from __future__ import annotations
import os, sys, time, contextlib, io
import numpy as np
import torch as pt
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from amore.isokann import ChiNetMulti
from amore.isotarget import isa_target, gramschmidt_target
from amore.features import make_featurizer
from amore.chi import chi_sensitivity
from amore.sims import OpenMMSimulation, phi as adp_phi, psi as adp_psi
from amore.mep import (
    reaction_path_face, reaction_path_edge, mfep_face, mfep_edge,
    separatrix_frames, FaceCV, EdgeCV, ActivityCV,
    energy_min_on_levelset, sample_levelset_projected,
)
from amore.mep.core import _chi_val

N_ATOMS = 22
PAIRS = np.array([(i, j) for i in range(N_ATOMS) for j in range(i + 1, N_ATOMS)])  # 231
_FEAT = make_featurizer(PAIRS)


def featurizer(x_t):
    return _FEAT(x_t)


def featurize_np(coords_np, device="cpu"):
    return _FEAT(pt.from_numpy(np.asarray(coords_np, np.float32)).to(device))


# ─────────────────────────── k=3 softmax ISA training ────────────────────────
def _gs_residual(chi_kn, kchi_kn):
    try:
        tgt = gramschmidt_target(kchi_kn)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(np.mean((chi_kn - tgt) ** 2))


def train_isokann_isa(feat, bursts, k=3, hidden=(128, 32, 8), seed=0,
                      max_iter=2500, min_iter=600, lr=1e-3, grad_clip=5.0,
                      plateau_w=200, rel_tol=1e-3, device="cpu", verbose=True):
    """k softmax memberships via the ISA isotarget, no warm-up (benchmark gold standard)."""
    pt.manual_seed(seed * 12345 + 7); np.random.seed(seed * 12345 + 7)
    feat = np.asarray(feat, np.float32); bursts = np.asarray(bursts, np.float32)
    N, Kb, IN = bursts.shape
    f0 = pt.tensor(feat, device=device)
    fts = [pt.tensor(bursts[:, j, :], device=device) for j in range(Kb)]
    m = np.random.rand(N) < 0.8; tr, te = np.where(m)[0], np.where(~m)[0]
    net = ChiNetMulti(IN, k, hidden=list(hidden)).to(device)
    opt = pt.optim.Adam(net.parameters(), lr=lr)

    def kchi(sub):
        net.eval()
        with pt.no_grad():
            return np.mean([net(ft[sub]).cpu().numpy().T for ft in fts], axis=0)

    loss_tr, loss_val = [], []
    best_val, best_chi = np.inf, None
    t0 = time.perf_counter()
    for it in range(max_iter):
        net.eval()
        with pt.no_grad():
            chi0 = net(f0[tr]).cpu().numpy().T
        kc = kchi(tr)
        try:
            tgt = isa_target(chi0, kc)
        except (ValueError, np.linalg.LinAlgError):
            loss_tr.append(np.nan); loss_val.append(np.nan); continue
        tt = pt.tensor(tgt.T, dtype=pt.float32, device=device)
        net.train(); loss = nn.functional.mse_loss(net(f0[tr]), tt)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), grad_clip); opt.step()
        net.eval()
        with pt.no_grad():
            chi_tr = net(f0[tr]).cpu().numpy().T
            chi_te = net(f0[te]).cpu().numpy().T
        loss_tr.append(_gs_residual(chi_tr, kchi(tr)))
        val = _gs_residual(chi_te, kchi(te)); loss_val.append(val)
        if np.isfinite(val) and val < best_val:
            with pt.no_grad():
                best_val, best_chi = val, net(f0).cpu().numpy().copy()
        if it >= min_iter and len(loss_val) >= plateau_w:
            r = np.array(loss_val[-plateau_w:]); r = r[np.isfinite(r)]
            if len(r) >= plateau_w // 2 and (r.max() - r.min()) < rel_tol * max(abs(np.median(r)), 1e-12):
                break
    if best_chi is None:
        with pt.no_grad():
            best_chi = net(f0).cpu().numpy()
    net.eval()
    if verbose:
        print(f"[train] iters={len(loss_val)} k_eff={int((best_chi.std(0) > 0.05).sum())} "
              f"t={time.perf_counter()-t0:.0f}s", flush=True)
    return dict(net=net, chi=best_chi.astype(np.float32),
                loss_train=np.array(loss_tr, np.float32),
                loss_val=np.array(loss_val, np.float32), n_iter=len(loss_val))


# ─────────────────────────── state naming ───────────────────────────────────
def state_centroids(chi, phi, psi, top_frac=0.02):
    out = []
    n_top = max(10, int(top_frac * len(chi)))
    for j in range(chi.shape[1]):
        idx = np.argsort(chi[:, j])[-n_top:]
        out.append((np.angle(np.mean(np.exp(1j * phi[idx]))),
                    np.angle(np.mean(np.exp(1j * psi[idx])))))
    return np.array(out)


def basin_name(cph, cps):
    if cph > 0.3:
        return "C7ax"
    return "αR" if cps < 0.0 else "C7eq"


# ─────────────────────── OpenMM potential / force ───────────────────────────
def make_potential(sim):
    from openmm import unit
    ctx, nat = sim._sim.context, sim.n_atoms

    def pot(x):
        ctx.setPositions(x.reshape(nat, 3))
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    def grd(x):
        ctx.setPositions(x.reshape(nat, 3))
        f = (ctx.getState(getForces=True).getForces(asNumpy=True)
             .value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
        return -f.flatten().astype(np.float64)

    return pot, grd


# ─────────────────────── seed preparation (minimise + equilibrate) ──────────
def _make_cv(model, view, i, j):
    return FaceCV(model, i) if view == "face" else EdgeCV(model, i, j)


def prepare_seed(model, view, i, j, x0, pot, grd, sim, min_iter=60, equil_steps=60):
    """LIGHT seed prep: a short energy minimisation on the χ_i (face) or s (edge) level
    set, then a short orthogonal-sampling equilibration for thermal spread.  No activity
    constraint — the membership level set χ_i=½ carries no basin (basins sit at χ_i∈{0,1}),
    and even the edge s=½ only meets the third basin at the saddle, so a *light* prep does
    not slide into it.  Keep min_iter modest: aggressive minimisation at s≈½ is what
    finds the third basin."""
    cv = _make_cv(model, view, i, j)
    x = np.asarray(x0, np.float64).flatten()
    s0 = _chi_val(cv, featurizer, x)
    xm = energy_min_on_levelset(cv, featurizer, pot, x, s0, grad_fn=grd, max_iter=min_iter)
    if equil_steps > 0:
        res = sample_levelset_projected(sim, cv, featurizer, xm, s0, equil_steps)
        xm = res["positions"][-1]
    return xm


# ───────────────────────── parallel pathway ensemble ────────────────────────
_WK: dict = {}


def _worker_init(state_dict, in_dim, k, hidden, temp):
    pt.set_num_threads(1)
    model = ChiNetMulti(in_dim, k, hidden=list(hidden))
    model.load_state_dict(state_dict); model.eval()
    sim = OpenMMSimulation(steps=1, dt=2e-3, temp=temp)
    pot, grd = make_potential(sim)
    _WK.update(model=model, sim=sim, pot=pot, grd=grd)


def _worker_task(task):
    """task = (gid, method, view, i, j, x0, kw).  method ∈ {prep, mep, mfep, ef};
    view ∈ {face, edge}; j is ignored for a face."""
    gid, method, view, i, j, x0, kw = task
    model, sim, pot, grd = _WK["model"], _WK["sim"], _WK["pot"], _WK["grd"]
    x0 = np.asarray(x0, np.float64).flatten()
    with contextlib.redirect_stdout(io.StringIO()):
        if method == "prep":
            out = prepare_seed(model, view, i, j, x0, pot, grd, sim, **kw)
        elif method == "mep":
            if view == "edge":
                out = reaction_path_edge(model, i, j, x0, featurizer, potential_fn=pot,
                                         grad_fn=grd, **kw)
            else:
                out = reaction_path_face(model, i, x0, featurizer, potential_fn=pot,
                                         grad_fn=grd, **kw)
        elif method in ("mfep", "ef"):
            if view == "edge":
                out = mfep_edge(sim, model, i, j, x0, featurizer, **kw)
            else:
                out = mfep_face(sim, model, i, x0, featurizer, **kw)
        else:
            raise ValueError(method)
    return gid, out


def run_ensemble(net, in_dim, k, hidden, tasks, procs=12, temp=300.0):
    """Run pathway/prep tasks across `procs` spawn workers (autograd-safe).  tasks:
    list of (gid, method, view, i, j, x0, kw).  Returns [(gid, result)]."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    sd = {kk: v.detach().cpu() for kk, v in net.state_dict().items()}
    out = []
    with ctx.Pool(procs, initializer=_worker_init,
                  initargs=(sd, in_dim, k, list(hidden), temp)) as pool:
        for gid, res in pool.imap_unordered(_worker_task, tasks):
            out.append((gid, res))
    return out


def path_coords(res):
    return np.array(res["images"]) if isinstance(res, dict) else np.asarray(res)


def edge_of_path(model, coords, i, device="cpu"):
    """A-posteriori edge realisation (robust, whole-path).  A membership-i path runs from
    χ_i≈0 ('the rest') toward χ_i≈1 (state i); the **partner** state is the one being
    converted — i.e. the non-i membership that *decreases* as χ_i increases (most
    anti-correlated with χ_i along the path), while the off-edge state stays ≈flat near 0.
    This is far more stable than reading the single χ_i≈0 endpoint, whose argmax flips when
    that end is ambiguous.  Falls back to mean membership if χ_i barely varies.  Returns a
    sorted (a,b) tuple."""
    coords = np.asarray(coords)
    with pt.no_grad():
        chi = model(featurize_np(coords, device)).numpy()       # (N, k)
    ci = chi[:, i]
    others = [m for m in range(chi.shape[1]) if m != i]
    if ci.std() < 1e-4:                                          # χ_i ~ constant
        partner = int(max(others, key=lambda m: chi[:, m].mean()))
    else:
        cor = {m: np.corrcoef(ci, chi[:, m])[0, 1] for m in others}
        # most anti-correlated = the converted partner; ties broken toward higher presence
        partner = int(min(others, key=lambda m: (cor[m], -chi[:, m].mean())))
    return tuple(sorted((i, partner)))


# ───────────────────────── robust PMF (interior of MFEP) ─────────────────────
def robust_pmf(res, temp, z_frac=0.05):
    kBT = 0.008314463 * temp
    cv = np.asarray(res["cv_values"], float)
    lam = np.asarray(res["mean_forces"], float)
    Zmean = np.array([np.mean(r["Zs"]) for r in res["results"]])
    inv_sqrtZ = np.array([np.mean(1.0 / np.sqrt(np.clip(r["Zs"], 1e-12, None)))
                          for r in res["results"]])
    o = np.argsort(cv)
    cv, lam, Zmean, inv_sqrtZ = cv[o], lam[o], Zmean[o], inv_sqrtZ[o]
    good = np.isfinite(lam) & (Zmean >= z_frac * np.nanmax(Zmean))
    F = np.full(len(cv), np.nan)
    if good.sum() >= 2:
        cg, lg, ig = cv[good], lam[good], inv_sqrtZ[good]
        Frigid = np.concatenate([[0.0], np.cumsum(0.5 * (lg[1:] + lg[:-1]) * np.diff(cg))])
        Ffree = Frigid - kBT * np.log(ig); Ffree -= np.nanmin(Ffree)
        F[good] = Ffree
    return dict(cv=cv, F_free=F, F_dagger=float(np.nanmax(F)) if np.isfinite(F).any() else np.nan)


# ───────────────────────── path-metric clustering (energy-free tubes) ────────
def _resample_phipsi(path, cv, grid):
    ph, ps = adp_phi(path), adp_psi(path)
    order = np.argsort(cv); s = cv[order]
    feats = []
    for ang in (ph, ps):
        a = ang[order]
        feats.append(np.interp(grid, s, np.cos(a)))
        feats.append(np.interp(grid, s, np.sin(a)))
    return np.stack(feats, 1)


def path_distance_matrix(paths, cv, n_grid=40):
    from scipy.spatial.distance import pdist, squareform
    grid = np.linspace(0.0, 1.0, n_grid)
    feats = np.array([_resample_phipsi(p, c, grid).ravel() for p, c in zip(paths, cv)])
    return squareform(pdist(feats / np.sqrt(n_grid)))


def cluster_paths(paths, cv, n_grid=40, thresh=0.8, min_size=1):
    """Cluster paths into distinct tubes by the level-set-aligned (phi,psi) metric.

    `thresh` is an ABSOLUTE distance in that metric, which is ≈ the RMS angular
    separation between two paths in (phi,psi) [rad] — so routes that differ by more than
    ~`thresh` rad become separate tubes.  This is robust to imbalanced route sizes; a
    relative (median-scaled) threshold over-segments the dominant route and merges the
    minor ones, which is why parallel channels were getting lost.  Clusters smaller than
    `min_size` are dropped as outliers when choosing medoids.  Returns (labels, medoids).
    """
    if len(paths) == 1:
        return np.array([1]), [0]
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = path_distance_matrix(paths, cv, n_grid)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=thresh, criterion="distance")
    medoids = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < min_size:
            continue
        sub = D[np.ix_(idx, idx)]
        medoids.append(int(idx[np.argmin(sub.mean(1))]))
    return labels, medoids


# ───────────────────────────── plotting / sensitivity ───────────────────────
def plot_path(ax, coords, sort_cv=None, halo=True, arrow=False, break_thresh=1.8, **kw):
    """Overlay a path on the (phi,psi) plane, breaking the line at any large
    DISCONTINUITY (default Δ>1.8 rad in φ or ψ between consecutive images).  This catches
    both ±π periodic wraps (Δ≈2π) and bad outlier images at the extreme level sets (a
    medoid that landed in a different basin, Δ≈3 rad but just under π) — both otherwise
    draw a spurious straight line across the plot.  Normal adjacent images are <~1 rad
    apart, so legitimate path is untouched.  `halo` draws a white outline (cr2 style);
    `arrow` adds an arrowhead at the χ=1 end."""
    import matplotlib.patheffects as pe
    ph, ps = np.asarray(adp_phi(coords), float), np.asarray(adp_psi(coords), float)
    if sort_cv is not None:
        o = np.argsort(np.asarray(sort_cv)); ph, ps = ph[o], ps[o]
    brk = np.where((np.abs(np.diff(ph)) > break_thresh) | (np.abs(np.diff(ps)) > break_thresh))[0]
    ph = np.insert(ph, brk + 1, np.nan); ps = np.insert(ps, brk + 1, np.nan)
    if halo:
        lw = kw.get("lw", kw.get("linewidth", 1.2))
        kw.setdefault("path_effects",
                      [pe.Stroke(linewidth=lw + 1.6, foreground="white", alpha=0.85), pe.Normal()])
        kw.setdefault("solid_capstyle", "round")
    line = ax.plot(ph, ps, **kw)
    if arrow:
        good = ~np.isnan(ph)
        xs, ys = ph[good], ps[good]
        if len(xs) >= 2:
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=kw.get("color", "k"),
                                        lw=kw.get("lw", 1.2)), zorder=kw.get("zorder", 5))
    return line


def fes_along_path(fes, axis, coords):
    from scipy.interpolate import RegularGridInterpolator
    ax = np.asarray(axis, float); full = np.append(ax, np.pi)
    fes_p = np.vstack([fes, fes[:1]]); fes_p = np.hstack([fes_p, fes_p[:, :1]])
    interp = RegularGridInterpolator((full, full), fes_p, bounds_error=False, fill_value=None)
    ph = (adp_phi(coords) + np.pi) % (2 * np.pi) - np.pi
    ps = (adp_psi(coords) + np.pi) % (2 * np.pi) - np.pi
    return interp(np.stack([ps, ph], 1))


def direction_sensitivity(cv, coords, nbins=1, device="cpu", max_pts=2000, seed=0):
    """Per-atom <|grad_x s|^2> of a CV (FaceCV/EdgeCV), optionally binned by s."""
    coords = np.asarray(coords, np.float32)
    if len(coords) > max_pts:
        coords = coords[np.random.default_rng(seed).choice(len(coords), max_pts, replace=False)]
    xs = pt.tensor(coords, device=device)
    centers, sens = chi_sensitivity(cv, featurizer, xs, nbins=nbins)
    return centers.detach().cpu().numpy(), sens.detach().cpu().numpy()
