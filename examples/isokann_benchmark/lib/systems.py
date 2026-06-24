# -*- coding: utf-8 -*-
"""
systems.py — uniform loaders for the benchmark systems (consolidates the data
interface used by both notebooks). Every loader returns the same dict so the
training harness is system-agnostic:

    feat      : (N, F)      anchor features
    bursts    : (N, K, F)   K Koopman-burst endpoints per anchor (same featurisation)
    coords    : (N, 2)      2D plotting coords (xy for TW/ring, (phi,psi) for ADP)
    refs      : (R, N)      continuous numerical reference shapes (scored via |r|)
    ref_names : list[str]
    labels    : (N,) | None discrete basin labels (ring only; AUROC scoring)
    reversible: bool
    tag       : str

Systems:
  triple_well    reversible 2D triple-well       — refs = committors p_A,p_B,p_C
  adp_300k_0p1   reversible vacuum ADP, 0.1 ps   — refs = operator EV2 (phi), EV3 (psi)
  directed_ring  NON-reversible 3-well ring       — refs = Re/Im of dominant complex EV
                                                    (+ committor available via ground_truth)
Large raw data is read from paths.DATA (scratch); generate it with generate_data.py
(TW, ADP) — the ring is pure-numpy and is simulated/cached here on first use.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import ground_truth as gt

_PAIRS = np.array([(i, j) for i in range(22) for j in range(i + 1, 22)])  # 231


# ───────────────────────────── triple well ──────────────────────────────────
def load_triple_well(max_anchors: int | None = None, seed: int = 0) -> dict:
    tw = np.load(os.path.join(paths.DATA, "triple_well_koopman.npz"))
    _, _, refs = gt.tw_committors()
    ev_refs, _, _ = gt.tw_eigvecs()
    feat = tw["anchors"].astype(np.float32)
    bursts = tw["bursts"].astype(np.float32)
    idx = np.arange(len(feat))
    if max_anchors and max_anchors < len(feat):
        idx = np.random.default_rng(seed).choice(len(feat), max_anchors, replace=False)
    return dict(tag="triple_well", reversible=True,
                feat=feat[idx], bursts=bursts[idx], coords=feat[idx],
                refs=refs[:, idx], ref_names=["p_A", "p_B", "p_C"],
                ev_refs=ev_refs[:, idx], ev_names=["EV2", "EV3"],
                committor_refs=refs[:, idx], labels=None,
                patch_splits=tw["patch_splits"][:, idx] if "patch_splits" in tw else None)


# ───────────────────────── alanine dipeptide 0.1 ps ─────────────────────────
def _featurize_adp(coords: np.ndarray) -> np.ndarray:
    x = coords.reshape(len(coords), 22, 3)
    d = x[:, _PAIRS[:, 0], :] - x[:, _PAIRS[:, 1], :]
    return np.linalg.norm(d, axis=2).astype(np.float32)


def load_adp_300k_0p1(max_anchors: int = 25000, seed: int = 0) -> dict:
    tag = "T300_0p1"
    X0 = np.load(os.path.join(paths.DATA, f"vac_X0_{tag}.npy"))
    Xt = np.load(os.path.join(paths.DATA, f"vac_Xtau_{tag}.npy"))
    phi = np.load(os.path.join(paths.DATA, f"vac_phi0_{tag}.npy"))
    psi = np.load(os.path.join(paths.DATA, f"vac_psi0_{tag}.npy"))
    N = len(X0)
    idx = np.arange(N)
    if max_anchors and max_anchors < N:
        idx = np.random.default_rng(seed).choice(N, max_anchors, replace=False)
    feat = _featurize_adp(X0[idx])
    bursts = _featurize_adp(Xt[idx])[:, None, :]   # K=1
    phi, psi = phi[idx], psi[idx]
    refs, evals, _, _, _ = gt.adp_eigvec_refs(phi, psi, tag)
    return dict(tag="adp_300k_0p1", reversible=True,
                feat=feat, bursts=bursts, coords=np.stack([phi, psi], 1),
                refs=refs, ref_names=["EV2 (phi-flip)", "EV3 (psi)"],
                ev_refs=refs, ev_names=["EV2 (phi-flip)", "EV3 (psi)"],
                committor_refs=None, labels=None, eigvals=evals)


# ───────────────────────────── directed ring ────────────────────────────────
_RING_CENTERS = np.array([np.pi / 3, np.pi, 5 * np.pi / 3])


def _ring_force(x, a, b, R, kappa):
    r = np.hypot(x[:, 0], x[:, 1]) + 1e-9
    th = np.arctan2(x[:, 1], x[:, 0]); s3 = np.sin(3 * th)
    gx = 2 * a * (r - R) * x[:, 0] / r + 3 * b * s3 * x[:, 1] / r**2
    gy = 2 * a * (r - R) * x[:, 1] / r - 3 * b * s3 * x[:, 0] / r**2
    fx, fy = -x[:, 1] / r, x[:, 0] / r
    return np.stack([-gx + kappa * fx, -gy + kappa * fy], 1)


def _ring_propagate(x, steps, dt, D, a, b, R, kappa, rng):
    sq = np.sqrt(2 * D * dt)
    for _ in range(steps):
        x = x + _ring_force(x, a, b, R, kappa) * dt + sq * rng.standard_normal(x.shape)
    return x


def _ring_basin(x):
    th = np.arctan2(x[..., 1], x[..., 0])
    d = np.abs((th[..., None] - _RING_CENTERS + np.pi) % (2 * np.pi) - np.pi)
    return d.argmin(-1)


def simulate_directed_ring(n_anchors: int = 2500, K: int = 8, burst: int = 250,
                           a: float = 6.0, b: float = 1.2, R: float = 1.0,
                           D: float = 0.25, dt: float = 2e-3, kappa: float = 2.0,
                           seed: int = 0, cache: bool = True) -> dict:
    """Overdamped Langevin on a 3-well ring + tangential drift (kappa breaks
    detailed balance -> directional current and a complex-conjugate operator pair)."""
    cpath = os.path.join(paths.DATA, f"ring_n{n_anchors}_K{K}_L{burst}_k{kappa}_b{b}_s{seed}.npz")
    if cache and os.path.exists(cpath):
        z = np.load(cpath)
        out = {k: (z[k] if k != "tag" else str(z[k])) for k in z.files}
        out.update(reversible=False, ref_names=["Re v2 (~cos)", "Im v2 (~sin)"],
                   ev_refs=out["refs"], ev_names=["Re v2 (~cos)", "Im v2 (~sin)"], committor_refs=None)
        return out
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi, n_anchors); r0 = R + 0.06 * rng.standard_normal(n_anchors)
    anch = np.stack([r0 * np.cos(th0), r0 * np.sin(th0)], 1)
    anch = _ring_propagate(anch, 40, dt, D, a, b, R, kappa, rng)
    ends = np.stack([_ring_propagate(anch.copy(), burst, dt, D, a, b, R, kappa, rng)
                     for _ in range(K)], 1)
    la = _ring_basin(anch); le = _ring_basin(ends)
    T = np.zeros((3, 3))
    for j in range(K):
        np.add.at(T, (la, le[:, j]), 1.0)
    T = T / T.sum(1, keepdims=True)
    w, V = np.linalg.eig(T); order = np.argsort(-np.abs(w)); w, V = w[order], V[:, order]
    v2 = V[:, 1]
    refs = np.stack([np.real(v2)[la], np.imag(v2)[la]]).astype(np.float64)
    out = dict(tag="directed_ring", feat=anch.astype(np.float32),
               bursts=ends.astype(np.float32), coords=anch.astype(np.float32),
               refs=refs, labels=la.astype(np.int64), T=T, eigs=w.astype(np.complex128))
    if cache:
        np.savez(cpath, **{k: v for k, v in out.items() if k != "tag"}, tag=out["tag"])
    out.update(reversible=False, ref_names=["Re v2 (~cos)", "Im v2 (~sin)"],
               ev_refs=out["refs"], ev_names=["Re v2 (~cos)", "Im v2 (~sin)"], committor_refs=None)
    return out


LOADERS = {
    "triple_well": load_triple_well,
    "adp_300k_0p1": load_adp_300k_0p1,
    "directed_ring": simulate_directed_ring,
}
