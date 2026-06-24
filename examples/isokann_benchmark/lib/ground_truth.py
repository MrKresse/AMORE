# -*- coding: utf-8 -*-
"""
ground_truth.py — the NUMERICAL reference solutions every method is scored against.

These are computed directly from the simulation data (not from any trained net),
so they are the trusted targets:

  triple_well   empirical basin committors p_A, p_B, p_C (fraction of bursts that
                first land in each well's basin) — the reversible TW reference.
  adp_300k_0p1  right eigenvectors of the discrete (phi,psi) 40x40 transfer
                operator at tau=0.1 ps. EV1=stationary, EV2=phi-flip
                (C7eq/alphaR <-> C7ax), EV3=psi (C7eq <-> alphaR). EV2/EV3 are the
                dominant non-trivial slow CVs the multi-D nets must recover.
  directed_ring NON-reversible 3-well ring: (a) the dominant complex-conjugate
                eigenvector of the transfer operator (Re/Im ~ cos/sin of the ring
                angle), and (b) the forward committor q = P(reach well B before
                well A), solved on a fine angular Markov model built from the same
                bursts. For a system with a probability current the committor's
                level sets are tilted by the drift — the non-reversible signature.
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.linalg import eig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths


# ───────────────────────────── triple well ──────────────────────────────────
def tw_committors():
    """Returns (anchors (N,2), wells (3,2), refs (3,N) = [p_A,p_B,p_C])."""
    z = np.load(os.path.join(paths.DATA, "tw_committor.npz"))
    refs = np.stack([z["p_A"], z["p_B"], z["p_C"]]).astype(np.float64)
    return z["anchors"], z["wells"], refs


def tw_eigvecs(NB: int = 30):
    """Dominant non-trivial transfer-operator eigenvectors EV2, EV3 of the 2D triple-well,
    built from its Koopman bursts on an NBxNB (x,y) grid (the eigenfunction reference,
    analogous to ADP's EV2/EV3). Returns (refs (2,N) mapped to anchors, eigvals, occ-grid)."""
    z = np.load(os.path.join(paths.DATA, "triple_well_koopman.npz"))
    anchors = z["anchors"]; bursts = z["bursts"]; N, Kb, _ = bursts.shape
    lo, hi = anchors.min(0), anchors.max(0)
    edges = [np.linspace(lo[d], hi[d], NB + 1) for d in range(2)]

    def cells(P):
        bx = np.clip(np.digitize(P[:, 0], edges[0]) - 1, 0, NB - 1)
        by = np.clip(np.digitize(P[:, 1], edges[1]) - 1, 0, NB - 1)
        return bx * NB + by

    ci = cells(anchors); C = np.zeros((NB * NB, NB * NB))
    for j in range(Kb):
        np.add.at(C, (ci, cells(bursts[:, j, :])), 1.0)
    rs = C.sum(1, keepdims=True); occ = rs[:, 0] > 0
    T = np.zeros_like(C); T[occ] = C[occ] / rs[occ]; T = 0.999 * T + 1e-3 / (NB * NB)
    v, R = eig(T); o = np.argsort(v.real)[::-1]; v, R = v[o].real, R[:, o].real
    refs = np.stack([R[ci, 1], R[ci, 2]]).astype(np.float64)
    return refs, v, occ


# ───────────────────────── alanine dipeptide operator ───────────────────────
def _build_adp_T(tag: str = "T300_0p1", NB: int = 40):
    """Shared (phi,psi) NBxNB transfer-operator build from the tau-lag pairs (exactly
    as benchmark_v4/score_v4.py). Returns (edges, T, occ)."""
    edges = np.linspace(-np.pi, np.pi, NB + 1)
    NCELL = NB * NB

    def cells(ph, ps):
        return (np.clip(np.digitize(ph, edges) - 1, 0, NB - 1) * NB
                + np.clip(np.digitize(ps, edges) - 1, 0, NB - 1))

    ph0 = np.load(os.path.join(paths.DATA, f"vac_phi0_{tag}.npy"))
    ps0 = np.load(os.path.join(paths.DATA, f"vac_psi0_{tag}.npy"))
    pht = np.load(os.path.join(paths.DATA, f"vac_phitau_{tag}.npy"))
    pst = np.load(os.path.join(paths.DATA, f"vac_psitau_{tag}.npy"))
    a, b = cells(ph0, ps0), cells(pht, pst)
    C = np.zeros((NCELL, NCELL)); np.add.at(C, (a, b), 1.0)
    rs = C.sum(1, keepdims=True); occ = rs[:, 0] > 0
    T = np.zeros_like(C); T[occ] = C[occ] / rs[occ]
    T = 0.999 * T + 1e-3 / NCELL
    return edges, T, occ


def adp_transfer_operator(tag: str = "T300_0p1", NB: int = 40):
    """Right eigenvectors of the (phi,psi) transfer operator: returns
    (edges, eigvals (real, sorted desc), R right-eigvecs (NCELL,*), occ mask)."""
    edges, T, occ = _build_adp_T(tag, NB)
    v, R = eig(T)
    o = np.argsort(v.real)[::-1]
    return edges, v[o].real, R[:, o].real, occ


def adp_stationary(tag: str = "T300_0p1", NB: int = 40):
    """Stationary distribution pi (LEFT Perron eigenvector of the operator, lambda=1),
    normalised to sum 1 over occupied cells. This is the genuine 'EV1 stationary' — it
    spans orders of magnitude (basins high, barriers tiny), so it is best viewed on a
    log scale. Returns (edges, pi (NCELL,), occ)."""
    edges, T, occ = _build_adp_T(tag, NB)
    w, L = eig(T.T)
    i0 = int(np.argmin(np.abs(w - 1.0)))
    pi = np.abs(L[:, i0].real)
    pi[~occ] = 0.0
    s = pi.sum()
    return edges, (pi / s if s > 0 else pi), occ


def adp_eigvec_refs(phi, psi, tag: str = "T300_0p1", NB: int = 40):
    """Map EV2 (phi), EV3 (psi) of the tau-lag operator onto anchor (phi,psi).
    Returns (refs (2,n), eigvals, edges, R, occ)."""
    edges, evals, R, occ = adp_transfer_operator(tag, NB)
    ci = (np.clip(np.digitize(phi, edges) - 1, 0, NB - 1) * NB
          + np.clip(np.digitize(psi, edges) - 1, 0, NB - 1))
    refs = np.stack([R[ci, 1], R[ci, 2]]).astype(np.float64)
    return refs, evals, edges, R, occ


# ───────────────────────────── directed ring ────────────────────────────────
_RING_CENTERS = np.array([np.pi / 3, np.pi, 5 * np.pi / 3])


def _ring_angle(xy):
    return np.arctan2(xy[..., 1], xy[..., 0]) % (2 * np.pi)


def ring_committor(system, well_A: int = 0, well_B: int = 1, M: int = 72):
    """Forward committor q = P(reach basin of well_B before basin of well_A),
    solved on an M-bin angular Markov model built from the ring's own bursts.

    Non-reversible signature: with the tangential current the committor rises
    preferentially in the drift direction, so its 0.5 level sits off the
    geometric midpoint between the two wells.

    Returns q over anchors (N,) in [0,1] (NaN inside the absorbing basins).
    """
    feat = system["feat"]; bursts = system["bursts"]          # (N,2),(N,K,2)
    th_a = _ring_angle(feat)                                   # (N,)
    th_b = _ring_angle(bursts)                                 # (N,K)
    edges = np.linspace(0, 2 * np.pi, M + 1)

    def to_bin(th):
        return np.clip(np.digitize(th, edges) - 1, 0, M - 1)

    ba = to_bin(th_a); bb = to_bin(th_b)
    C = np.zeros((M, M))
    for j in range(bb.shape[1]):
        np.add.at(C, (ba, bb[:, j]), 1.0)
    rs = C.sum(1, keepdims=True); occ = rs[:, 0] > 0
    T = np.zeros_like(C); T[occ] = C[occ] / rs[occ]

    # absorbing target bins = the two wells' angular bins
    binA = int(to_bin(np.array([_RING_CENTERS[well_A]]))[0])
    binB = int(to_bin(np.array([_RING_CENTERS[well_B]]))[0])
    absorbing = {binA, binB}
    trans = np.array([i for i in range(M) if i not in absorbing and occ[i]])
    # q on transient bins: (I - T_tt) q = T_tB  (committor BC: q(A)=0, q(B)=1)
    Tt = T[np.ix_(trans, trans)]
    rhs = T[trans, binB].copy()
    q_bin = np.full(M, np.nan)
    q_bin[binB] = 1.0; q_bin[binA] = 0.0
    try:
        q_bin[trans] = np.linalg.solve(np.eye(len(trans)) - Tt, rhs)
    except np.linalg.LinAlgError:
        q_bin[trans] = np.linalg.lstsq(np.eye(len(trans)) - Tt, rhs, rcond=None)[0]
    q = q_bin[to_bin(th_a)]
    return q, dict(edges=edges, q_bin=q_bin, binA=binA, binB=binB, T=T, occ=occ)


def ring_complex_ev(system):
    """Dominant complex-conjugate transfer-operator pair from the bundled 3-state
    matrix (Re/Im over anchors). Returns (refs (2,N), eigs)."""
    return system["refs"], system.get("eigs", None)
