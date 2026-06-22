# -*- coding: utf-8 -*-
"""
toy_systems.py — uniform loaders for the three benchmark systems.

Each loader returns a dict with the SAME interface so the training harness
(train_eval.py) is system-agnostic:

    feat     : (N, F)        anchor features
    bursts   : (N, K, F)     K Koopman-burst endpoints per anchor (same featurization)
    coords   : (N, 2)        2D coordinates for plotting (xy for TW/ring, (phi,psi) for ADP)
    refs     : (R, N)        continuous reference shape functions (shape scoring, |r|)
    ref_names: list[str]
    labels   : (N,) or None  discrete ground-truth state labels (AUROC scoring; ring only)
    reversible: bool         whether the system obeys detailed balance
    tag      : str

Systems
-------
  triple_well   reversible 2D triple-well (benchmark_v3 data) — refs = committors p_A,p_B,p_C
  adp_300k_0p1  reversible vacuum alanine dipeptide, 300 K, tau=0.1 ps (benchmark_v4 data)
                231 pairwise heavy-atom distances; refs = 0.1 ps transfer-operator EV2 (phi),
                EV3 (psi)
  directed_ring NON-REVERSIBLE toy: overdamped Langevin on a 3-well ring with a tangential,
                non-conservative drift -> a probability current A->B->C->A and a
                complex-conjugate pair in the dominant transfer-operator subspace.
                refs = real/imag of the dominant complex eigenvector (~cos/sin of the ring
                angle); labels = nearest-well basin (0/1/2).
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data")
os.makedirs(CACHE, exist_ok=True)

# All reference data is BUNDLED in ./data so this benchmark is fully standalone — the
# benchmark / benchmark_v2 / benchmark_v3 / benchmark_v4 source directories can be deleted.
#   triple_well_koopman.npz   (TW anchors+bursts)        <- benchmark/data
#   tw_committor.npz          (TW FEM committors p_A/B/C) <- benchmark_v2/panel0
#   vac_{X0,Xtau,phi0,psi0,phitau,psitau}_T300_0p1.npy    <- benchmark_v4/data
#   ring_*.npz                (directed-ring cache, self-generated)
BENCH_DATA = V2_PANEL0 = V4_DATA = CACHE


# --------------------------------------------------------------------------- #
# 1. reversible triple-well (benchmark_v3)
# --------------------------------------------------------------------------- #
def load_triple_well(max_anchors: int | None = None, seed: int = 0) -> dict:
    tw = np.load(os.path.join(BENCH_DATA, "triple_well_koopman.npz"))
    cm = np.load(os.path.join(V2_PANEL0, "tw_committor.npz"))
    feat   = tw["anchors"].astype(np.float32)          # (1600, 2)
    bursts = tw["bursts"].astype(np.float32)           # (1600, 20, 2)
    refs   = np.stack([cm["p_A"], cm["p_B"], cm["p_C"]]).astype(np.float64)  # (3, 1600)
    idx = np.arange(len(feat))
    if max_anchors and max_anchors < len(feat):
        idx = np.random.default_rng(seed).choice(len(feat), max_anchors, replace=False)
    return dict(
        tag="triple_well", reversible=True,
        feat=feat[idx], bursts=bursts[idx], coords=feat[idx],
        refs=refs[:, idx], ref_names=["p_A", "p_B", "p_C"],
        labels=None,
    )


# --------------------------------------------------------------------------- #
# 2. reversible alanine dipeptide, 300 K, 0.1 ps (benchmark_v4)
# --------------------------------------------------------------------------- #
_PAIRS = np.array([(i, j) for i in range(22) for j in range(i + 1, 22)])  # 231

def _featurize_adp(coords: np.ndarray) -> np.ndarray:
    x = coords.reshape(len(coords), 22, 3)
    d = x[:, _PAIRS[:, 0], :] - x[:, _PAIRS[:, 1], :]
    return np.linalg.norm(d, axis=2).astype(np.float32)

def _adp_reference(tag: str = "T300_0p1", NB: int = 40):
    """Right eigenvectors of the (phi,psi) 40x40 transfer operator at lag `tag`, built
    fresh from the transition pairs exactly as benchmark_v4/score_v4.py does (the
    precomputed vac_transfer_eigvecs file uses a different normalization and does NOT
    match). Returns (edges, R) with R[:,1]=EV2 (phi-flip), R[:,2]=EV3 (psi)."""
    from scipy.linalg import eig
    edges = np.linspace(-np.pi, np.pi, NB + 1)
    NCELL = NB * NB
    def cells(ph, ps):
        return (np.clip(np.digitize(ph, edges) - 1, 0, NB - 1) * NB
                + np.clip(np.digitize(ps, edges) - 1, 0, NB - 1))
    ph0 = np.load(os.path.join(V4_DATA, f"vac_phi0_{tag}.npy"))
    ps0 = np.load(os.path.join(V4_DATA, f"vac_psi0_{tag}.npy"))
    pht = np.load(os.path.join(V4_DATA, f"vac_phitau_{tag}.npy"))
    pst = np.load(os.path.join(V4_DATA, f"vac_psitau_{tag}.npy"))
    a, b = cells(ph0, ps0), cells(pht, pst)
    C = np.zeros((NCELL, NCELL)); np.add.at(C, (a, b), 1.0)
    rs = C.sum(1, keepdims=True); occ = rs[:, 0] > 0
    T = np.zeros_like(C); T[occ] = C[occ] / rs[occ]; T = 0.999 * T + 1e-3 / NCELL
    v, R = eig(T)
    o = np.argsort(v.real)[::-1]
    return edges, R[:, o].real, v[o].real

def load_adp_300k_0p1(max_anchors: int = 12000, seed: int = 0) -> dict:
    tag = "T300_0p1"
    X0  = np.load(os.path.join(V4_DATA, f"vac_X0_{tag}.npy"))
    Xt  = np.load(os.path.join(V4_DATA, f"vac_Xtau_{tag}.npy"))
    phi = np.load(os.path.join(V4_DATA, f"vac_phi0_{tag}.npy"))
    psi = np.load(os.path.join(V4_DATA, f"vac_psi0_{tag}.npy"))

    N = len(X0)
    idx = np.arange(N)
    if max_anchors and max_anchors < N:
        idx = np.random.default_rng(seed).choice(N, max_anchors, replace=False)

    feat   = _featurize_adp(X0[idx])                   # (n, 231)
    bursts = _featurize_adp(Xt[idx])[:, None, :]       # (n, 1, 231)  K=1
    phi, psi = phi[idx], psi[idx]

    # reference = fresh 0.1 ps transfer-operator right eigenvectors EV2 (phi), EV3 (psi)
    edges, R, _ = _adp_reference(tag)
    NB = 40
    ci = (np.clip(np.digitize(phi, edges) - 1, 0, NB - 1) * NB
          + np.clip(np.digitize(psi, edges) - 1, 0, NB - 1))
    refs = np.stack([R[ci, 1], R[ci, 2]]).astype(np.float64)   # (2, n)

    return dict(
        tag="adp_300k_0p1", reversible=True,
        feat=feat, bursts=bursts, coords=np.stack([phi, psi], 1),
        refs=refs, ref_names=["EV2 (phi-flip)", "EV3 (psi)"],
        labels=None,
    )


# --------------------------------------------------------------------------- #
# 3. NON-REVERSIBLE directed-ring toy
# --------------------------------------------------------------------------- #
_RING_CENTERS = np.array([np.pi / 3, np.pi, 5 * np.pi / 3])   # 3 wells (minima of cos 3θ)

def _ring_force(x, a, b, R, kappa):
    r = np.hypot(x[:, 0], x[:, 1]) + 1e-9
    th = np.arctan2(x[:, 1], x[:, 0])
    s3 = np.sin(3 * th)
    gx = 2 * a * (r - R) * x[:, 0] / r + 3 * b * s3 * x[:, 1] / r**2
    gy = 2 * a * (r - R) * x[:, 1] / r - 3 * b * s3 * x[:, 0] / r**2
    fx, fy = -x[:, 1] / r, x[:, 0] / r              # tangential, non-conservative (curl) force
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
    """Overdamped Langevin on a 3-well ring + tangential drift (kappa). kappa>0 breaks
    detailed balance -> directional current and a complex-conjugate transfer-operator pair.
    Anchors are seeded UNIFORMLY in angle (balanced basin coverage, well-conditioned
    reference operator), then K independent bursts of `burst` steps give the Koopman pairs."""
    cpath = os.path.join(CACHE, f"ring_n{n_anchors}_K{K}_L{burst}_k{kappa}_b{b}_s{seed}.npz")
    if cache and os.path.exists(cpath):
        z = np.load(cpath)
        return {k: (z[k] if k not in ("tag",) else str(z[k]))
                for k in z.files} | dict(reversible=False, ref_names=["Re v2 (~cos)", "Im v2 (~sin)"])

    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi, n_anchors)
    r0  = R + 0.06 * rng.standard_normal(n_anchors)
    anch = np.stack([r0 * np.cos(th0), r0 * np.sin(th0)], 1)
    anch = _ring_propagate(anch, 40, dt, D, a, b, R, kappa, rng)        # settle in-distribution
    ends = np.stack([_ring_propagate(anch.copy(), burst, dt, D, a, b, R, kappa, rng)
                     for _ in range(K)], 1)                              # (n, K, 2)

    la = _ring_basin(anch)
    le = _ring_basin(ends)

    # reference: dominant complex eigenvector of the 3-state burst transfer operator,
    # lifted back to every anchor via its basin -> Re/Im give cos/sin-like ring shapes.
    T = np.zeros((3, 3))
    for j in range(K):
        np.add.at(T, (la, le[:, j]), 1.0)
    T = T / T.sum(1, keepdims=True)
    w, V = np.linalg.eig(T)
    order = np.argsort(-np.abs(w))
    w, V = w[order], V[:, order]
    v2 = V[:, 1]                                   # dominant non-stationary (complex) mode
    refs = np.stack([np.real(v2)[la], np.imag(v2)[la]]).astype(np.float64)   # (2, n)

    out = dict(
        tag="directed_ring",
        feat=anch.astype(np.float32), bursts=ends.astype(np.float32),
        coords=anch.astype(np.float32),
        refs=refs, labels=la.astype(np.int64),
        T=T, eigs=w.astype(np.complex128),
    )
    if cache:
        np.savez(cpath, **{k: v for k, v in out.items() if k != "tag"}, tag=out["tag"])
    out.update(reversible=False, ref_names=["Re v2 (~cos)", "Im v2 (~sin)"])
    return out
