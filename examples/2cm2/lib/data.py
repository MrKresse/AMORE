# -*- coding: utf-8 -*-
"""
data.py — feature computation for the 2cm2 multi-D ISOKANN example.

Reproduces the featurisation of Fazil's `compute_coords_350_2000.ipynb`:

  * per-RESIDUE centre of mass of the protein heavy atoms
    (`select_atoms("protein and not element H")`, `center_of_mass(compound="residues")`),
    giving one 3-D point per residue (285 residues);
  * pairwise Euclidean distances over ALL residue pairs (C(285,2) = 40470),
    with the minimum-image convention applied using the (static, cubic) box
    → periodic boundary conditions, via `amore.features.features_pairs(..., box=box)`.

Two feature sets are produced, exactly as in the reference pipeline:

  D0  (N_anchor, n_feat)          — instantaneous features at the anchor frames
                                    Nstart..Nend of the half trajectory.
  Dt  (N_anchor, N_rep, n_feat)   — features at the Koopman-burst endpoints: for each
                                    anchor i and replica j, the *last* frame (index 9) of
                                    `xt_{i}_r{j}_CA_aligned.dcd` — the lag-time image used
                                    as the ISOKANN Koopman target.

The result is cached under $CM2_SCRATCH (default /scratch/htc/<user>/2cm2) and shared by
all of the per-dimension notebooks (the features do not depend on k).
"""
from __future__ import annotations
import os, sys, time, getpass
import numpy as np
import torch as pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))


def features_pairs(pairs, coords_flat, box=None):
    """Pairwise Euclidean distances for atom/COM index pairs, with the minimum-image
    convention (periodic boundary conditions) when `box` is given.

    Same formula as Fazil's `amore.features.features_pairs` (MIC: d -= L*round(d/L)).
    Reproduced here because the jkresse `amore.features.features_pairs` has no box arg.

    pairs (n_feat,2) int; coords_flat (n_frames, 3*n) or (3*n,); box (3,) or (n_frames,3).
    """
    pairs_t = pt.as_tensor(pairs, dtype=pt.long, device=coords_flat.device)
    single = coords_flat.ndim == 1
    if single:
        coords_flat = coords_flat.unsqueeze(0)
    nf = coords_flat.shape[0]
    na = coords_flat.shape[1] // 3
    xyz = coords_flat.view(nf, na, 3)
    diff = xyz[:, pairs_t[:, 0], :] - xyz[:, pairs_t[:, 1], :]     # (nf, n_feat, 3)
    if box is not None:
        b = pt.as_tensor(box, dtype=coords_flat.dtype, device=coords_flat.device)
        bview = b.view(1, 1, 3) if b.ndim == 1 else b.view(nf, 1, 3)
        diff = diff - bview * pt.round(diff / bview)
    feats = pt.linalg.norm(diff, dim=-1)
    return feats.squeeze(0) if single else feats

# ── paths / config ───────────────────────────────────────────────────────────
INP_DIR   = "/scratch/htc/fsafarov/2cm2_simulation/md2/input/"
PDB_FILE  = "pdbfile_water_pose2_100ns_10000_1.pdb"
DCD_DIR   = "/scratch/htc/fsafarov/2cm2_simulation/md2/output_pose2/trajectories/openmm_files/"
DCD_FILE  = "trajectory_water_100ns_10000_half.dcd"
TIMELAG_DIR = os.path.join(DCD_DIR, "final_states_rep1", "aligned_dcds")

SELECTION = "protein and not element H"     # COM taken per residue
NSTART, NEND = 350, 2000                     # anchor frame window (as compute_coords_350_2000)
LAG = int(os.environ.get("CM2_LAG", 1))      # Koopman lag in trajectory frames (x0[i] -> xt[i+LAG])
N_REP  = 1                                    # one lag image per anchor (trajectory pairing)

SCRATCH = os.environ.get("CM2_SCRATCH", f"/scratch/htc/{getpass.getuser()}/2cm2")


def pdb_path():  return os.path.join(INP_DIR, PDB_FILE)
def dcd_path():  return os.path.join(DCD_DIR, DCD_FILE)


# ── low-level COM extraction ─────────────────────────────────────────────────
def _residue_pairs(n_res: int) -> np.ndarray:
    """All (i,j) residue-COM index pairs, i<j — the feature index set."""
    import itertools
    return np.array(sorted(set(itertools.combinations(range(n_res), 2))), dtype=np.int64)


def compute_traj_com(f0, f1, verbose=True):
    """Per-residue COM at trajectory frames [f0, f1) of the half trajectory (one pass).

    Returns (coms, box) with coms (f1-f0, n_res, 3) float32 and box (3,) the static
    orthorhombic box lengths."""
    import MDAnalysis as mda
    u = mda.Universe(pdb_path(), dcd_path())
    prot = u.select_atoms(SELECTION)
    n = int(f1 - f0)
    n_res = prot.center_of_mass(compound="residues").shape[0]
    coms = np.zeros((n, n_res, 3), np.float32)
    box = u.dimensions[:3].astype(np.float32).copy()
    t0 = time.perf_counter()
    for i in range(n):
        u.trajectory[f0 + i]
        coms[i] = prot.center_of_mass(compound="residues")
        if verbose and (i % 400 == 0 or i == n - 1):
            print(f"  frame {i+1}/{n}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return coms, box


# ── feature assembly (per-residue COM → PBC pairwise distances) ──────────────
def _feats_from_com(coms, pairs, box):
    """coms (..., n_res, 3) → pairwise MIC distances (..., n_feat)."""
    t = pt.as_tensor(np.asarray(coms, np.float32))
    lead = t.shape[:-2]
    flat = t.reshape(-1, t.shape[-2] * 3)                        # (M, 3*n_res)
    d = features_pairs(pairs, flat, box=pt.as_tensor(box, dtype=pt.float32))
    return d.reshape(*lead, -1)


def com_cache_path():
    return os.path.join(SCRATCH, f"coms_{NSTART}_{NEND}_lag{LAG}.pt")


def build_coms(force=False, verbose=True):
    """Compute (or load cached) the per-residue COMs for the Koopman pairs, paired ALONG the
    trajectory: anchor frames [NSTART, NEND) and their lag images [NSTART+LAG, NEND+LAG).

    Done in a single trajectory pass over [NSTART, NEND+LAG); com0 and comt are slices of
    that.  comt carries a singleton replica axis (N, 1, n_res, 3) so it plugs straight into
    the ISA training loop's (N, K_burst, feat) burst convention."""
    os.makedirs(SCRATCH, exist_ok=True)
    cache = com_cache_path()
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[coms] loading cache {cache}", flush=True)
        return pt.load(cache, weights_only=False)
    if verbose:
        print(f"[coms] computing per-residue COMs for frames [{NSTART}, {NEND+LAG}) "
              f"(lag={LAG}) ...", flush=True)
    coms, box = compute_traj_com(NSTART, NEND + LAG, verbose=verbose)
    n = int(NEND - NSTART)
    com0 = coms[:n]                                  # (N, n_res, 3)
    comt = coms[LAG:LAG + n][:, None, :, :]          # (N, 1, n_res, 3)
    out = dict(com0=com0, comt=comt, box=box, lag=LAG,
               nstart=NSTART, nend=NEND, n_rep=N_REP, selection=SELECTION)
    pt.save(out, cache)
    if verbose:
        print(f"[coms] cached → {cache} (com0 {tuple(com0.shape)}, comt {tuple(comt.shape)})",
              flush=True)
    return out


def build_features(force=False, use_pbc=True, verbose=True):
    """Assemble D0, Dt pairwise-distance features from the (cached) COMs.

    `use_pbc` toggles the minimum-image convention.  NOTE: the protein is whole in both the
    raw and aligned trajectories and its diameter (~64 A) exceeds half the box (~40.5 A), so
    MIC wraps ~8% of residue pairs.  `use_pbc=True` reproduces Fazil's reference pipeline
    (features_pairs with box); `use_pbc=False` gives raw physical COM distances.

    Returns dict: D0 (N,n_feat), Dt (N,N_rep,n_feat), pairs, box, com0, use_pbc, ...
    """
    os.makedirs(SCRATCH, exist_ok=True)
    tag = "pbc" if use_pbc else "raw"
    cache = os.path.join(SCRATCH, f"features_{NSTART}_{NEND}_lag{LAG}_{tag}.pt")
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[features] loading cache {cache}", flush=True)
        return pt.load(cache, weights_only=False)

    c = build_coms(force=False, verbose=verbose)
    com0, comt, box = c["com0"], c["comt"], c["box"]
    pairs = _residue_pairs(com0.shape[1])
    fbox = box if use_pbc else None
    if verbose:
        print(f"[features] assembling pairwise-distance features "
              f"(use_pbc={use_pbc}, {len(pairs)} features) ...", flush=True)
    D0 = _feats_from_com(com0, pairs, fbox)                      # (N, n_feat)
    Dt = _feats_from_com(comt, pairs, fbox)                      # (N, N_rep, n_feat)

    out = dict(D0=D0, Dt=Dt, pairs=pairs, box=box, com0=com0, use_pbc=use_pbc, lag=LAG,
               nstart=NSTART, nend=NEND, n_rep=N_REP, selection=SELECTION)
    pt.save(out, cache)
    if verbose:
        print(f"[features] cached → {cache}  (D0 {tuple(D0.shape)}, Dt {tuple(Dt.shape)})",
              flush=True)
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_coms(force=force)
    build_features(force=force, use_pbc="--no-pbc" not in sys.argv)
