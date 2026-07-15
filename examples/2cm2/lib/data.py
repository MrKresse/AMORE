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
LIG_SELECTION = "resname KB8 and not element H"    # ligand heavy atoms (21 for KB8)
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


# ── ligand-inclusive features (protein residue-COM pairs PLUS ligand-heavy-atom <->
#    residue-COM distances) ───────────────────────────────────────────────────
# The original features above are protein-only: chi trained on them has EXACTLY ZERO
# gradient w.r.t. any ligand atom (confirmed directly: autograd through the differentiable
# reproduction of this pipeline gave nonzero gradient on precisely the 2313 protein heavy
# atoms and nothing else).  That's fine for state discrimination, but it means the
# membership-flow direction used for chi-MEP work can never move the ligand -- only the
# protein pocket moves, and the ligand can only respond passively (and far too slowly to
# matter at any affordable per-step MD budget).  These functions add the ligand's own heavy
# atoms into the feature set (distances to every residue COM) so chi -- and its gradient --
# can depend on where the ligand actually sits, not just on the protein's conformation.

def _ligand_residue_pairs(n_res: int, n_lig: int) -> np.ndarray:
    """All (residue_idx, n_res + ligand_atom_idx) pairs -- indices into a COMBINED per-frame
    points array laid out as [residue COMs (0..n_res-1)] ++ [ligand heavy atoms (n_res..)]."""
    import itertools
    return np.array([(i, n_res + j) for i, j in itertools.product(range(n_res), range(n_lig))],
                     dtype=np.int64)


def compute_traj_com_lig(f0, f1, verbose=True):
    """Per-residue COM AND ligand heavy-atom coordinates at trajectory frames [f0, f1), one
    trajectory pass (same loop as `compute_traj_com`, extended -- avoids a second full scan).

    Returns (coms, lig, box): coms (n, n_res, 3), lig (n, n_lig, 3), box (3,) float32."""
    import MDAnalysis as mda
    u = mda.Universe(pdb_path(), dcd_path())
    prot = u.select_atoms(SELECTION)
    lig = u.select_atoms(LIG_SELECTION)
    n = int(f1 - f0)
    n_res = prot.center_of_mass(compound="residues").shape[0]
    n_lig = lig.n_atoms
    coms = np.zeros((n, n_res, 3), np.float32)
    lig_xyz = np.zeros((n, n_lig, 3), np.float32)
    box = u.dimensions[:3].astype(np.float32).copy()
    t0 = time.perf_counter()
    for i in range(n):
        u.trajectory[f0 + i]
        coms[i] = prot.center_of_mass(compound="residues")
        lig_xyz[i] = lig.positions
        if verbose and (i % 400 == 0 or i == n - 1):
            print(f"  frame {i+1}/{n}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return coms, lig_xyz, box


def com_cache_path_lig():
    return os.path.join(SCRATCH, f"coms_lig_{NSTART}_{NEND}_lag{LAG}.pt")


def build_coms_lig(force=False, verbose=True):
    """Ligand-inclusive counterpart to `build_coms`: same anchor/lag-image pairing, plus the
    ligand heavy-atom coordinates at every one of those frames."""
    os.makedirs(SCRATCH, exist_ok=True)
    cache = com_cache_path_lig()
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[coms_lig] loading cache {cache}", flush=True)
        return pt.load(cache, weights_only=False)
    if verbose:
        print(f"[coms_lig] computing per-residue COMs + ligand heavy atoms for frames "
              f"[{NSTART}, {NEND+LAG}) (lag={LAG}) ...", flush=True)
    coms, lig_xyz, box = compute_traj_com_lig(NSTART, NEND + LAG, verbose=verbose)
    n = int(NEND - NSTART)
    com0 = coms[:n]; comt = coms[LAG:LAG + n][:, None, :, :]
    lig0 = lig_xyz[:n]; ligt = lig_xyz[LAG:LAG + n][:, None, :, :]
    out = dict(com0=com0, comt=comt, lig0=lig0, ligt=ligt, box=box, lag=LAG,
               nstart=NSTART, nend=NEND, n_rep=N_REP, selection=SELECTION,
               lig_selection=LIG_SELECTION)
    pt.save(out, cache)
    if verbose:
        print(f"[coms_lig] cached → {cache} (com0 {tuple(com0.shape)}, lig0 {tuple(lig0.shape)})",
              flush=True)
    return out


def build_features_lig(force=False, use_pbc=True, verbose=True):
    """Ligand-inclusive counterpart to `build_features`: protein residue-residue pairwise
    COM distances (unchanged, first `len(_residue_pairs(n_res))` columns) PLUS every ligand
    heavy atom's distance to every residue COM (remaining columns), same MIC/PBC convention
    throughout.  Column layout lets a model trained on the plain `build_features` set still
    be compared against the same leading columns here, though the two are trained
    independently (different input dimension).

    Returns dict: D0 (N,n_feat), Dt (N,1,n_feat), pairs, lig_pairs, n_res, n_lig, box, ...
    """
    os.makedirs(SCRATCH, exist_ok=True)
    tag = "pbc" if use_pbc else "raw"
    cache = os.path.join(SCRATCH, f"features_lig_{NSTART}_{NEND}_lag{LAG}_{tag}.pt")
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[features_lig] loading cache {cache}", flush=True)
        return pt.load(cache, weights_only=False)

    c = build_coms_lig(force=False, verbose=verbose)
    com0, comt, lig0, ligt, box = c["com0"], c["comt"], c["lig0"], c["ligt"], c["box"]
    n_res, n_lig = com0.shape[1], lig0.shape[1]
    res_pairs = _residue_pairs(n_res)
    lig_pairs = _ligand_residue_pairs(n_res, n_lig)
    pairs = np.concatenate([res_pairs, lig_pairs], axis=0)
    fbox = box if use_pbc else None
    if verbose:
        print(f"[features_lig] assembling pairwise-distance features (use_pbc={use_pbc}, "
              f"{len(res_pairs)} protein-protein + {len(lig_pairs)} ligand-protein = "
              f"{len(pairs)} features) ...", flush=True)

    # combined per-frame points: [residue COMs] ++ [ligand heavy atoms], same trailing axis
    points0 = np.concatenate([com0, lig0], axis=1)                       # (N, n_res+n_lig, 3)
    pointst = np.concatenate([comt, ligt], axis=2)                       # (N, 1, n_res+n_lig, 3)
    D0 = _feats_from_com(points0, pairs, fbox)                           # (N, n_feat)
    Dt = _feats_from_com(pointst, pairs, fbox)                          # (N, 1, n_feat)

    out = dict(D0=D0, Dt=Dt, pairs=pairs, res_pairs=res_pairs, lig_pairs=lig_pairs,
               n_res=n_res, n_lig=n_lig, box=box, com0=com0, lig0=lig0, use_pbc=use_pbc,
               lag=LAG, nstart=NSTART, nend=NEND, n_rep=N_REP, selection=SELECTION,
               lig_selection=LIG_SELECTION)
    pt.save(out, cache)
    if verbose:
        print(f"[features_lig] cached → {cache}  (D0 {tuple(D0.shape)}, Dt {tuple(Dt.shape)})",
              flush=True)
    return out


# ── pocket features (whole-protein SIDE-CHAIN COM distances PLUS all-atom (incl. H)
#    ligand<->protein distances for the SPECIFIC atom pairs that ever come within 5A of
#    each other across the full trajectory) ────────────────────────────────────────────
# Design rationale (user's call, after the ligand-inclusive MFEP work showed hydrogens
# lagging behind heavy-atom jumps -- they have zero chi-gradient in `build_features_lig`
# since only heavy COM/atoms are featurized): whole-protein conformational/allosteric
# signal only needs SIDE-CHAIN COM<->COM (a ligand's effect on distant residues is already
# transmitted THROUGH the protein's own side-chain network once the near-pocket residues
# respond to it -- no separate ligand-COM<->residue-COM term needed, unlike
# `build_features_lig`).  Side-chain COM (not whole-residue COM) better tracks rotameric/
# conformational state than the backbone-dominated whole-residue COM.  Right at the
# pocket, COM of any kind is too coarse to give hydrogens (on either side) real gradient,
# which is what caused the H-atom relaxation-lag artifacts -- so the pocket gets full
# ALL-ATOM (both sides, hydrogens included) pairwise distances, restricted to the specific
# atom pairs that ever come within 5 A of each other somewhere in the FULL trajectory (a
# fixed, data-derived contact list -- not every possible protein-atom x ligand-atom pair,
# which would be 4601*37=170,237; the real 5A-ever list for 2cm2 pose2 is 6,341).

LIG_RESNAME = "KB8"
SIDECHAIN_EXCLUDE_NAMES = ("N", "CA", "C", "O", "OXT")     # backbone heavy-atom names


def _sidechain_or_fallback_mask(prot_heavy):
    """Boolean mask over `prot_heavy` (a `SELECTION`-selected AtomGroup): True for
    side-chain heavy atoms, OR for every heavy atom of a residue with NO side chain at all
    (glycine; capping groups like NME) -- those residues fall back to their own
    whole-residue COM, the only sensible reference point left once backbone-only atoms are
    all there is.  Confirmed on 2cm2 pose2: 15 of 285 residues (14 GLY + the NME cap) need
    this fallback."""
    names = np.asarray(prot_heavy.names)
    is_backbone = np.isin(names, SIDECHAIN_EXCLUDE_NAMES)
    sidechain_mask = ~is_backbone
    resindices = np.asarray(prot_heavy.resindices)
    n_res = int(resindices.max()) + 1
    has_sidechain = np.zeros(n_res, dtype=bool)
    np.logical_or.at(has_sidechain, resindices[sidechain_mask], True)
    fallback = ~has_sidechain[resindices]
    return sidechain_mask | fallback


def find_contact_pairs(cutoff=5.0, f0=None, f1=None, verbose=True):
    """Scan the trajectory range [f0,f1) (defaults to the full [NSTART, NEND+LAG) anchor+lag
    range) for every (protein_atom, ligand_atom) pair -- ALL atoms, hydrogens included on
    both sides -- whose minimum-image distance ever drops below `cutoff` (Angstrom).

    Returns (prot_atom_indices, lig_atom_indices, min_dist): the first two are GLOBAL atom
    indices (into the full Universe), same length, one entry per contact pair found;
    min_dist is that pair's minimum distance ever observed.
    """
    import MDAnalysis as mda
    u = mda.Universe(pdb_path(), dcd_path())
    prot_all = u.select_atoms("protein")
    lig_all = u.select_atoms(f"resname {LIG_RESNAME}")
    f0 = NSTART if f0 is None else f0
    f1 = (NEND + LAG) if f1 is None else f1
    box = u.dimensions[:3].astype(np.float64)
    min_dist = np.full((prot_all.n_atoms, lig_all.n_atoms), np.inf, dtype=np.float64)
    t0 = time.perf_counter()
    n = f1 - f0
    for i, fi in enumerate(range(f0, f1)):
        u.trajectory[fi]
        diff = (prot_all.positions[:, None, :].astype(np.float64)
                - lig_all.positions[None, :, :].astype(np.float64))
        diff -= box * np.round(diff / box)
        d = np.linalg.norm(diff, axis=-1)
        np.minimum(min_dist, d, out=min_dist)
        if verbose and (i % 500 == 0 or i == n - 1):
            print(f"  contact scan frame {i+1}/{n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    mask = min_dist < cutoff
    pi, li = np.where(mask)
    return prot_all.indices[pi], lig_all.indices[li], min_dist[pi, li]


def contact_pairs_cache_path():
    return os.path.join(SCRATCH, "contact_pairs_5A.npz")


def get_contact_pairs(force=False, verbose=True):
    """Cached wrapper around `find_contact_pairs` (a full-trajectory scan is ~1 minute but
    no need to repeat it)."""
    os.makedirs(SCRATCH, exist_ok=True)
    cache = contact_pairs_cache_path()
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[contact_pairs] loading cache {cache}", flush=True)
        d = np.load(cache)
        return d["prot_atom_indices"], d["lig_atom_indices"], d["min_dist"]
    prot_idx, lig_idx, min_dist = find_contact_pairs(verbose=verbose)
    np.savez(cache, prot_atom_indices=prot_idx, lig_atom_indices=lig_idx, min_dist=min_dist)
    if verbose:
        print(f"[contact_pairs] found {len(prot_idx)} pairs, cached -> {cache}", flush=True)
    return prot_idx, lig_idx, min_dist


def compute_traj_com_pocket(f0, f1, contact_prot_atoms, contact_lig_atoms, verbose=True):
    """Per-frame, one trajectory pass over [f0,f1): (a) side-chain COM per residue (285
    points, mass-weighted, GLY/NME whole-residue fallback), (b) the raw all-atom positions
    of every UNIQUE atom appearing in the contact-pair list (both protein- and ligand-side).

    Returns (sc_com, prot_contact_xyz, lig_contact_xyz, box, unique_prot, unique_lig):
      sc_com            (n, n_res, 3)
      prot_contact_xyz  (n, len(unique_prot), 3)
      lig_contact_xyz   (n, len(unique_lig), 3)
      unique_prot/lig   sorted unique GLOBAL atom indices, matching the xyz arrays' 2nd axis
    """
    import MDAnalysis as mda
    u = mda.Universe(pdb_path(), dcd_path())
    prot_heavy = u.select_atoms(SELECTION)
    keep_mask = _sidechain_or_fallback_mask(prot_heavy)
    resindices = np.asarray(prot_heavy.resindices)[keep_mask]
    masses = np.asarray(prot_heavy.masses, dtype=np.float64)[keep_mask]
    n_res = int(np.asarray(prot_heavy.resindices).max()) + 1
    kept_atomgroup = prot_heavy[keep_mask]

    res_mass_total = np.zeros(n_res, dtype=np.float64)
    np.add.at(res_mass_total, resindices, masses)
    weights = masses / res_mass_total[resindices]

    unique_prot = np.unique(contact_prot_atoms)
    unique_lig = np.unique(contact_lig_atoms)
    prot_contact_ag = u.atoms[unique_prot]
    lig_contact_ag = u.atoms[unique_lig]

    n = int(f1 - f0)
    sc_com = np.zeros((n, n_res, 3), np.float32)
    prot_contact_xyz = np.zeros((n, len(unique_prot), 3), np.float32)
    lig_contact_xyz = np.zeros((n, len(unique_lig), 3), np.float32)
    box = u.dimensions[:3].astype(np.float32).copy()

    t0 = time.perf_counter()
    for i in range(n):
        u.trajectory[f0 + i]
        pos = kept_atomgroup.positions.astype(np.float64)
        weighted = pos * weights[:, None]
        com = np.zeros((n_res, 3), dtype=np.float64)
        np.add.at(com, resindices, weighted)
        sc_com[i] = com
        prot_contact_xyz[i] = prot_contact_ag.positions
        lig_contact_xyz[i] = lig_contact_ag.positions
        if verbose and (i % 400 == 0 or i == n - 1):
            print(f"  frame {i+1}/{n}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return sc_com, prot_contact_xyz, lig_contact_xyz, box, unique_prot, unique_lig


def com_cache_path_pocket():
    return os.path.join(SCRATCH, f"coms_pocket_{NSTART}_{NEND}_lag{LAG}.pt")


def build_coms_pocket(force=False, verbose=True):
    os.makedirs(SCRATCH, exist_ok=True)
    cache = com_cache_path_pocket()
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[coms_pocket] loading cache {cache}", flush=True)
        return pt.load(cache, weights_only=False)

    contact_prot_atoms, contact_lig_atoms, _ = get_contact_pairs(force=force, verbose=verbose)
    if verbose:
        print(f"[coms_pocket] computing side-chain COMs + contact-atom positions for frames "
              f"[{NSTART}, {NEND+LAG}) (lag={LAG}) ...", flush=True)
    sc_com, protc0_full, ligc0_full, box, unique_prot, unique_lig = compute_traj_com_pocket(
        NSTART, NEND + LAG, contact_prot_atoms, contact_lig_atoms, verbose=verbose)

    n = int(NEND - NSTART)
    com0 = sc_com[:n]; comt = sc_com[LAG:LAG + n][:, None, :, :]
    protc0 = protc0_full[:n]; protct = protc0_full[LAG:LAG + n][:, None, :, :]
    ligc0 = ligc0_full[:n]; ligct = ligc0_full[LAG:LAG + n][:, None, :, :]

    out = dict(com0=com0, comt=comt, protc0=protc0, protct=protct, ligc0=ligc0, ligct=ligct,
               contact_prot_atoms=contact_prot_atoms, contact_lig_atoms=contact_lig_atoms,
               unique_prot_atoms=unique_prot, unique_lig_atoms=unique_lig,
               box=box, lag=LAG, nstart=NSTART, nend=NEND, n_rep=N_REP, selection=SELECTION)
    pt.save(out, cache)
    if verbose:
        print(f"[coms_pocket] cached -> {cache} (com0 {tuple(com0.shape)}, "
              f"protc0 {tuple(protc0.shape)}, ligc0 {tuple(ligc0.shape)})", flush=True)
    return out


RESPAIR_CUTOFF = 25.0    # Angstrom; see _safe_residue_pairs docstring


def _safe_residue_pairs(com0, box, cutoff=RESPAIR_CUTOFF):
    """Filter `_residue_pairs(n_res)` down to pairs whose side-chain-COM separation never
    exceeds `cutoff` (Angstrom) anywhere in the real trajectory `com0` (N, n_res, 3).

    Rationale: with an unrestricted whole-protein pair set, several residue pairs sit
    persistently near half the box length (box=81.02 A here, half=40.51 A) -- confirmed on
    2cm2 pose2's MFEP work, where `PocketFeaturizer`'s minimum-image wrap
    (`diff - box*round(diff/box)`) is genuinely discontinuous exactly there (not a coding
    bug -- minimum-image distance is mathematically non-differentiable as true separation
    crosses L/2).  A pair hovering near that boundary can flip its wrapped image on
    ordinary thermal fluctuation within one level-set's short relaxation window, corrupting
    chi/its gradient for any network trained on the unfiltered set -- this was traced as
    the actual cause of `euler_step_to_levelset` producing a huge raw Euler jump (a fixed CV
    increment divided by a near-zero/corrupted ||grad chi||^2), not integrator error or
    genuine landscape roughness.  25 A leaves a 15.5 A safety margin below L/2 and drops
    40,470 -> 17,178 pairs; residues farther apart than that aren't capturing a direct
    interaction anyway -- any real allosteric signal that far is relayed through
    intermediate residues, which stay in the set.
    """
    n_res = com0.shape[1]
    pairs = _residue_pairs(n_res)
    i, j = pairs[:, 0], pairs[:, 1]
    diff = com0[:, i, :].astype(np.float64) - com0[:, j, :].astype(np.float64)
    diff -= box.astype(np.float64) * np.round(diff / box.astype(np.float64))
    maxdist = np.linalg.norm(diff, axis=-1).max(axis=0)
    return pairs[maxdist < cutoff]


def build_features_pocket(force=False, use_pbc=True, verbose=True):
    """Side-chain COM-COM (whole protein, restricted to pairs whose real-trajectory
    separation never exceeds `RESPAIR_CUTOFF` -- see `_safe_residue_pairs`) PLUS all-atom
    ligand<->protein distances restricted to the pairs that ever contact within 5 A over the
    full trajectory (6,341 pairs for 2cm2 pose2) -- NO ligand-COM<->residue-COM term (see
    the module-level design note above).  Same PBC/minimum-image convention throughout.

    Returns dict: D0 (N,n_feat), Dt (N,1,n_feat), res_pairs, contact_pairs, n_res,
    n_contact, contact_prot_atoms, contact_lig_atoms (global atom indices), box, ...
    """
    os.makedirs(SCRATCH, exist_ok=True)
    tag = "pbc" if use_pbc else "raw"
    cache = os.path.join(SCRATCH,
                         f"features_pocket_{NSTART}_{NEND}_lag{LAG}_{tag}_rcut{int(RESPAIR_CUTOFF)}.pt")
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[features_pocket] loading cache {cache}", flush=True)
        return pt.load(cache, weights_only=False)

    c = build_coms_pocket(force=False, verbose=verbose)
    com0, comt = c["com0"], c["comt"]
    protc0, protct = c["protc0"], c["protct"]
    ligc0, ligct = c["ligc0"], c["ligct"]
    box = c["box"]
    n_res = com0.shape[1]

    prot_local = {int(g): i for i, g in enumerate(c["unique_prot_atoms"].tolist())}
    lig_local = {int(g): i for i, g in enumerate(c["unique_lig_atoms"].tolist())}
    contact_prot_local = np.array([prot_local[int(g)] for g in c["contact_prot_atoms"]])
    contact_lig_local = np.array([lig_local[int(g)] for g in c["contact_lig_atoms"]])

    res_pairs_all = _residue_pairs(n_res)
    res_pairs = _safe_residue_pairs(com0, box)
    if verbose:
        print(f"[features_pocket] res_pairs filtered by {RESPAIR_CUTOFF}A cutoff: "
              f"{len(res_pairs)}/{len(res_pairs_all)} kept", flush=True)
    fbox = box if use_pbc else None
    n_prot_c = protc0.shape[1]
    contact_pairs = np.stack([contact_prot_local, n_prot_c + contact_lig_local], axis=1)
    if verbose:
        print(f"[features_pocket] assembling features (use_pbc={use_pbc}): "
              f"{len(res_pairs)} sidechain-COM-COM + {len(contact_pairs)} "
              f"ligand-protein all-atom contact pairs = {len(res_pairs)+len(contact_pairs)} "
              f"total", flush=True)

    D0_sc = _feats_from_com(com0, res_pairs, fbox)                          # (N, 40470)
    Dt_sc = _feats_from_com(comt, res_pairs, fbox)                          # (N, 1, 40470)

    points0 = np.concatenate([protc0, ligc0], axis=1)          # (N, n_prot_c+n_lig_c, 3)
    pointst = np.concatenate([protct, ligct], axis=2)          # (N, 1, n_prot_c+n_lig_c, 3)
    D0_contact = _feats_from_com(points0, contact_pairs, fbox)
    Dt_contact = _feats_from_com(pointst, contact_pairs, fbox)

    D0 = pt.cat([D0_sc, D0_contact], dim=-1)
    Dt = pt.cat([Dt_sc, Dt_contact], dim=-1)

    out = dict(D0=D0, Dt=Dt, res_pairs=res_pairs, contact_pairs=contact_pairs,
               n_res=n_res, n_contact=len(contact_pairs),
               unique_prot_atoms=c["unique_prot_atoms"], unique_lig_atoms=c["unique_lig_atoms"],
               contact_prot_atoms=c["contact_prot_atoms"], contact_lig_atoms=c["contact_lig_atoms"],
               box=box, use_pbc=use_pbc, lag=LAG, nstart=NSTART, nend=NEND, n_rep=N_REP,
               selection=SELECTION)
    pt.save(out, cache)
    if verbose:
        print(f"[features_pocket] cached -> {cache}  (D0 {tuple(D0.shape)})", flush=True)
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_coms(force=force)
    build_features(force=force, use_pbc="--no-pbc" not in sys.argv)
