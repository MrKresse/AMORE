# -*- coding: utf-8 -*-
"""
comfeat.py -- differentiable atoms -> per-residue COM -> PBC pairwise-distance featurizer.

`data.build_features_pocket` computes the training features (side-chain COMs + all-atom
contact-pair distances) from PRECOMPUTED numpy arrays -- fine for training, but the
chi-MEP work needs chi as a function of the FULL solvated system's raw atom coordinates,
differentiable end to end (`amore.mep.core._chi_and_grad` autograds straight through
`featurizer` to the network). This module is that differentiable pipeline.

Two earlier feature-set tracks (protein-only `ResidueCOMPairFeaturizer`/
`make_torch_featurizer`/`load_trained_model`, and ligand-inclusive
`LigandResidueCOMFeaturizer`/`make_torch_featurizer_lig`/`load_trained_model_lig`) were
removed once the pocket feature set below superseded both -- protein-only gives the
ligand exactly zero gradient (blocking ligand-aware chi-MEP work), and ligand-inclusive's
COM-only ligand term gives hydrogens zero gradient (H-atom relaxation-lag artifacts in
chi-MEP work). See `examples/2cm2_full/POSTMORTEM.md`'s addendum for the ligand-inclusive
feature set's own spectral-gap finding (k=4, not k=3) before it was removed.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch as pt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "2cm2", "lib"))
import data  # noqa: E402


class NormalizedChiNet(pt.nn.Module):
    """net(feats) expects z-scored features (`train.normalise`'s `(feats-mu)/sd`) -- `mu`/`sd`
    are NOT saved in `run.get_model`'s cache (only `net_state`), but are deterministic given the
    same cached `D0` the model was trained on, so they're cheap to recompute exactly.  This
    wraps a raw `ChiNetMulti` with that fixed affine normalisation so the composite maps RAW
    (unnormalised) `comfeat` features straight to chi -- the `model` argument
    `amore.mep.FaceCV`/`EdgeCV`/`mfep_face` etc. expect, paired with `comfeat`'s raw featurizer.
    """

    def __init__(self, net, mu, sd):
        super().__init__()
        self.net = net
        self.register_buffer("mu", pt.as_tensor(np.asarray(mu), dtype=pt.float32).view(1, -1))
        self.register_buffer("sd", pt.as_tensor(np.asarray(sd), dtype=pt.float32).view(1, -1))

    def forward(self, feats):
        return self.net((feats - self.mu) / self.sd)

    def logits(self, feats):
        """Pre-softmax trunk output (same normalisation, no softmax).  Softmax's Jacobian
        is diag(p) - p p^T, which vanishes as any p_i -> 1 -- exactly the saturating-
        gradient regime chi sits in near any pure state.  The raw linear-layer gradient
        w.r.t. input has no such saturation, so using logit_i - logit_j (or logit_i alone)
        as the reaction coordinate instead of chi_i can stay numerically well-conditioned
        exactly where chi's own gradient collapses."""
        return self.net.trunk((feats - self.mu) / self.sd)


class PocketFeaturizer(pt.nn.Module):
    """Differentiable counterpart to `data.build_features_pocket`: whole-protein
    side-chain COM<->COM distances (285 points, GLY/NME whole-residue fallback -- see
    `data._sidechain_or_fallback_mask`) PLUS all-atom (hydrogens included, BOTH sides)
    ligand<->protein distances restricted to the specific atom pairs that ever contact
    within 5A over the full trajectory (`data.build_features_pocket`'s cached
    `contact_pairs`/`unique_prot_atoms`/`unique_lig_atoms` -- reused directly here, not
    recomputed, so the atom-pair ordering is guaranteed identical to what the network was
    trained on).  No separate ligand-COM<->residue-COM term (dropped by design).

    Gradient is nonzero on: every side-chain (or fallback whole-residue) protein heavy
    atom, AND every atom -- ligand or protein, hydrogens included -- that appears in the
    contact-pair list. That last part matters: earlier COM-only featurizers gave
    hydrogens exactly zero gradient, which caused H-atom relaxation-lag artifacts in
    chi-MEP work (a rough Euler/gradient jump moves heavy atoms but leaves H's behind,
    needing more MD relaxation time than is affordable to catch up). Contact-list
    hydrogens now move consistently with their heavy-atom partner from the start.
    """

    def __init__(self, pdb_path=None, device="cpu"):
        super().__init__()
        import MDAnalysis as mda

        pdb_path = pdb_path or data.pdb_path()
        u = mda.Universe(pdb_path)
        prot_heavy = u.select_atoms(data.SELECTION)
        keep_mask = data._sidechain_or_fallback_mask(prot_heavy)
        sc_atom_idx = prot_heavy.indices[keep_mask]                          # global indices
        sc_resindices = np.asarray(prot_heavy.resindices)[keep_mask]
        sc_masses = np.asarray(prot_heavy.masses, dtype=np.float64)[keep_mask]
        n_res = int(np.asarray(prot_heavy.resindices).max()) + 1

        res_mass_total = np.zeros(n_res, dtype=np.float64)
        np.add.at(res_mass_total, sc_resindices, sc_masses)
        sc_weights = (sc_masses / res_mass_total[sc_resindices]).astype(np.float32)

        feats = data.build_features_pocket(use_pbc=True, verbose=False)
        assert n_res == feats["n_res"], "sidechain residue count mismatch vs cached features"
        res_pairs = np.asarray(feats["res_pairs"])
        contact_pairs = np.asarray(feats["contact_pairs"])
        unique_prot_atoms = np.asarray(feats["unique_prot_atoms"])
        unique_lig_atoms = np.asarray(feats["unique_lig_atoms"])
        box = np.asarray(feats["box"], dtype=np.float32)

        self.n_atoms_total = u.atoms.n_atoms
        self.n_res = n_res
        self.register_buffer("sc_atom_idx", pt.as_tensor(sc_atom_idx, dtype=pt.long))
        self.register_buffer("sc_resindices", pt.as_tensor(sc_resindices, dtype=pt.long))
        self.register_buffer("sc_weights", pt.as_tensor(sc_weights, dtype=pt.float32))
        self.register_buffer("res_pairs", pt.as_tensor(res_pairs, dtype=pt.long))
        self.register_buffer("prot_contact_idx", pt.as_tensor(unique_prot_atoms, dtype=pt.long))
        self.register_buffer("lig_contact_idx", pt.as_tensor(unique_lig_atoms, dtype=pt.long))
        self.register_buffer("contact_pairs", pt.as_tensor(contact_pairs, dtype=pt.long))
        self.register_buffer("box", pt.as_tensor(box, dtype=pt.float32))   # Angstrom
        self.to(device)

    def forward(self, flat_coords):
        """flat_coords (B, n_atoms_total*3) or (n_atoms_total*3,), in NM -> (B, n_feat) or
        (n_feat,) pairwise distances in Angstrom (matching the trained network's units)."""
        single = flat_coords.ndim == 1
        if single:
            flat_coords = flat_coords.unsqueeze(0)
        flat_coords = flat_coords * 10.0                                # nm -> Angstrom
        B = flat_coords.shape[0]
        xyz = flat_coords.view(B, self.n_atoms_total, 3)
        b = self.box.view(1, 1, 3)

        # side-chain COM -> COM-COM distances
        sc_xyz = xyz[:, self.sc_atom_idx, :]
        weighted = sc_xyz * self.sc_weights.view(1, -1, 1)
        com = pt.zeros(B, self.n_res, 3, dtype=flat_coords.dtype, device=flat_coords.device)
        idx = self.sc_resindices.view(1, -1, 1).expand(B, -1, 3)
        com = com.scatter_add(1, idx, weighted)
        i, j = self.res_pairs[:, 0], self.res_pairs[:, 1]
        diff_sc = com[:, i, :] - com[:, j, :]
        diff_sc = diff_sc - b * pt.round(diff_sc / b)
        d_sc = pt.linalg.norm(diff_sc, dim=-1)

        # all-atom contact-pair distances
        prot_c_xyz = xyz[:, self.prot_contact_idx, :]
        lig_c_xyz = xyz[:, self.lig_contact_idx, :]
        pts = pt.cat([prot_c_xyz, lig_c_xyz], dim=1)
        ci, cj = self.contact_pairs[:, 0], self.contact_pairs[:, 1]
        diff_c = pts[:, ci, :] - pts[:, cj, :]
        diff_c = diff_c - b * pt.round(diff_c / b)
        d_c = pt.linalg.norm(diff_c, dim=-1)

        d = pt.cat([d_sc, d_c], dim=-1)
        return d.squeeze(0) if single else d


def make_torch_featurizer_pocket(pdb_path=None, device="cpu"):
    """Return a plain-callable featurizer(x_t) -> features, matching the signature
    `amore.mep` expects (`featurizer(x_t)` for an (n_frames, dim) or (1,dim) torch tensor),
    paired with `load_trained_model_pocket` (models trained on `data.build_features_pocket`)."""
    mod = PocketFeaturizer(pdb_path=pdb_path, device=device)
    mod.eval()

    def featurizer(x_t):
        return mod(x_t)

    featurizer.module = mod
    return featurizer


def load_trained_model_pocket(k, device="cpu"):
    """Load the cached k-dim pocket model (`run.get_model(k, pocket=True)`) and wrap it with
    the normalisation implied by `data.build_features_pocket`'s D0, so the result is a plain
    features->chi callable ready to hand to `amore.mep.FaceCV`/`EdgeCV` alongside
    `make_torch_featurizer_pocket`'s raw featurizer.

    IMPORTANT (Phase-0 finding, applies to every `run.get_model` variant): the cached `chi`
    array is NOT what `net_state` predicts. `train.train_isa` snapshots `best_chi` only when
    a NEW best validation score is found, but keeps training `net` afterward without ever
    restoring those best-val weights -- `net.state_dict()` at return time is the LAST
    iteration, `chi` is an EARLIER iteration. Since chi-MEP work needs the net that PRODUCES
    THE GRADIENTS to also define the states/separatrices it starts from, this recomputes
    `chi` fresh from `net_state` rather than trusting the cache, and returns it as `chi` in
    the second dict entry (`m["chi"]` from the raw cache is left untouched/unused, moved to
    `m["chi_cached_stale"]`).
    """
    import run
    import train
    from amore.isokann import ChiNetMulti

    m = run.get_model(k, pocket=True, verbose=False)
    feats = data.build_features_pocket(use_pbc=True, verbose=False)
    _, _, mu, sd = train.normalise(feats["D0"], feats["Dt"])

    net = ChiNetMulti(feats["D0"].shape[-1], k, hidden=m["hidden"]).to(device)
    net.load_state_dict(m["net_state"])
    net.eval()
    model = NormalizedChiNet(net, mu.numpy(), sd.numpy()).to(device)
    model.eval()

    with pt.no_grad():
        chi_live = model(pt.as_tensor(np.asarray(feats["D0"]), dtype=pt.float32,
                                       device=device)).cpu().numpy()
    m = dict(m)
    m["chi_cached_stale"] = m["chi"]
    m["chi"] = chi_live
    return model, m
