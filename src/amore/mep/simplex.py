"""
Simplex views for multi-membership reaction paths.

A k-membership ISOKANN model maps configurations to the probability simplex
Δ^{k-1} (χ_m ≥ 0, Σ_m χ_m = 1).  Its faces are the metastable structure:

  * vertices  χ_i = 1                      — the metastable states (nodes)
  * edges     χ_k = 0 ∀ k ∉ {i,j}          — the i↔j interconversions (1-faces)

This module exposes the **two views** as one-line entry points that take the FULL
model plus state indices and build the scalar reaction coordinate + constraints
internally — so the caller never touches the level-set plumbing:

  reaction_path_face(model, i, x0, featurizer, ...)      # 1-D / committor view: ∇χ_i
  reaction_path_edge(model, i, j, x0, featurizer, ...)   # edge view: ∇s, s=½(χ_i−χ_j+1)

A NOTE ON THE s=½ DEGENERACY.  The edge coordinate s = ½(χ_i − χ_j + 1) only encodes
χ_i = χ_j at s = ½, and the locus χ_i = χ_j contains BOTH the genuine i↔j saddle and the
*third* basin (χ_i=χ_j≈0).  But the third basin lies ONLY on the s = ½ level set — for
s ≠ ½ it has χ_i−χ_j ≠ 0, so it is not on the level set the path integrates along, and
the path cannot drift into it.  The drift is therefore purely a SEED-INITIALISATION
issue: an over-aggressive energy minimisation of a seed at s ≈ ½ finds the third basin
(the lowest-energy point reachable on that level set).  Keep the seed prep light and the
problem disappears — no constraint needed.  The committor coordinate χ_i (face view) does
not even have this issue: the basins sit at χ_i = 0 or 1, never on the χ_i = ½ transition
surface, so ∇χ_i is the cleaner object — "which edge" is then an a-posteriori sort of the
face paths by the two states they connect.
"""
from __future__ import annotations

import numpy as np
import torch as pt

from .core import reaction_path_minimum, transition_state, _chi_val
from .constrained import build_chi_mep_projected, build_chi_string_independent


# ---------------------------------------------------------------------------
# Scalar collective-variable wrappers over a k-membership model
# ---------------------------------------------------------------------------

class FaceCV(pt.nn.Module):
    """s = χ_i  — membership i vs the rest (the 1-D / committor coordinate)."""

    def __init__(self, model, i):
        super().__init__()
        self.model, self.i = model, int(i)

    def forward(self, feats):
        return self.model(feats)[..., self.i:self.i + 1]


class EdgeCV(pt.nn.Module):
    """s = ½(χ_i − χ_j + 1) ∈ [0,1] — the i↔j reaction coordinate (saddle at ½)."""

    def __init__(self, model, i, j):
        super().__init__()
        self.model, self.i, self.j = model, int(i), int(j)

    def forward(self, feats):
        chi = self.model(feats)
        return 0.5 * (chi[..., self.i:self.i + 1] - chi[..., self.j:self.j + 1] + 1.0)


class LogitEdgeCV(pt.nn.Module):
    """s = logit_i - logit_j, the PRE-softmax analogue of EdgeCV.

    Softmax's Jacobian is diag(p) - p p^T, which vanishes as any p_i -> 1 -- exactly the
    saturating-gradient regime chi sits in near any pure state or deep in a basin. The
    raw trunk (logit) output has no such saturation, so this coordinate can stay well-
    conditioned exactly where EdgeCV's ‖∇χ‖^2-based projection/retraction collapses.
    Requires `model` to expose `.logits(feats)` (e.g. comfeat.NormalizedChiNet) -- unlike
    chi, logits are unbounded, so this is not restricted to [0,1]; treat step_size/targets
    in logit units, not chi units."""

    def __init__(self, model, i, j):
        super().__init__()
        self.model, self.i, self.j = model, int(i), int(j)

    def forward(self, feats):
        logits = self.model.logits(feats)
        return logits[..., self.i:self.i + 1] - logits[..., self.j:self.j + 1]


class LogitFaceCV(pt.nn.Module):
    """s = logit_i, the PRE-softmax analogue of FaceCV. See `LogitEdgeCV` for rationale."""

    def __init__(self, model, i):
        super().__init__()
        self.model, self.i = model, int(i)

    def forward(self, feats):
        return self.model.logits(feats)[..., self.i:self.i + 1]


class ActivityCV(pt.nn.Module):
    """a = χ_i + χ_j — edge activity (1 on the edge, →0 in the third basin)."""

    def __init__(self, model, i, j):
        super().__init__()
        self.model, self.i, self.j = model, int(i), int(j)

    def forward(self, feats):
        chi = self.model(feats)
        return chi[..., self.i:self.i + 1] + chi[..., self.j:self.j + 1]


# ---------------------------------------------------------------------------
# Seed selection on the edge / face
# ---------------------------------------------------------------------------

def separatrix_frames(model, featurizer, anchors, i, j=None, lo=0.45, hi=0.55,
                      activity_min=None, device=None):
    """Transition-state frames of a face (j=None → s=χ_i) or edge (s=½(χ_i−χ_j+1))
    with s∈[lo,hi].  For an edge, optionally also require activity χ_i+χ_j ≥
    activity_min so frames sit on the genuine edge (not the third basin).

    Returns (frames, s_values).
    """
    cv = FaceCV(model, i) if j is None else EdgeCV(model, i, j)
    frames, s_vals = transition_state(cv, featurizer, anchors, chi_lo=lo, chi_hi=hi)
    if j is not None and activity_min is not None and len(frames):
        device = device or next(model.parameters()).device
        with pt.no_grad():
            chi = model(featurizer(pt.as_tensor(np.asarray(frames, np.float32),
                                                device=device))).cpu().numpy()
        keep = (chi[:, i] + chi[:, j]) >= activity_min
        frames, s_vals = frames[keep], s_vals[keep]
    return frames, s_vals


# ---------------------------------------------------------------------------
# The two entry points — MEP (energy)
# ---------------------------------------------------------------------------

def reaction_path_face(model, i, x0, featurizer, potential_fn=None, grad_fn=None,
                       steps=100, stepsize=None, **kw):
    """FACE view — minimum-(free-)energy path along the single membership χ_i.
    This is the classic 1-D / committor coordinate: pure ∇χ_i integration, no extra
    constraint (you disambiguate states yourself).  `model` is the full k-output net.
    """
    cv = FaceCV(model, i)
    stepsize = stepsize if stepsize is not None else 1.0 / steps
    return reaction_path_minimum(cv, featurizer, x0, steps=steps, stepsize=stepsize,
                                 potential_fn=potential_fn, grad_fn=grad_fn, **kw)


def reaction_path_edge(model, i, j, x0, featurizer, potential_fn=None, grad_fn=None,
                       steps=100, stepsize=None, **kw):
    """EDGE view — minimum-(free-)energy path of the i↔j interconversion along
    s = ½(χ_i−χ_j+1).  No activity constraint: the third basin only lies on the s=½
    level set, so away from the saddle the path cannot drift into it; keeping the SEED
    off an over-minimised s=½ collapse is handled at the seed-prep stage.  Which states
    a path actually connects is read off its ends (a-posteriori edge sorting).
    """
    cv = EdgeCV(model, i, j)
    stepsize = stepsize if stepsize is not None else 1.0 / steps
    return reaction_path_minimum(cv, featurizer, x0, steps=steps, stepsize=stepsize,
                                 potential_fn=potential_fn, grad_fn=grad_fn, **kw)


# ---------------------------------------------------------------------------
# The two entry points — MFEP (projected Langevin, free energy)
# ---------------------------------------------------------------------------

def mfep_face(sim, model, i, x0, featurizer, **kw):
    """FACE-view minimum-FREE-energy path along χ_i."""
    return build_chi_mep_projected(sim, FaceCV(model, i), featurizer, x0, **kw)


def mfep_edge(sim, model, i, j, x0, featurizer, **kw):
    """EDGE-view minimum-FREE-energy path of i↔j along s = ½(χ_i−χ_j+1)."""
    return build_chi_mep_projected(sim, EdgeCV(model, i, j), featurizer, x0, **kw)


def mfep_logit_edge(sim, model, i, j, x0, featurizer, **kw):
    """EDGE-view MFEP of i<->j along the PRE-softmax coordinate s = logit_i - logit_j.
    See `LogitEdgeCV` -- avoids softmax's vanishing-Jacobian saturation near pure states."""
    return build_chi_mep_projected(sim, LogitEdgeCV(model, i, j), featurizer, x0, **kw)


# ---------------------------------------------------------------------------
# The two entry points — decoupled "string" MFEP (independent per-level sampling)
# ---------------------------------------------------------------------------

def string_face(sim, model, i, seeds, cv_levels, featurizer, **kw):
    """FACE-view decoupled/"string" MFEP along χ_i: independent per-level sampling,
    each level seeded from its own configuration (ideally a real frame near that
    level) instead of build_chi_mep_projected's sequential chain. See
    `build_chi_string_independent`."""
    return build_chi_string_independent(sim, FaceCV(model, i), featurizer, seeds,
                                        cv_levels, **kw)


def string_edge(sim, model, i, j, seeds, cv_levels, featurizer, **kw):
    """EDGE-view decoupled/"string" MFEP of i<->j along s = ½(χ_i−χ_j+1): independent
    per-level sampling, each level seeded from its own configuration (ideally a real
    frame near that level) instead of build_chi_mep_projected's sequential chain. See
    `build_chi_string_independent`."""
    return build_chi_string_independent(sim, EdgeCV(model, i, j), featurizer, seeds,
                                        cv_levels, **kw)


__all__ = [
    "FaceCV", "EdgeCV", "ActivityCV", "LogitFaceCV", "LogitEdgeCV", "separatrix_frames",
    "reaction_path_face", "reaction_path_edge", "mfep_face", "mfep_edge", "mfep_logit_edge",
    "string_face", "string_edge",
]
