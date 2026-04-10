import numpy as np
import torch as pt


def features_pairs(pairs, coords_flat):
    """
    Compute pairwise distance features for given atom index pairs.

    Args
    ----
    pairs : (n_feat, 2) array-like of int
        Each row is (i, j) atom indices (0-based).
    coords_flat : pt.Tensor
        Coordinates in flattened format:
          - (n_frames, 3*n_atoms) or
          - (3*n_atoms,) for a single frame

    Returns
    -------
    feats : pt.Tensor
        Distances:
          - (n_frames, n_feat) if coords_flat is 2D
          - (n_feat,) if coords_flat is 1D
    """
    if not pt.is_tensor(coords_flat):
        raise TypeError("coords_flat must be a pt.Tensor")

    device = coords_flat.device
    pairs_t = pt.as_tensor(pairs, dtype=pt.long, device=device)

    if pairs_t.ndim != 2 or pairs_t.shape[1] != 2:
        raise ValueError(f"pairs must have shape (n_feat, 2), got {tuple(pairs_t.shape)}")

    single_frame = (coords_flat.ndim == 1)
    if single_frame:
        coords_flat = coords_flat.unsqueeze(0)

    if coords_flat.ndim != 2 or coords_flat.shape[1] % 3 != 0:
        raise ValueError("coords_flat must have shape (n_frames, 3*n_atoms)")

    n_frames = coords_flat.shape[0]
    n_atoms = coords_flat.shape[1] // 3
    xyz = coords_flat.view(n_frames, n_atoms, 3)

    i = pairs_t[:, 0]
    j = pairs_t[:, 1]
    diff = xyz[:, i, :] - xyz[:, j, :]
    feats = pt.linalg.norm(diff, dim=-1)

    return feats.squeeze(0) if single_frame else feats


def make_featurizer(pairs):
    """
    Return a featurizer function that computes pairwise distances for the given pairs.

    Parameters
    ----------
    pairs : (n_feat, 2) array-like of int

    Returns
    -------
    featurizer : callable
        featurizer(xs) -> pt.Tensor of distances
    """
    pairs = np.asarray(pairs, dtype=np.int64)

    def featurizer(xs):
        return features_pairs(pairs, xs)

    return featurizer
