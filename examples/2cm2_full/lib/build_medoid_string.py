# -*- coding: utf-8 -*-
"""
build_medoid_string.py -- a-posteriori "string" pathways built ENTIRELY from real trajectory
frames (no simulation): for each relevant edge, bin the real on-edge frames into N_WINDOWS
equal-population windows along s_ij (amore's `analysis.windowed_edges`), then pick each
window's own aligned-RMSD medoid (`amore.mep.constrained.select_medoid`, restricted to the
pocket featurizer's own atom set) as its representative frame.

A neighbor-pooled "smoothed" variant (medoid drawn from windows w-1/w/w+1 together) was
tried and dropped -- overlapping pools frequently picked the literal SAME frame for
several consecutive windows (confirmed directly: e.g. frame 55 for both window 0 and 1,
frame 2507 for three windows in a row on edge 0-1), which reads as smoother in a quick
visual check but is actually a degenerate lack of motion for those stretches, not a
genuine smooth transition -- worse, not better.

Run (no GPU/OpenMM needed -- regular venv):
    python examples/2cm2_full/lib/build_medoid_string.py
"""
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "2cm2", "lib"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import data                                                    # noqa: E402
import comfeat                                                  # noqa: E402
import analysis as A                                             # noqa: E402
from amore.mep.constrained import select_medoid                    # noqa: E402

data.NSTART, data.NEND, data.LAG = 0, 2979, 20
K = 3
N_WINDOWS = 20
MAX_PER_WINDOW = 60
SCRATCH = os.environ.get("CM2_MFEP_SCRATCH", "/scratch/htc/jkresse/2cm2_mfep")


def main():
    model, m = comfeat.load_trained_model_pocket(K, device="cpu")
    feat = comfeat.make_torch_featurizer_pocket(device="cpu")
    chi = np.asarray(m["chi"])

    import MDAnalysis as mda
    u = mda.Universe(data.pdb_path(), data.dcd_path())
    n_atoms = len(u.atoms)

    def positions_of(idx_list):
        pos = np.empty((len(idx_list), n_atoms, 3))
        for k, a in enumerate(idx_list):
            u.trajectory[data.NSTART + int(a)]
            pos[k] = u.atoms.positions.astype(np.float64) / 10.0
        return pos

    def cap(w):
        if len(w) <= MAX_PER_WINDOW:
            return w
        pick = np.linspace(0, len(w) - 1, MAX_PER_WINDOW).round().astype(int)
        return w[pick]

    rows = A.edge_table(chi)
    results = {}
    for r in rows:
        i, j = r["edge"]
        if r["transition"] == 0:
            print(f"edge {i}-{j}: 0 genuine transition frames -- skipping")
            continue
        windows, window_s = A.windowed_edges(chi, i, j, n_windows=N_WINDOWS)
        windows = [cap(w) for w in windows]
        print(f"edge {i}-{j}: {N_WINDOWS} windows, sizes {[len(w) for w in windows]}")

        plain_idx = []
        for w in range(N_WINDOWS):
            pos = positions_of(windows[w])
            midx, _ = select_medoid(pos, featurizer=feat)
            plain_idx.append(int(windows[w][midx]))

        print(f"edge {i}-{j}: medoid frames {plain_idx}")
        results[(i, j)] = dict(window_s=window_s, plain_idx=plain_idx)

    out = os.path.join(SCRATCH, "medoid_string_results.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print("saved ->", out)


if __name__ == "__main__":
    main()
