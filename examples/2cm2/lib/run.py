# -*- coding: utf-8 -*-
"""
run.py — thin orchestration used by the per-dimension notebooks.

`get_model(k)` trains a k-dimensional softmax-ISA ISOKANN (or loads it from the scratch
cache) and returns the chi memberships, loss curves, and the trained network.  Training is
cached per k so re-executing a notebook is instant; delete the cache file to retrain.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch as pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data, train                                         # noqa: E402

SCRATCH = data.SCRATCH


def model_cache(k, use_pbc=True, include_ligand=False, pocket=False):
    tag = "pbc" if use_pbc else "raw"
    variant = "_pocket" if pocket else ("_lig" if include_ligand else "")
    return os.path.join(SCRATCH,
                        f"model_k{k}{variant}_{data.NSTART}_{data.NEND}_lag{data.LAG}_{tag}.pt")


def get_model(k, use_pbc=True, include_ligand=False, pocket=False, force=False, n_iter=200,
              epochs_per_iter=50, hidden=(4096, 512, 64), lr=5e-4, wd=1e-8, batch=128,
              seed=0, verbose=True):
    """Return dict(chi (N,k), loss_train, loss_val, net_state, k, hidden, nstart, ...).

    Trained with the ptb1b_isokann_500_2 hyperparameters (hidden [4096,512,64], lr 5e-4,
    wd 1e-8, batch 128) and the ISA softmax loss.  Cached under SCRATCH per k.

    include_ligand=True trains on `data.build_features_lig`'s feature set (protein
    residue-residue COM distances PLUS ligand-heavy-atom<->residue-COM distances) instead of
    the protein-only set -- needed for chi to have any gradient at all w.r.t. the ligand
    (see build_features_lig's docstring).

    pocket=True (takes precedence over include_ligand) trains on `data.build_features_pocket`'s
    feature set instead: whole-protein SIDE-CHAIN COM-COM distances PLUS all-atom (hydrogens
    included, both sides) ligand<->protein distances restricted to atom pairs that ever
    contact within 5A over the full trajectory -- no separate ligand-COM<->residue-COM term.
    Built to fix the H-atom relaxation-lag artifacts `include_ligand`'s COM-only ligand
    features caused in chi-MEP work (hydrogens had zero gradient there).

    Each variant gets its own cache tag (`_pocket`/`_lig`/none) -- `hidden`'s default first
    layer width may need to grow with a bigger input dimension."""
    cache = model_cache(k, use_pbc, include_ligand, pocket)
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[model] loading cache {cache}", flush=True)
        return pt.load(cache, weights_only=False)

    if pocket:
        feats = data.build_features_pocket(use_pbc=use_pbc, verbose=verbose)
    elif include_ligand:
        feats = data.build_features_lig(use_pbc=use_pbc, verbose=verbose)
    else:
        feats = data.build_features(use_pbc=use_pbc, verbose=verbose)
    D0n, Dtn, mu, sd = train.normalise(feats["D0"], feats["Dt"])
    res = train.train_isa(D0n, Dtn, k=k, hidden=hidden, n_iter=n_iter,
                          epochs_per_iter=epochs_per_iter, lr=lr, wd=wd, batch=batch,
                          seed=seed, verbose=verbose)
    out = dict(chi=res["chi"], loss_train=res["loss_train"], loss_val=res["loss_val"],
               net_state={kk: v.cpu() for kk, v in res["net"].state_dict().items()},
               k=k, hidden=list(hidden), nstart=data.NSTART, nend=data.NEND,
               lag=data.LAG, use_pbc=use_pbc, include_ligand=include_ligand, pocket=pocket)
    pt.save(out, cache)
    if verbose:
        print(f"[model] cached -> {cache}", flush=True)
    return out
