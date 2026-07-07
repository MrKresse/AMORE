# -*- coding: utf-8 -*-
"""
tests/_inverse_pcca_helpers.py

Shared plumbing for tests/test_inverse_pcca.py: reaches into
examples/isokann_benchmark/lib (the benchmark harness/systems/ground_truth
modules) the same way the benchmark's own notebooks do, trains (or loads a
cached) chi network for a given system/variant, and builds the burst-averaging
`propagate` closure `inverse_pcca` expects.

`harness.train_chi`'s own on-disk cache stores only the final chi VALUES at
the fixed anchor set (`chi_best.npy`) -- never the network weights -- so there
is no way to re-propagate new points (bursts) from that cache alone. This
module adds a small net-weight checkpoint under a filename `train_chi` itself
never writes or reads (`net_state_TEST.pt`), purely so repeated test runs
don't retrain from scratch. Uses a seed slot (97) dedicated to these tests so
it never collides with or overwrites the benchmark's own seed_0..4 caches.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_TESTS_DIR, ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
_BENCH_LIB = os.path.join(_REPO_ROOT, "examples", "isokann_benchmark", "lib")

for _p in (_SRC, _BENCH_LIB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                        # noqa: E402  (examples/isokann_benchmark/lib)
import harness                      # noqa: E402
import systems as bench_systems     # noqa: E402
import ground_truth as gt           # noqa: E402

from amore.isokann import ChiNetMulti, ChiNetMultiLinear  # noqa: E402

TEST_SEED = 97


def _net_ckpt_path(system: dict, variant: str, seed: int, k: int | None) -> str:
    # goes through harness._run_dir so a non-default k gets its own cache slot
    # (same k-suffixing harness.train_chi's own on-disk chi_best.npy cache uses),
    # never colliding with the default-k checkpoint.
    d = harness._run_dir(system["tag"], variant, seed, head=None, warmup=False, k=k)
    return os.path.join(d, "net_state_TEST.pt")


def get_trained_net(system: dict, variant: str, cfg=None, seed: int = TEST_SEED,
                    use_gpu: bool = True, k: int | None = None):
    """Train (or load a cached) chi network for `variant` on `system`, returning
    a live torch net that can be evaluated on arbitrary new inputs (bursts).

    k=None trains at the variant's default output dimension (N_STATES for
    membership variants). Pass k explicitly to train at a different -- e.g.
    deliberately overparametrized -- dimension; see harness.train_chi's own
    k parameter and amore.inverse_pcca for why."""
    cfg = cfg or harness.get_cfg(system["tag"])
    ckpt = _net_ckpt_path(system, variant, seed, k)
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    IN = system["feat"].shape[1]
    net_k = k if k is not None else harness.n_out(variant)
    head = harness.default_head(variant)
    Net = ChiNetMulti if head == "softmax" else ChiNetMultiLinear
    net = Net(IN, net_k, hidden=cfg.HIDDEN).to(device)

    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()
        return net

    res = harness.train_chi(system, variant, seed, cfg=cfg, use_gpu=use_gpu,
                            force=True, verbose=True, k=k)
    net = res["net"]
    torch.save(net.state_dict(), ckpt)
    net.eval()
    return net


def eval_chi(net, system: dict, sub=None) -> np.ndarray:
    """chi(x0) at anchors `sub` (default: all). Returns (n, k)."""
    device = next(net.parameters()).device
    F0 = system["feat"]
    sub = np.arange(len(F0)) if sub is None else sub
    f0 = torch.tensor(F0, device=device)
    net.eval()
    with torch.no_grad():
        return net(f0[sub]).cpu().numpy()


def make_propagate(net, system: dict, sub=None):
    """The Monte-Carlo burst-averaging K_tau chi estimator, in (N, k) orientation
    (matching `chi`'s orientation) -- the same average-over-bursts idiom
    `harness.py`'s own (k, n)-oriented `kchi(sub)` closure uses internally,
    just returned without its transpose."""
    device = next(net.parameters()).device
    F0 = system["feat"]
    BURSTS = system["bursts"]
    sub = np.arange(len(F0)) if sub is None else sub
    fts = [torch.tensor(BURSTS[:, j, :], device=device) for j in range(BURSTS.shape[1])]

    def propagate():
        net.eval()
        with torch.no_grad():
            return np.mean([net(ft[sub]).cpu().numpy() for ft in fts], axis=0)
    return propagate


def raw_burst_pairs(net, system: dict, sub=None):
    """Per-burst (not averaged) chi(x0), chi(x1) pairs -- (N*K, k) each -- for an
    external VAMP cross-check that needs individual samples, not the MC average."""
    device = next(net.parameters()).device
    F0 = system["feat"]
    BURSTS = system["bursts"]
    sub = np.arange(len(F0)) if sub is None else sub
    K = BURSTS.shape[1]
    f0 = torch.tensor(F0[sub], device=device)
    net.eval()
    with torch.no_grad():
        chi0 = net(f0).cpu().numpy()
        chi0_rep = np.repeat(chi0, K, axis=0)
        chi1 = np.concatenate(
            [net(torch.tensor(BURSTS[sub, j, :], device=device)).cpu().numpy()
             for j in range(K)], axis=0
        )
    # interleave to match repeat semantics: chi1 columns currently grouped by
    # burst index, chi0_rep by repeated-row; reorder chi1 to row-major (anchor,burst)
    n = len(sub)
    chi1 = chi1.reshape(K, n, -1).transpose(1, 0, 2).reshape(n * K, -1)
    return chi0_rep, chi1
