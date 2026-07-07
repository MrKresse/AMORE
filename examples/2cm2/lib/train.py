# -*- coding: utf-8 -*-
"""
train.py — multi-dimensional (k=3..6) softmax-ISA ISOKANN for the 2cm2 example.

Trains k membership functions chi_1..chi_k on the probability simplex Delta^{k-1} with the
**ISA isotarget and a softmax head** (`amore.isokann.ChiNetMulti`, no warm-up) — the AMORE
gold standard (examples/isokann_benchmark).  The network geometry and optimiser settings are
taken from Fazil's `ptb1b_isokann_500_2.ipynb`:

    hidden widths [4096, 512, 64],  lr 5e-4,  weight_decay 1e-8,  batch 128.

Difference from the scalar ptb1b run: the head is a k-way softmax (simplex memberships) and
the regression target is the ISA target recomputed each outer power-iteration, instead of the
1-D shift-scale committor.  Features are z-scored with the anchor (D0) statistics first.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch as pt
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))
from amore.isokann import ChiNetMulti                       # noqa: E402
from amore.isotarget import isa_target, gramschmidt_target  # noqa: E402


# ── feature normalisation ────────────────────────────────────────────────────
def normalise(D0, Dt):
    """z-score with anchor (D0) statistics; apply the same shift/scale to the lag images."""
    D0 = pt.as_tensor(np.asarray(D0), dtype=pt.float32)
    Dt = pt.as_tensor(np.asarray(Dt), dtype=pt.float32)
    mu = D0.mean(0, keepdim=True)
    sd = D0.std(0, keepdim=True) + 1e-8
    D0n = (D0 - mu) / sd
    Dtn = (Dt - mu.unsqueeze(0)) / sd.unsqueeze(0)          # Dt is (N, Kb, F)
    return D0n, Dtn, mu, sd


def _gs_residual(chi_kn, kchi_kn):
    try:
        tgt = gramschmidt_target(kchi_kn)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(np.mean((chi_kn - tgt) ** 2))


# ── training ─────────────────────────────────────────────────────────────────
def train_isa(D0, Dt, k, hidden=(4096, 512, 64), n_iter=200, epochs_per_iter=50,
              lr=5e-4, wd=1e-8, batch=128, grad_clip=5.0, val_frac=0.2,
              device=None, seed=0, verbose=True):
    """k softmax memberships via the ISA isotarget (no warm-up).

    Outer loop = Koopman power iteration: each iteration computes the ISA regression target
    from the current chi and its Koopman expectation E[chi(x_tau)|x0], then runs a short
    minibatch-SGD inner loop (batch 128) to fit it.  Tracks a train MSE-to-target curve and a
    validation Gram-Schmidt residual, and keeps the chi with the best validation residual.

    D0 (N, F) normalised anchor features; Dt (N, Kb, F) normalised lag features.
    Returns dict(net, chi (N,k), loss_train, loss_val, n_iter, k, hidden).
    """
    device = device or ("cuda" if pt.cuda.is_available() else "cpu")
    pt.manual_seed(seed); np.random.seed(seed)
    D0 = pt.as_tensor(np.asarray(D0), dtype=pt.float32, device=device)
    Dt = pt.as_tensor(np.asarray(Dt), dtype=pt.float32, device=device)
    N, F = D0.shape
    Kb = Dt.shape[1]

    m = np.random.rand(N) < (1 - val_frac)
    tr = pt.tensor(np.where(m)[0], device=device)
    te = pt.tensor(np.where(~m)[0], device=device)
    D0_tr, D0_te = D0[tr], D0[te]
    Dt_tr = Dt[tr]

    net = ChiNetMulti(F, k, hidden=list(hidden)).to(device)
    opt = pt.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    def kchi(feat_bursts):
        """E[chi(x_tau)] averaged over the Kb burst/lag images -> (k, n)."""
        net.eval()
        with pt.no_grad():
            n = feat_bursts.shape[0]
            flat = feat_bursts.reshape(n * Kb, F)
            c = net(flat).reshape(n, Kb, k).mean(1)         # (n, k)
        return c.cpu().numpy().T

    loss_tr, loss_val = [], []
    best_val, best_chi = np.inf, None
    t0 = time.perf_counter()
    for it in range(n_iter):
        net.eval()
        with pt.no_grad():
            chi0 = net(D0_tr).cpu().numpy().T               # (k, Ntr)
        kc = kchi(Dt_tr)                                     # (k, Ntr)
        try:
            tgt = isa_target(chi0, kc)                       # (k, Ntr)
        except (ValueError, np.linalg.LinAlgError):
            loss_tr.append(np.nan); loss_val.append(np.nan); continue
        tgt_t = pt.tensor(tgt.T, dtype=pt.float32, device=device)  # (Ntr, k)

        net.train()
        ntr = D0_tr.shape[0]
        ep_loss = 0.0
        for _ in range(epochs_per_iter):
            idx = pt.randint(0, ntr, (batch,), device=device)
            loss = nn.functional.mse_loss(net(D0_tr[idx]), tgt_t[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip); opt.step()
            ep_loss += loss.item()
        loss_tr.append(ep_loss / epochs_per_iter)

        net.eval()
        with pt.no_grad():
            chi_te = net(D0_te).cpu().numpy().T
        val = _gs_residual(chi_te, kchi(Dt[te]))
        loss_val.append(val)
        if np.isfinite(val) and val < best_val:
            with pt.no_grad():
                best_val = val
                best_chi = net(D0).cpu().numpy().copy()
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            keff = int((best_chi.std(0) > 0.05).sum()) if best_chi is not None else 0
            print(f"[k={k}] iter {it+1}/{n_iter}  train {loss_tr[-1]:.4e}  "
                  f"val {val:.4e}  k_eff {keff}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    if best_chi is None:
        net.eval()
        with pt.no_grad():
            best_chi = net(D0).cpu().numpy()
    net.eval()
    return dict(net=net, chi=best_chi.astype(np.float32),
                loss_train=np.array(loss_tr, np.float32),
                loss_val=np.array(loss_val, np.float32),
                n_iter=len(loss_val), k=k, hidden=list(hidden))
