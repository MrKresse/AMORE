"""
amore.scrna.koopman — ISOKANN training for single-cell data where the Koopman
conditional expectation is supplied by a *transition matrix* T (row-stochastic),
e.g. a CellRank kernel:

    E[chi(x_{t+tau}) | x_t]  =  T @ chi(X)

This is the "plug-in" expectation (Arm K): instead of burst-averaging sampled
dynamics, the kernel gives the Koopman action in closed form as one sparse matmul.

The training schedule is the standard ISOKANN warm-up -> ISA:
    warm-up : 1D chi to the ShiftScale isotarget (gives a committor-like coordinate)
    main    : k-D chi to the ISA isotarget (simplex memberships), the k-D net
              initialised from the warm-up.

All isotarget transforms are imported from amore.isotarget; nothing mathematical
is re-implemented here.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..isotarget import shiftscale, isa_target
from ..isokann.network import ChiNetHVG, ChiNetMultiLinear


# ── operator helpers ───────────────────────────────────────────────────────────

def to_sparse_torch(T: sp.spmatrix, device="cpu") -> torch.Tensor:
    """scipy sparse (row-stochastic) -> coalesced torch sparse_coo on `device`."""
    coo = T.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, T.shape, device=device).coalesce()


def koopman_expectation(T_t: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """E[chi(x_tau)|x] = T @ chi.  T_t sparse (N,N); chi (N,k) -> (N,k)."""
    return torch.sparse.mm(T_t, chi)


def split_operator(T: sp.spmatrix, frac: float, seed: int = 0
                   ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Split the transition ENTRIES (the pairs) into a train operator (1-frac of the
    mass) and a monitor operator (frac), each row-renormalised to stochastic.
    Rows emptied by the split fall back to a self-transition. Use the monitor
    operator only for a monitoring-only generalisation curve (no model selection).
    """
    rng = np.random.default_rng(seed)
    coo = T.tocoo()
    is_mon = rng.random(coo.nnz) < frac

    def _build(mask):
        m = sp.coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])),
                          shape=T.shape).tocsr()
        rs = np.asarray(m.sum(1)).ravel()
        empty = rs <= 0
        if empty.any():
            idx = np.where(empty)[0]
            m = m + sp.csr_matrix((np.ones(len(idx)), (idx, idx)), shape=T.shape)
            rs = np.asarray(m.sum(1)).ravel()
        return (sp.diags(1.0 / rs) @ m).tocsr()

    return _build(~is_mon), _build(is_mon)


# ── warm-up -> main initialisation ─────────────────────────────────────────────

def init_from_warmup(warm: nn.Module, main: nn.Module, k: int,
                     noise: float = 0.05, seed: int = 0) -> None:
    """
    Seed the k-D net from the 1D warm-up: copy the shared trunk (and input BN if
    present), then initialise every output row from the warm-up committor plus a
    small perturbation, so ISA deflation starts from the dominant coordinate.
    Works for ChiNetHVG (has .bn) and ChiNetMultiLinear (no .bn).
    """
    if hasattr(warm, "bn") and hasattr(main, "bn"):
        main.bn.load_state_dict(warm.bn.state_dict())
    wl, ml = list(warm.net), list(main.net)
    for a, b in zip(wl[:-1], ml[:-1]):
        if isinstance(a, nn.Linear):
            b.weight.data.copy_(a.weight.data)
            b.bias.data.copy_(a.bias.data)
    g = torch.Generator().manual_seed(seed)
    bw, bb = wl[-1].weight.data[0], wl[-1].bias.data[0]
    for i in range(k):
        ml[-1].weight.data[i] = bw + noise * torch.randn(bw.shape, generator=g)
        ml[-1].bias.data[i] = bb + noise * torch.randn(1, generator=g).item()


def _isa_mse(chi_np: np.ndarray, kchi_np: np.ndarray):
    """ISA-target MSE (invariance loss). Returns (mse, target(n,k)) or (nan, None)."""
    try:
        tgt = isa_target(chi_np.T, kchi_np.T).T
    except ValueError:
        return float("nan"), None
    return float(np.mean((chi_np - tgt) ** 2)), tgt


def _make_net(in_dim, k, hidden, batchnorm):
    return (ChiNetHVG(in_dim, k, hidden) if batchnorm
            else ChiNetMultiLinear(in_dim, k, hidden))


# ── main entry point ────────────────────────────────────────────────────────────

def train_chi(X: np.ndarray, T: sp.spmatrix, k: int, *,
              hidden=(256, 128, 64), batchnorm: bool = True,
              warmup_iters: int = 100, main_iters: int = 400,
              epochs_per_iter: int = 30, lr: float = 5e-4, lr_decay: float = 0.999,
              batch: int = 4096,
              grad_clip: float = 5.0, monitor_frac: float = 0.10,
              sd_live: float = 0.05, seed: int = 0, device: str | None = None,
              verbose: bool = True) -> dict:
    """
    Train a k-D ISOKANN chi on features X with the plug-in expectation kchi = T @ chi.

    Parameters
    ----------
    X : (N, F) features (e.g. HVG expression if batchnorm=True, or PCs).
    T : (N, N) row-stochastic transition matrix (scipy sparse).
    k : number of chi membership functions (e.g. CR2 macrostate count).
    batchnorm : ChiNetHVG (BN-input) for HVG features; else ChiNetMultiLinear (PCs).
    monitor_frac : fraction of transition entries held out for a monitoring-only
                   invariance curve (affects no decision).

    Returns dict: net, chi (N,k), loss_train, loss_monitor, sd_history, k_eff,
    sd_final, T_train, T_monitor.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, in_dim = X.shape
    Tr, Mon = split_operator(T, monitor_frac, seed)
    Xt = torch.tensor(np.asarray(X, np.float32), device=device)
    T_tr = to_sparse_torch(Tr, device); T_mon = to_sparse_torch(Mon, device)

    # warm-up: 1D ShiftScale
    torch.manual_seed(seed)
    warm = _make_net(in_dim, 1, hidden, batchnorm).to(device)
    opt = torch.optim.Adam(warm.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)
    for it in range(warmup_iters):
        warm.eval()
        with torch.no_grad():
            kc = koopman_expectation(T_tr, warm(Xt))
            tgt = torch.tensor(shiftscale(kc.cpu().numpy().T).T, dtype=torch.float32, device=device)
        warm.train()
        for _ in range(epochs_per_iter):
            idx = torch.randperm(N, device=device)[:batch]
            loss = F.mse_loss(warm(Xt[idx]), tgt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(warm.parameters(), grad_clip); opt.step()
        sched.step()
        if verbose and ((it + 1) % 25 == 0 or it == 0):
            with torch.no_grad():
                warm.eval(); sd = float(warm(Xt).std())
            print(f"  [warm] {it+1:4d} loss {loss.item():.5f} sd {sd:.4f}", flush=True)

    # main: ISA k-D
    torch.manual_seed(seed + 1)
    net = _make_net(in_dim, k, hidden, batchnorm).to(device)
    init_from_warmup(warm, net, k, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)
    loss_tr, loss_mon, sdh = [], [], []
    import copy
    best = {"mon": float("inf"), "chi": None, "state": None, "it": -1}
    for it in range(main_iters):
        net.eval()
        with torch.no_grad():
            chi = net(Xt); chi_np = chi.cpu().numpy()
            kchi_np = koopman_expectation(T_tr, chi).cpu().numpy()
        mse_tr, tgt = _isa_mse(chi_np, kchi_np); loss_tr.append(mse_tr)
        with torch.no_grad():
            kmon = koopman_expectation(T_mon, chi).cpu().numpy()
        mse_m, _ = _isa_mse(chi_np, kmon); loss_mon.append(mse_m); sdh.append(chi_np.std(0))
        # track best held-out (monitor) iterate among ALL-MODES-ALIVE states, so the
        # scored χ is the best generaliser, not the drifting final iterate (and never
        # a collapsed one with spuriously low invariance loss).
        if int((sdh[-1] > sd_live).sum()) == k and np.isfinite(mse_m) and mse_m < best["mon"]:
            best = {"mon": float(mse_m), "chi": chi_np.copy(),
                    "state": copy.deepcopy(net.state_dict()), "it": it}
        if tgt is not None:
            tt = torch.tensor(tgt, dtype=torch.float32, device=device)
            net.train()
            for _ in range(epochs_per_iter):
                idx = torch.randperm(N, device=device)[:batch]
                loss = F.mse_loss(net(Xt[idx]), tt[idx])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip); opt.step()
        if verbose and ((it + 1) % 25 == 0 or it == 0):
            sd = sdh[-1]; keff = int((sd > sd_live).sum())
            print(f"  [main] {it+1:4d} loss_tr {mse_tr:.5f} loss_mon {mse_m:.5f} "
                  f"sd [{', '.join(f'{s:.3f}' for s in sd)}] k_eff {keff}", flush=True)

    net.eval()
    with torch.no_grad():
        chi_final = net(Xt).cpu().numpy()
    sd_final = sdh[-1]
    # restore best-on-monitor weights for the returned net + chi (fallback to final)
    if best["state"] is not None:
        net.load_state_dict(best["state"]); net.eval()
        chi_best, sd_best = best["chi"], best["chi"].std(0)
    else:
        chi_best, sd_best = chi_final, sd_final
    return {"net": net, "chi": chi_best, "chi_final": chi_final,
            "best_iter": best["it"], "best_monitor": best["mon"],
            "loss_train": np.array(loss_tr, np.float32),
            "loss_monitor": np.array(loss_mon, np.float32),
            "sd_history": np.array(sdh, np.float32),
            "sd_final": [float(s) for s in sd_best],
            "k_eff": int((np.asarray(sd_best) > sd_live).sum()),
            "T_train": Tr, "T_monitor": Mon, "device": device}
