"""
12 — Arm K ISOKANN trained on FULL HVG expression (no PCA bottleneck).

Motivation (user): the PC-based gradient is biased by PC-representation
(Spearman(grad, loading_norm)=0.977), so rare-lineage TFs are invisible to it.
With genes as DIRECT inputs, ∂χ/∂gene is a genuine per-gene attribution that can
express nonlinear/causal effects — the real test of whether the gradient can
beat correlation/GPCCA at driver recovery.

Same plug-in conditional expectation as 03 (kchi = T_RealTimeKernel @ chi) and the
same ShiftScale-1D → ISA-kD schedule; only the input changes (1576 z-scored HVGs)
and the net gets BatchNorm at the input (per the project's HVG architecture notes:
Tanh hidden, linear output, BN at input). All transforms imported from src/amore.

Outputs -> artifacts/armK_hvg/: chi.npy, net.pt, loss_train/monitor.npy, sd_history.npy, meta.json
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
from amore.isotarget import shiftscale, isa_target          # noqa: E402
import config as C                                           # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(C.ARTIFACTS, "armK_hvg" + os.environ.get("ARMK_SUFFIX", "")); os.makedirs(OUT, exist_ok=True)

# lighter schedule than the PC model (input is 30x wider) — keep CPU time sane
HIDDEN = [256, 128, 64]
WARMUP_ITERS = 100
MAIN_ITERS   = 400
EPOCHS_PER_ITER = 30
LR = 5e-4
BATCH = 4096
SEED = 0


class ChiNetHVG(nn.Module):
    """BatchNorm(input) -> [Linear, Tanh]* -> Linear (k).  Linear output for ISA."""
    def __init__(self, in_dim, k, hidden):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        dims = [in_dim] + list(hidden) + [k]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.bn(x))


def split_operator(T, frac, seed):
    rng = np.random.default_rng(seed); coo = T.tocoo()
    is_mon = rng.random(coo.nnz) < frac
    def _build(mask):
        m = sp.coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=T.shape).tocsr()
        rs = np.asarray(m.sum(1)).ravel(); empty = rs <= 0
        if empty.any():
            idx = np.where(empty)[0]
            m = m + sp.csr_matrix((np.ones(len(idx)), (idx, idx)), shape=T.shape)
            rs = np.asarray(m.sum(1)).ravel()
        return (sp.diags(1.0 / rs) @ m).tocsr()
    return _build(~is_mon), _build(is_mon)


def to_sparse_torch(T):
    coo = T.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, T.shape, device=DEVICE).coalesce()


def kexp(T_t, chi):
    return torch.sparse.mm(T_t, chi)


def init_from_warmup(warm, main, k, noise=0.05, seed=0):
    main.bn.load_state_dict(warm.bn.state_dict())
    wl, ml = list(warm.net), list(main.net)
    for a, b in zip(wl[:-1], ml[:-1]):
        if isinstance(a, nn.Linear):
            b.weight.data.copy_(a.weight.data); b.bias.data.copy_(a.bias.data)
    g = torch.Generator().manual_seed(seed)
    bw, bb = wl[-1].weight.data[0], wl[-1].bias.data[0]
    for i in range(k):
        ml[-1].weight.data[i] = bw + noise * torch.randn(bw.shape, generator=g)
        ml[-1].bias.data[i] = bb + noise * torch.randn(1, generator=g).item()


def isa_mse(chi_np, kchi_np):
    try:
        tgt = isa_target(chi_np.T, kchi_np.T).T
    except ValueError:
        return float("nan"), None
    return float(np.mean((chi_np - tgt) ** 2)), tgt


def main():
    t0 = time.perf_counter(); print(f"Device {DEVICE}  threads {torch.get_num_threads()}")
    T = sp.load_npz(os.path.join(C.ARTIFACTS, "T.npz")).tocsr()
    X = np.load(os.path.join(C.ARTIFACTS, "hvg_expr.npy")).astype(np.float32)
    N, in_dim = X.shape; k = C.K_CHI
    print(f"N={N} HVG_in={in_dim} k={k} T.nnz={T.nnz}")
    Tr, Mon = split_operator(T, C.PAIR_MONITOR_FRAC, SEED)
    Xt = torch.tensor(X, device=DEVICE); T_tr = to_sparse_torch(Tr); T_mon = to_sparse_torch(Mon)

    # ── warm-up: 1D ShiftScale ──
    torch.manual_seed(SEED)
    warm = ChiNetHVG(in_dim, 1, HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(warm.parameters(), lr=LR)
    print(f"[warm] ShiftScale 1D ({WARMUP_ITERS})")
    for it in range(WARMUP_ITERS):
        warm.eval()
        with torch.no_grad():
            kc = kexp(T_tr, warm(Xt))
            tgt = torch.tensor(shiftscale(kc.cpu().numpy().T).T, dtype=torch.float32, device=DEVICE)
        warm.train()
        for _ in range(EPOCHS_PER_ITER):
            idx = torch.randperm(N, device=DEVICE)[:BATCH]
            loss = F.mse_loss(warm(Xt[idx]), tgt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(warm.parameters(), C.GRAD_CLIP); opt.step()
        if (it + 1) % 25 == 0 or it == 0:
            with torch.no_grad():
                warm.eval(); sd = float(warm(Xt).std())
            print(f"  warm {it+1:4d} loss {loss.item():.5f} sd {sd:.4f}", flush=True)

    # ── main: ISA k-D ──
    torch.manual_seed(SEED + 1)
    net = ChiNetHVG(in_dim, k, HIDDEN).to(DEVICE)
    init_from_warmup(warm, net, k, seed=SEED)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_tr, loss_mon, sdh = [], [], []
    print(f"[main] ISA k={k} ({MAIN_ITERS})")
    for it in range(MAIN_ITERS):
        net.eval()
        with torch.no_grad():
            chi = net(Xt); chi_np = chi.cpu().numpy()
            kchi_np = kexp(T_tr, chi).cpu().numpy()
        mse_tr, tgt = isa_mse(chi_np, kchi_np); loss_tr.append(mse_tr)
        with torch.no_grad():
            kmon = kexp(T_mon, chi).cpu().numpy()
        mse_m, _ = isa_mse(chi_np, kmon); loss_mon.append(mse_m); sdh.append(chi_np.std(0))
        if tgt is not None:
            tt = torch.tensor(tgt, dtype=torch.float32, device=DEVICE)
            net.train()
            for _ in range(EPOCHS_PER_ITER):
                idx = torch.randperm(N, device=DEVICE)[:BATCH]
                loss = F.mse_loss(net(Xt[idx]), tt[idx])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP); opt.step()
        if (it + 1) % 25 == 0 or it == 0:
            sd = sdh[-1]; keff = int((sd > C.SD_LIVE_THRESHOLD).sum())
            print(f"  main {it+1:4d} loss_tr {mse_tr:.5f} loss_mon {mse_m:.5f} "
                  f"sd [{', '.join(f'{s:.3f}' for s in sd)}] k_eff {keff}", flush=True)

    net.eval()
    with torch.no_grad():
        chi_final = net(Xt).cpu().numpy()
    np.save(os.path.join(OUT, "chi.npy"), chi_final)
    np.save(os.path.join(OUT, "loss_train.npy"), np.array(loss_tr, np.float32))
    np.save(os.path.join(OUT, "loss_monitor.npy"), np.array(loss_mon, np.float32))
    np.save(os.path.join(OUT, "sd_history.npy"), np.array(sdh, np.float32))
    torch.save(net.state_dict(), os.path.join(OUT, "net.pt"))
    sd = sdh[-1]
    meta = {"N": int(N), "in_dim": int(in_dim), "k": int(k), "hidden": HIDDEN,
            "warmup_iters": WARMUP_ITERS, "main_iters": MAIN_ITERS, "lr": LR,
            "sd_final": [float(s) for s in sd], "k_eff": int((sd > C.SD_LIVE_THRESHOLD).sum()),
            "loss_tr_final": float(loss_tr[-1]), "loss_mon_final": float(loss_mon[-1]),
            "any_nan": bool(np.any(~np.isfinite(loss_tr))), "elapsed_s": time.perf_counter() - t0}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"\nDONE {meta['elapsed_s']:.0f}s k_eff={meta['k_eff']} sd={meta['sd_final']} "
          f"nan={meta['any_nan']}")


if __name__ == "__main__":
    main()
