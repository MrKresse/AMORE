"""
Arm K — ISOKANN+AMORE with the CR2 RealTimeKernel as the plug-in conditional
expectation.

This is ORCHESTRATION ONLY. Every transform/target is imported from src/amore:
    - amore.isotarget.shiftscale   (1D warm-up target)
    - amore.isotarget.isa_target   (k-D simplex target)
    - amore.isokann.ChiNetMultiLinear  (Tanh hidden, linear output)
Nothing mathematical is re-implemented here.

The ONLY thing that makes this "Arm K" rather than any other ISOKANN run is how
the Koopman conditional expectation is formed:

    E[chi(x_{t+tau}) | x_t]  =  T @ chi(X)

where T is CellRank2's exact RealTimeKernel transition matrix (row-stochastic,
read from artifacts/T.npz — built by 01_cr2_reproduce.py with the pinned WOT
parameters). In the MD ISOKANN this expectation is a burst average over sampled
endpoints; here the kernel gives it in closed form as one sparse matmul. No
other kernel is substituted.

Schedule (fixed by the task):
    warm-up : k=1 chi trained to the ShiftScale isotarget
    main    : k=K chi (K = CR2 macrostate count) trained to the ISA isotarget,
              the k-D net initialised from the warm-up (shared trunk; output row
              0 = warm-up committor, rows 1..K-1 = small perturbations of it).

Monitoring-only generalisation curve (confirmed with user): 10% of the kernel's
transition entries are held out. They never enter the training target; we only
evaluate the ISA-target MSE under the held-out operator each iteration. This
influences no decision (training runs a fixed schedule, no early stopping).

Outputs -> artifacts/armK/:
    chi.npy              (N, K)  final memberships
    chi_warmup.npy       (N, 1)  1D warm-up committor
    loss_train.npy       (iters,)  invariance (ISA-target) loss on training operator
    loss_monitor.npy     (iters,)  same loss under the held-out operator
    sd_history.npy       (iters, K)  per-mode SD over training
    net.pt               trained k-D ChiNetMultiLinear state_dict
    meta.json            schedule, k_eff, device, timing
"""

from __future__ import annotations
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import math from src/amore — no local reimplementation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
from amore.isotarget import shiftscale, isa_target          # noqa: E402
from amore.isokann import ChiNetMultiLinear                 # noqa: E402

import config as C                                           # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SEED_OVERRIDE re-seeds the whole run (warm-up, ISA init, vertex search, operator split).
# ISA at small k can land in a vertex local-optimum (e.g. k=3 missing a small terminal); a
# different seed escapes it. Default = config SEED, so existing runs are unchanged.
C.SEED = int(os.environ.get("SEED_OVERRIDE", C.SEED))
# ARMK_SUFFIX lets a k-sweep write to its own dir (e.g. armK_k4) without clobbering the
# canonical armK. Empty by default.
OUT = os.path.join(C.ARTIFACTS, "armK" + os.environ.get("ARMK_SUFFIX", ""))
os.makedirs(OUT, exist_ok=True)


# ── Operator (conditional-expectation) helpers ─────────────────────────────────

def load_operator() -> tuple[sp.csr_matrix, np.ndarray]:
    """Load the CR2 RealTimeKernel transition matrix T and the ISOKANN features."""
    T = sp.load_npz(os.path.join(C.ARTIFACTS, "T.npz")).tocsr()
    X = np.load(os.path.join(C.ARTIFACTS, "features.npy")).astype(np.float32)
    assert T.shape[0] == T.shape[1] == X.shape[0], "T and features disagree on N"
    return T, X


def split_operator(T: sp.csr_matrix, frac: float, seed: int
                   ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Split the transition ENTRIES (the pairs) into a train operator (1-frac of the
    mass) and a monitor operator (frac), each row-renormalised to stochastic.
    Rows emptied by the split fall back to a self-transition so T @ chi stays
    well defined (a self-pair is the identity for the invariance objective and
    contributes no spurious signal).
    """
    rng = np.random.default_rng(seed)
    coo = T.tocoo()
    is_mon = rng.random(coo.nnz) < frac

    def _build(mask):
        m = sp.coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])),
                          shape=T.shape).tocsr()
        rs = np.asarray(m.sum(1)).ravel()
        empty = rs <= 0
        if empty.any():                          # self-transition for emptied rows
            idx = np.where(empty)[0]
            m = m + sp.csr_matrix((np.ones(len(idx)), (idx, idx)), shape=T.shape)
            rs = np.asarray(m.sum(1)).ravel()
        m = sp.diags(1.0 / rs) @ m               # row-renormalise
        return m.tocsr()

    return _build(~is_mon), _build(is_mon)


def koopman_expectation(T_t: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """E[chi(x_tau)|x] = T @ chi.  T_t: sparse (N,N) on DEVICE; chi: (N,k). -> (N,k)."""
    return torch.sparse.mm(T_t, chi)


def to_sparse_torch(T: sp.csr_matrix) -> torch.Tensor:
    coo = T.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, T.shape, device=DEVICE).coalesce()


# ── Network init: copy warm-up trunk into the k-D net ──────────────────────────

def init_from_warmup(warm: ChiNetMultiLinear, main: ChiNetMultiLinear,
                     k: int, noise: float = 0.05, seed: int = 0) -> None:
    """
    Share the hidden trunk; seed every output row from the warm-up committor so
    the ISA deflation starts from the dominant coordinate (as the task's
    'initialised from the warm-up' requires) rather than from noise.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    wlayers = list(warm.net)
    mlayers = list(main.net)
    # All layers except the final Linear are identical in shape -> copy verbatim.
    for wl, ml in zip(wlayers[:-1], mlayers[:-1]):
        if isinstance(wl, nn.Linear):
            ml.weight.data.copy_(wl.weight.data)
            ml.bias.data.copy_(wl.bias.data)
    w_last, m_last = wlayers[-1], mlayers[-1]    # Linear(h,1) -> Linear(h,k)
    base_w = w_last.weight.data[0]               # (h,)
    base_b = w_last.bias.data[0]
    for i in range(k):
        m_last.weight.data[i] = base_w + noise * torch.randn(base_w.shape, generator=g)
        m_last.bias.data[i]   = base_b + noise * torch.randn(1, generator=g).item()


# ── Training ───────────────────────────────────────────────────────────────────

def train_warmup(X_t: torch.Tensor, T_tr: torch.Tensor, in_dim: int) -> ChiNetMultiLinear:
    """1D ISOKANN warm-up with the ShiftScale isotarget."""
    torch.manual_seed(C.SEED)
    net = ChiNetMultiLinear(in_dim, k=1, hidden=C.ISOKANN_HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=C.LR_DECAY)
    print(f"[warm-up] ShiftScale 1D  ({C.WARMUP_ITERS} iters)")
    for it in range(C.WARMUP_ITERS):
        net.eval()
        with torch.no_grad():
            chi = net(X_t)                                   # (N,1)
            kchi = koopman_expectation(T_tr, chi)            # (N,1) = T@chi
            tgt = torch.tensor(shiftscale(kchi.cpu().numpy().T).T,  # (N,1) in [0,1]
                               dtype=torch.float32, device=DEVICE)
        net.train()
        for _ in range(C.EPOCHS_PER_ITER):
            idx = torch.randperm(X_t.shape[0], device=DEVICE)[:C.BATCH]
            loss = F.mse_loss(net(X_t[idx]), tgt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP)
            opt.step()
        sched.step()
        if (it + 1) % 25 == 0 or it == 0:
            with torch.no_grad():
                sd = float(net(X_t).std())
            print(f"  warm it={it+1:4d}  loss={loss.item():.5f}  chi_sd={sd:.4f}")
    return net


def isa_mse(chi_np: np.ndarray, kchi_np: np.ndarray) -> tuple[float, np.ndarray | None]:
    """ISA-target MSE = invariance loss. Returns (mse, target) or (nan, None) if degenerate."""
    try:
        tgt = isa_target(chi_np.T, kchi_np.T)        # transforms take (k,n); -> (k,n)
    except ValueError:
        return float("nan"), None
    tgt = tgt.T                                       # (n,k)
    return float(np.mean((chi_np - tgt) ** 2)), tgt


def train_main(warm: ChiNetMultiLinear, X_t: torch.Tensor,
               T_tr: torch.Tensor, T_mon: torch.Tensor, in_dim: int) -> dict:
    """k-D ISA main loop, initialised from the warm-up."""
    k = C.K_CHI
    torch.manual_seed(C.SEED + 1)
    net = ChiNetMultiLinear(in_dim, k=k, hidden=C.ISOKANN_HIDDEN).to(DEVICE)
    init_from_warmup(warm, net, k=k, seed=C.SEED)
    opt = torch.optim.Adam(net.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=C.LR_DECAY)

    loss_tr, loss_mon, sd_hist = [], [], []
    print(f"[main] ISA k={k}  ({C.MAIN_ITERS} iters)")
    for it in range(C.MAIN_ITERS):
        net.eval()
        with torch.no_grad():
            chi = net(X_t)                                   # (N,k)
            kchi = koopman_expectation(T_tr, chi)            # T_train @ chi
            chi_np, kchi_np = chi.cpu().numpy(), kchi.cpu().numpy()
        mse_tr, tgt = isa_mse(chi_np, kchi_np)
        loss_tr.append(mse_tr)
        # monitoring-only loss under the held-out operator (no gradient)
        with torch.no_grad():
            kchi_m = koopman_expectation(T_mon, chi).cpu().numpy()
        mse_m, _ = isa_mse(chi_np, kchi_m)
        loss_mon.append(mse_m)
        sd_hist.append(chi_np.std(0))

        if tgt is not None:
            tgt_t = torch.tensor(tgt, dtype=torch.float32, device=DEVICE)
            net.train()
            for _ in range(C.EPOCHS_PER_ITER):
                idx = torch.randperm(X_t.shape[0], device=DEVICE)[:C.BATCH]
                loss = F.mse_loss(net(X_t[idx]), tgt_t[idx])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP)
                opt.step()
            sched.step()

        if (it + 1) % 25 == 0 or it == 0:
            sd = sd_hist[-1]
            keff = int((sd > C.SD_LIVE_THRESHOLD).sum())
            print(f"  main it={it+1:4d}  loss_tr={mse_tr:.5f}  loss_mon={mse_m:.5f}  "
                  f"sd=[{', '.join(f'{s:.3f}' for s in sd)}]  k_eff={keff}")

    net.eval()
    with torch.no_grad():
        chi_final = net(X_t).cpu().numpy()
    return {"net": net, "chi": chi_final,
            "loss_tr": np.array(loss_tr, np.float32),
            "loss_mon": np.array(loss_mon, np.float32),
            "sd_hist": np.array(sd_hist, np.float32)}


def main():
    t0 = time.perf_counter()
    print(f"Device: {DEVICE}")
    T, X = load_operator()
    N, in_dim = X.shape
    print(f"N={N}  features={in_dim}  T.nnz={T.nnz}  k={C.K_CHI}")

    T_tr_sp, T_mon_sp = split_operator(T, C.PAIR_MONITOR_FRAC, C.SEED)
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    T_tr = to_sparse_torch(T_tr_sp)
    T_mon = to_sparse_torch(T_mon_sp)

    warm = train_warmup(X_t, T_tr, in_dim)
    with torch.no_grad():
        chi_w = warm(X_t).cpu().numpy()
    np.save(os.path.join(OUT, "chi_warmup.npy"), chi_w)

    res = train_main(warm, X_t, T_tr, T_mon, in_dim)

    np.save(os.path.join(OUT, "chi.npy"), res["chi"])
    np.save(os.path.join(OUT, "loss_train.npy"), res["loss_tr"])
    np.save(os.path.join(OUT, "loss_monitor.npy"), res["loss_mon"])
    np.save(os.path.join(OUT, "sd_history.npy"), res["sd_hist"])
    torch.save(res["net"].state_dict(), os.path.join(OUT, "net.pt"))

    sd_final = res["sd_hist"][-1]
    meta = {
        "N": int(N), "in_dim": int(in_dim), "k": int(C.K_CHI),
        "device": str(DEVICE), "T_nnz": int(T.nnz),
        "warmup_iters": C.WARMUP_ITERS, "main_iters": C.MAIN_ITERS,
        "epochs_per_iter": C.EPOCHS_PER_ITER, "batch": C.BATCH,
        "pair_monitor_frac": C.PAIR_MONITOR_FRAC,
        "sd_final": [float(s) for s in sd_final],
        "k_eff": int((sd_final > C.SD_LIVE_THRESHOLD).sum()),
        "hidden": C.ISOKANN_HIDDEN,
        "elapsed_s": time.perf_counter() - t0,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone in {meta['elapsed_s']:.0f}s. k_eff={meta['k_eff']}  "
          f"sd_final={meta['sd_final']}")


if __name__ == "__main__":
    main()
