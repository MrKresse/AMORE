"""
03b — CONTINUE the k=6 Arm-K PC model from its cached weights for a few hundred more
ISA iterations, and keep the BEST iterate (not the last). The from-scratch run
(03_train_armK.py) ends on whatever step iter 600 happens to be — and the ISA loss
spikes intermittently when the simplex vertex assignment flips, so the final iterate
can be a bad one (e.g. Neural top-30 purity stuck at 0.80). This warm-starts from
artifacts/armK/net.pt, runs EXTRA_ITERS more ISA steps at a low fine-tuning LR, and
saves the iterate with the lowest training ISA loss among non-collapsed steps.

Same operator (kchi = T_RealTimeKernel @ chi), same monitor split (C.SEED), same ISA
target as 03 — only the schedule continues. Writes back into artifacts/armK/ (the
iter-600 snapshot is preserved in artifacts/armK_iter600/).
"""
from __future__ import annotations
import os, sys, json, time, copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
from amore.isotarget import isa_target            # noqa: E402
from amore.isokann import ChiNetMultiLinear       # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C                                # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ARMK_SUFFIX continues a k-sweep alternate (e.g. armK_k6) instead of the primary armK.
OUT = os.path.join(C.ARTIFACTS, "armK" + os.environ.get("ARMK_SUFFIX", ""))

EXTRA_ITERS = int(os.environ.get("EXTRA_ITERS", 300))
LR = float(os.environ.get("CONT_LR", 3e-4))       # low fine-tuning LR (continuation)


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


def isa_mse(chi_np, kchi_np):
    try:
        tgt = isa_target(chi_np.T, kchi_np.T).T
    except ValueError:
        return float("nan"), None
    return float(np.mean((chi_np - tgt) ** 2)), tgt


def main():
    t0 = time.perf_counter()
    torch.manual_seed(C.SEED + 2)                 # reproducible continuation
    k = C.K_CHI
    T = sp.load_npz(os.path.join(C.ARTIFACTS, "T.npz")).tocsr()
    X = np.load(os.path.join(C.ARTIFACTS, "features.npy")).astype(np.float32)
    N, in_dim = X.shape
    print(f"continue: N={N} in_dim={in_dim} k={k} EXTRA_ITERS={EXTRA_ITERS} LR={LR}", flush=True)

    Tr, Mon = split_operator(T, C.PAIR_MONITOR_FRAC, C.SEED)
    Xt = torch.tensor(X, device=DEVICE)
    T_tr = to_sparse_torch(Tr); T_mon = to_sparse_torch(Mon)

    net = ChiNetMultiLinear(in_dim, k, C.ISOKANN_HIDDEN).to(DEVICE)
    net.load_state_dict(torch.load(os.path.join(OUT, "net.pt"), map_location=DEVICE))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=C.LR_DECAY)

    # seed the "best" tracker with the current cached iterate
    def eval_state():
        net.eval()
        with torch.no_grad():
            chi = net(Xt); chi_np = chi.cpu().numpy()
            kchi_np = kexp(T_tr, chi).cpu().numpy()
        mse_tr, _ = isa_mse(chi_np, kchi_np)
        with torch.no_grad():
            kmon = kexp(T_mon, chi).cpu().numpy()
        mse_m, _ = isa_mse(chi_np, kmon)
        sd = chi_np.std(0)
        return chi_np, mse_tr, mse_m, sd

    chi0, best_tr, best_mon, sd0 = eval_state()
    best = {"loss_tr": best_tr, "loss_mon": best_mon, "sd": sd0,
            "chi": chi0, "state": copy.deepcopy(net.state_dict())}
    print(f"  start  loss_tr={best_tr:.5f} loss_mon={best_mon:.5f} "
          f"sd=[{', '.join(f'{s:.3f}' for s in sd0)}]", flush=True)

    loss_tr, loss_mon, sdh = [], [], []
    for it in range(EXTRA_ITERS):
        net.eval()
        with torch.no_grad():
            chi = net(Xt); chi_np = chi.cpu().numpy()
            kchi_np = kexp(T_tr, chi).cpu().numpy()
        mse_tr, tgt = isa_mse(chi_np, kchi_np)
        with torch.no_grad():
            kmon = kexp(T_mon, chi).cpu().numpy()
        mse_m, _ = isa_mse(chi_np, kmon)
        sd = chi_np.std(0)
        loss_tr.append(mse_tr); loss_mon.append(mse_m); sdh.append(sd)

        keff = int((sd > C.SD_LIVE_THRESHOLD).sum())
        # keep the best NON-COLLAPSED iterate by training ISA loss
        if keff == k and np.isfinite(mse_tr) and mse_tr < best["loss_tr"]:
            best = {"loss_tr": mse_tr, "loss_mon": mse_m, "sd": sd,
                    "chi": chi_np, "state": copy.deepcopy(net.state_dict())}

        if tgt is not None:
            tt = torch.tensor(tgt, dtype=torch.float32, device=DEVICE)
            net.train()
            for _ in range(C.EPOCHS_PER_ITER):
                idx = torch.randperm(N, device=DEVICE)[:C.BATCH]
                loss = F.mse_loss(net(Xt[idx]), tt[idx])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP); opt.step()
            sched.step()
        if (it + 1) % 25 == 0 or it == 0:
            print(f"  cont it={it+1:4d} loss_tr={mse_tr:.5f} loss_mon={mse_m:.5f} "
                  f"sd=[{', '.join(f'{s:.3f}' for s in sd)}] k_eff={keff} "
                  f"best_tr={best['loss_tr']:.5f}", flush=True)

    # ── save BEST iterate back into armK/ ─────────────────────────────────────
    net.load_state_dict(best["state"]); net.eval()
    np.save(os.path.join(OUT, "chi.npy"), best["chi"])
    torch.save(best["state"], os.path.join(OUT, "net.pt"))
    # append the continuation histories to the originals
    def app(name, arr):
        p = os.path.join(OUT, name)
        old = np.load(p) if os.path.exists(p) else np.zeros((0,) + np.asarray(arr).shape[1:])
        np.save(p, np.concatenate([old, np.asarray(arr, np.float32)], axis=0))
    app("loss_train.npy", np.array(loss_tr, np.float32))
    app("loss_monitor.npy", np.array(loss_mon, np.float32))
    app("sd_history.npy", np.array(sdh, np.float32))

    meta = json.load(open(os.path.join(OUT, "meta.json")))
    meta["continued_extra_iters"] = EXTRA_ITERS
    meta["continue_lr"] = LR
    meta["sd_final"] = [float(s) for s in best["sd"]]
    meta["k_eff"] = int((best["sd"] > C.SD_LIVE_THRESHOLD).sum())
    meta["best_loss_tr"] = float(best["loss_tr"])
    meta["best_loss_mon"] = float(best["loss_mon"])
    meta["continue_elapsed_s"] = time.perf_counter() - t0
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"\nDONE. best loss_tr={best['loss_tr']:.5f} loss_mon={best['loss_mon']:.5f} "
          f"k_eff={meta['k_eff']} sd={meta['sd_final']}  ({meta['continue_elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
