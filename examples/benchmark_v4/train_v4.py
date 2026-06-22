# -*- coding: utf-8 -*-
"""
benchmark_v4 — ISOKANN on vacuum ADP (300 K), k=3, 231 pairwise distances.
Five variants: isa, gramschmidt, svd_power, gs_isa, ssm_isa (FIXED isotarget).

Lag conditions (via args), all sharing the same anchors:
  --xtau_tags T300_m50               5 ps        -> runs_v4/{variant}/
  --xtau_tags T300_0p1  --suffix _0p1   0.1 ps   -> runs_v4/{variant}_0p1/
  --xtau_tags T300_m50,T300_0p1 --suffix _mt  multitau (K=2) -> runs_v4/{variant}_mt/
"""
import os, sys, time, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); RUNS = os.path.join(HERE, "runs_v4"); os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from amore.isotarget import apply_target, gramschmidt_target, shiftscale
from amore.isokann import ChiNetMultiLinear, power_method_multi

ap = argparse.ArgumentParser()
ap.add_argument("--x0_tag", default="T300_m50")
ap.add_argument("--xtau_tags", default="T300_m50")   # comma list; >1 = multitau
ap.add_argument("--suffix", default="")              # output subdir suffix
args = ap.parse_args()
XTAUS = args.xtau_tags.split(",")

DEVICE = torch.device("cpu"); K = 3; N_SEEDS = 3
MAX_ITER, MIN_ITER, W, REL_TOL = 3000, 500, 300, 1e-3
WARMUP, LR, GRAD_CLIP = 100, 1e-3, 5.0
N_TRAIN = 25000; POWER_N_ITER, POWER_EPOCHS = 100, 50
VARIANTS = ["isa", "gramschmidt", "svd_power", "gs_isa", "ssm_isa"]
WARMTYPE = {"isa": None, "gramschmidt": None, "gs_isa": "gramschmidt", "ssm_isa": "shiftscale"}
BASE     = {"isa": "isa", "gramschmidt": "gramschmidt", "gs_isa": "isa", "ssm_isa": "isa"}

PAIRS = np.array([(i, j) for i in range(22) for j in range(i + 1, 22)])
def featurize(coords):
    x = coords.reshape(len(coords), 22, 3); d = x[:, PAIRS[:, 0], :] - x[:, PAIRS[:, 1], :]
    return np.linalg.norm(d, axis=2).astype(np.float32)

F0 = featurize(np.load(os.path.join(DATA, f"vac_X0_{args.x0_tag}.npy")))
FT = [featurize(np.load(os.path.join(DATA, f"vac_Xtau_{t}.npy"))) for t in XTAUS]  # list of (N,231)
N_all, IN, nlag = len(F0), F0.shape[1], len(FT)
print(f"data x0={args.x0_tag} xtau={XTAUS} (K={nlag}) -> {N_all} anchors, {IN} feats, suffix='{args.suffix}'")
feat_all = torch.tensor(F0, device=DEVICE)

def shiftscale_rows(kchi): return np.stack([shiftscale(kchi[r:r+1, :]).ravel() for r in range(kchi.shape[0])])
def warm_target(wt, kc): return gramschmidt_target(kc) if wt == "gramschmidt" else shiftscale_rows(kc)
def chi_sd(c): return c.std(0)
def plateau(vh, it):
    if it < MIN_ITER or len(vh) < W: return False
    r = np.array(vh[-W:]); r = r[np.isfinite(r)]
    return len(r) >= W//2 and (r.max()-r.min()) < REL_TOL*max(abs(np.median(r)), 1e-12)

def run_isotarget(variant, seed):
    torch.manual_seed(seed*12345+7); np.random.seed(seed*12345+7)
    idx = np.random.choice(N_all, min(N_TRAIN, N_all), replace=False)
    f0 = torch.tensor(F0[idx], device=DEVICE)
    fts = [torch.tensor(ft[idx], device=DEVICE) for ft in FT]
    m = np.random.rand(len(idx)) < 0.8; tr, te = np.where(m)[0], np.where(~m)[0]
    net = ChiNetMultiLinear(IN, K, hidden=[128, 32, 8]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    wt, base = WARMTYPE[variant], BASE[variant]
    vh, sdh, best_val, best_chi = [], [], np.inf, None
    def kchi(sub):                                   # mean over lags of chi(endpoint)
        net.eval()
        with torch.no_grad():
            return np.mean([net(ft[sub]).cpu().numpy().T for ft in fts], axis=0)
    for it in range(MAX_ITER):
        net.eval()
        with torch.no_grad(): chi0 = net(f0[tr]).cpu().numpy().T
        kc = kchi(tr)
        try:
            tgt = warm_target(wt, kc) if (wt and it < WARMUP) else apply_target(base, chi0, kc)
        except (ValueError, np.linalg.LinAlgError):
            vh.append(np.nan)
            with torch.no_grad(): sdh.append(chi_sd(net(feat_all).cpu().numpy()));
            continue
        tt = torch.tensor(tgt.T, dtype=torch.float32, device=DEVICE)
        net.train(); loss = nn.functional.mse_loss(net(f0[tr]), tt)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP); opt.step()
        net.eval()
        with torch.no_grad(): chi_te = net(f0[te]).cpu().numpy().T
        kc_te = kchi(te)
        try:
            tgt_te = apply_target(base, chi_te, kc_te)
            with torch.no_grad():
                val = float(nn.functional.mse_loss(net(f0[te]), torch.tensor(tgt_te.T, dtype=torch.float32, device=DEVICE)))
        except (ValueError, np.linalg.LinAlgError):
            val = np.nan
        vh.append(val)
        with torch.no_grad(): chi_all = net(feat_all).cpu().numpy()
        sdh.append(chi_sd(chi_all))
        if np.isfinite(val) and val < best_val: best_val, best_chi = val, chi_all.copy()
        if plateau(vh, it): break
    if best_chi is None:
        with torch.no_grad(): best_chi = net(feat_all).cpu().numpy()
    return np.array(vh, np.float32), np.array(sdh, np.float32), best_chi.astype(np.float32)

def run_power(seed):
    torch.manual_seed(seed*12345+7); np.random.seed(seed*12345+7)
    net = ChiNetMultiLinear(IN, K, hidden=[128, 32, 8]).to(DEVICE)
    x0 = torch.tensor(np.tile(F0, (nlag, 1)), device=DEVICE)        # anchors repeated per lag
    x1 = torch.tensor(np.concatenate(FT, axis=0), device=DEVICE)    # all endpoints
    res = power_method_multi(net, x0, x1, n_iter=POWER_N_ITER, epochs_per_iter=POWER_EPOCHS, lr=LR, verbose=False)
    net.eval()
    with torch.no_grad(): chi = net(feat_all).cpu().numpy().astype(np.float32)
    return np.array(res["losses"], np.float32), (np.array(res["spans"])/(2*np.sqrt(3))).astype(np.float32), chi

for variant in VARIANTS:
    print(f"\n=== {variant}{args.suffix} ===")
    for seed in range(N_SEEDS):
        d = os.path.join(RUNS, variant + args.suffix, f"seed_{seed}"); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "chi_best.npy")): print(f"  seed{seed} [done]"); continue
        t0 = time.perf_counter()
        vl, sd, chi = run_power(seed) if variant == "svd_power" else run_isotarget(variant, seed)
        np.save(os.path.join(d, "chi_best.npy"), chi); np.save(os.path.join(d, "val_loss.npy"), vl)
        np.save(os.path.join(d, "chi_sd_history.npy"), sd)
        sdf = sd[-1] if len(sd) else np.zeros(K)
        print(f"  seed{seed} iters={len(vl):4d} sd=[{', '.join(f'{s:.3f}' for s in sdf)}] "
              f"k_eff={int((sdf>0.05).sum())} t={time.perf_counter()-t0:.0f}s", flush=True)
print(f"\nV4 TRAIN DONE ({args.suffix or '5ps'})")
