# -*- coding: utf-8 -*-
"""
Re-run Python ISA WITH the standard 100-iter warm-up but using the FIXED
isa_target (transpose restored). Output -> runs/{ds}/isa_fixed/seed_*.
This shows whether the original benchmark's ISA (previously r~0.80 with k_eff=0,
a 'hollow PASS') works correctly once the transpose bug is fixed.
"""
import os, sys, json, time, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "benchmark", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

spec = importlib.util.spec_from_file_location(
    "train_v3b", os.path.join(HERE, "02_train_benchmark_v3.py"))
t2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2)
from amore.isokann import ChiNetMultiLinear

# WARMUP stays at the default (100); only the isa_target fix differs from the
# original runs/{ds}/isa.
DEVICE = t2.DEVICE
K, N_SEEDS = 3, t2.N_SEEDS
print(f"Device: {DEVICE}  WARMUP={t2.WARMUP}  (ISA fixed, output -> isa_fixed)")


def run_ds(ds_name, feat, bursts, splits):
    print(f"\n{'='*60}\n{ds_name}  feat={feat.shape}  WARMUP={t2.WARMUP}\n{'='*60}")
    in_dim = feat.shape[1]
    feat_t = torch.tensor(feat, dtype=torch.float32, device=DEVICE)
    bursts_t = torch.tensor(bursts, dtype=torch.float32, device=DEVICE)
    for seed in range(N_SEEDS):
        out_dir = os.path.join(HERE, "runs", ds_name, "isa_fixed", f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(os.path.join(out_dir, "chi_best.npy")):
            print(f"  seed={seed} [done]"); continue
        torch.manual_seed(seed * 12345 + 7)
        net = ChiNetMultiLinear(in_dim, K, hidden=[128, 32, 8]).to(DEVICE)
        split = splits[seed]
        f_tr, f_te = feat_t[split == 0], feat_t[split == 1]
        b_tr, b_te = bursts_t[split == 0], bursts_t[split == 1]
        t0 = time.perf_counter()
        res = t2.run_isotarget("isa", f_tr, f_te, feat_t, b_tr, b_te, net, seed=seed)
        for k in ("val_loss", "chi_sd_history", "chi_atstop", "chi_best"):
            np.save(os.path.join(out_dir, k + ".npy"), res[k])
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump({"n_iter": res["n_iter"], "elapsed_s": res["elapsed_s"],
                       "warmup": t2.WARMUP, "fix": "isa transpose"}, fh)
        sd = res["chi_sd_history"][-1] if len(res["chi_sd_history"]) else np.zeros(K)
        print(f"  seed={seed}  iters={res['n_iter']:4d}  "
              f"sd=[{', '.join(f'{s:.3f}' for s in sd)}]  "
              f"k_eff={int((sd>0.05).sum())}  t={time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    tw = np.load(os.path.join(DATA, "triple_well_koopman.npz"))
    al = np.load(os.path.join(DATA, "alanine_koopman.npz"))
    al01 = np.load(os.path.join(DATA, "alanine_0p1ps_koopman.npz"))
    run_ds("triple_well", tw["anchors"], tw["bursts"], tw["patch_splits"])
    run_ds("alanine_5ps", al["anchors_feat"], al["bursts_feat"], al["patch_splits"])
    run_ds("alanine_0p1ps", al["anchors_feat"], al01["bursts_feat"], al["patch_splits"])
    joint = np.concatenate([al["bursts_feat"], al01["bursts_feat"]], axis=1)
    run_ds("alanine_multitau", al["anchors_feat"], joint, al["patch_splits"])
    print("\nPY ISA-FIXED (warmup) DONE")
