"""Standalone pathway compute (decoupled from the nbconvert cell timeout).
Mirrors the notebook §4 task-building exactly, saves pathways.pkl in the format the
notebook loads: {"prepared": ..., "results": (method,i)->[res,...]}."""
import os, sys, time, pickle
import numpy as np, torch as pt
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
import pipeline as pl
from amore.isokann import ChiNetMulti

D = "/scratch/htc/jkresse/amore_pathway"
STATES = [0, 1, 2]
MEP_STEPS, MFEP_STEPS, EF_STEPS, MFEP_LS, EF_LS = 30, 30, 70, 50, 3

if __name__ == "__main__":
    model = ChiNetMulti(231, 3, hidden=[128, 32, 8])
    model.load_state_dict(pt.load(D + "/isokann_k3.pt")); model.eval()
    prepared = np.load(D + "/prep_seeds.npy", allow_pickle=True).item()
    tasks = []
    for i in STATES:
        for x in prepared[i]:
            tasks.append((("mep", i), "mep", "face", i, None, x, dict(steps=MEP_STEPS, energy_max_iter=25)))
            tasks.append((("mfep", i), "mfep", "face", i, None, x, dict(steps=MFEP_STEPS, steps_per_levelset=MFEP_LS)))
            tasks.append((("ef", i), "ef", "face", i, None, x, dict(steps=EF_STEPS, steps_per_levelset=EF_LS)))
    print(f"integrating {len(tasks)} pathways on 14 procs ...", flush=True)
    t0 = time.time()
    raw_out = pl.run_ensemble(model, 231, 3, [128, 32, 8], tasks, procs=14)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    results = defaultdict(list)
    for gid, res in raw_out:
        results[gid].append(res)
    with open(D + "/pathways.pkl", "wb") as f:
        pickle.dump({"prepared": prepared, "results": dict(results)}, f)
    for m in ["mep", "mfep", "ef"]:
        n = sum(len(results.get((m, i), [])) for i in STATES)
        print(f"  {m}: {n} paths", flush=True)
    print("saved pathways.pkl", flush=True)
