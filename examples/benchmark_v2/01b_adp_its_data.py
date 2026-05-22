# -*- coding: utf-8 -*-
"""
Generate multi-lag ADP burst data for the ITS plot.

Runs one trajectory of 5 ps per anchor (checkpoints at 0.1, 0.2, 0.5, 1.0,
2.0, 5.0 ps), using a random subset of N_SAMPLE anchors. For the ITS
plateau check, N_SAMPLE=1200, N_BURSTS=1 is sufficient and fits in ~10 min.

Output: data/alanine_multilag.npz
"""

import io, sys, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# suppress ALL tqdm progress bars (inner trajectory bars are noisy in log files)
os.environ["TQDM_DISABLE"] = "1"

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from amore.sims import OpenMMSimulation, phi, psi

HERE     = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "benchmark", "data")
OUT_FILE = os.path.join(DATA_DIR, "alanine_multilag.npz")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
TEMP              = 450.0
DT                = 2e-3
N_SAMPLE          = 1200      # anchors to use (fits in ~9 min on CPU)
N_BURSTS          = 1         # per anchor (1 is enough for ITS plateau check)
SAVE_EVERY_STEPS  = 50        # 0.1 ps per checkpoint frame
T_MAX_PS          = 5.0
CHECKPOINT_EVERY  = 100       # save partial results every N anchors
SEED              = 42

LAGS_PS   = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
FRAME_IDX = [int(round(l / (SAVE_EVERY_STEPS * DT))) for l in LAGS_PS]
N_LAGS    = len(LAGS_PS)

print(f"Lag times (ps)   : {LAGS_PS}")
print(f"Frame indices    : {FRAME_IDX}")
print(f"Anchors to use   : {N_SAMPLE}  (N_BURSTS={N_BURSTS})")
sys.stdout.flush()

# ── Load anchors ───────────────────────────────────────────────────────────────
v1 = np.load(os.path.join(DATA_DIR, "alanine_koopman.npz"))
anchors_cart = v1["anchors_cart"]             # (N_ANC_ALL, 22, 3)
N_ANC_ALL    = len(anchors_cart)

rng     = np.random.default_rng(SEED)
idx_sel = rng.choice(N_ANC_ALL, size=min(N_SAMPLE, N_ANC_ALL), replace=False)
idx_sel = np.sort(idx_sel)

anchors_flat = anchors_cart[idx_sel].reshape(len(idx_sel), -1).astype(np.float64)
anchors_phi_sel = v1["anchors_phi"][idx_sel]
anchors_psi_sel = v1["anchors_psi"][idx_sel]
N_ANC = len(idx_sel)
print(f"Selected {N_ANC} anchors out of {N_ANC_ALL}")
sys.stdout.flush()

# ── OpenMM ─────────────────────────────────────────────────────────────────────
sim = OpenMMSimulation(steps=int(T_MAX_PS / DT), dt=DT, temp=TEMP)
print(f"Sim: {sim}")
sys.stdout.flush()

# ── Run ────────────────────────────────────────────────────────────────────────
bursts_phi = np.empty((N_ANC, N_BURSTS, N_LAGS), dtype=np.float32)
bursts_psi = np.empty((N_ANC, N_BURSTS, N_LAGS), dtype=np.float32)

t0 = time.perf_counter()
for i in range(N_ANC):
    for k in range(N_BURSTS):
        traj = sim.trajectory(anchors_flat[i], T=T_MAX_PS,
                              save_every_steps=SAVE_EVERY_STEPS)
        for li, fi in enumerate(FRAME_IDX):
            frame = traj[fi]
            bursts_phi[i, k, li] = float(phi(frame))
            bursts_psi[i, k, li] = float(psi(frame))

    if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == N_ANC:
        dt_so_far = time.perf_counter() - t0
        rate = (i + 1) / dt_so_far
        remaining = (N_ANC - i - 1) / rate if rate > 0 else 0
        print(f"  {i+1}/{N_ANC}  ({dt_so_far:.0f}s elapsed, "
              f"~{remaining:.0f}s remaining, {rate:.2f} anchor/s)")
        sys.stdout.flush()
        # Intermediate save
        np.savez(OUT_FILE,
                 anchors_phi  = anchors_phi_sel[:i+1].astype(np.float32),
                 anchors_psi  = anchors_psi_sel[:i+1].astype(np.float32),
                 lags_ps      = np.array(LAGS_PS, dtype=np.float32),
                 frame_idx    = np.array(FRAME_IDX, dtype=np.int32),
                 bursts_phi   = bursts_phi[:i+1],
                 bursts_psi   = bursts_psi[:i+1],
                 n_bursts     = np.array([N_BURSTS]),
                 n_anchors    = np.array([i+1]),
                 temp         = np.array([TEMP]))

dt_total = time.perf_counter() - t0
print(f"\nDone. Total time: {dt_total:.1f}s ({dt_total/60:.1f} min)")
print(f"Saved: {OUT_FILE}")
