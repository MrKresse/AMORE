"""
Run power_method_multi (LARRY implementation) on alanine dipeptide data.

Architecture: ChiNetMultiRaw, hidden=[512,256,128], sigmoid output
k=3, 5 seeds, 80 power iterations × 400 epochs each.

Outputs
-------
  results/alanine_power_method.npz  — chi_seeds (5, N_ANC, 3)
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from amore.isokann import ChiNetMultiRaw, power_method_multi

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

K            = 3
N_POWER_ITER = 80
EPOCHS_PER_ITER = 400
LR           = 2e-3
LR_DECAY     = 0.97
N_SEEDS      = 5
DEVICE       = torch.device("cpu")

print("Loading alanine data …")
data = np.load(os.path.join(DATA_DIR, "alanine_koopman.npz"))
anchors_feat = data["anchors_feat"].astype(np.float32)  # (N_ANC, 231)
bursts_feat  = data["bursts_feat"].astype(np.float32)   # (N_ANC, 20, 231)
phi_anc      = data["anchors_phi"]
psi_anc      = data["anchors_psi"]
N_ANC, N_FEAT = anchors_feat.shape
N_K = bursts_feat.shape[1]

print(f"  {N_ANC} anchors, {N_FEAT} features, {N_K} bursts")

# Flatten bursts → Koopman pairs
x0_all = np.repeat(anchors_feat, N_K, axis=0)     # (N_ANC*N_K, 231)
x1_all = bursts_feat.reshape(-1, N_FEAT)

x0_t    = torch.tensor(x0_all,        dtype=torch.float32, device=DEVICE)
x1_t    = torch.tensor(x1_all,        dtype=torch.float32, device=DEVICE)
x_all_t = torch.tensor(anchors_feat,  dtype=torch.float32, device=DEVICE)

chi_seeds = []

print(f"\nRunning power_method_multi (k={K}, {N_POWER_ITER} iters × {EPOCHS_PER_ITER} epochs) …")
for seed in range(N_SEEDS):
    torch.manual_seed(seed * 137)
    np.random.seed(seed * 137)

    net    = ChiNetMultiRaw(in_dim=N_FEAT, k=K, hidden=[512, 256, 128]).to(DEVICE)
    result = power_method_multi(
        net, x0_t, x1_t,
        n_iter=N_POWER_ITER,
        epochs_per_iter=EPOCHS_PER_ITER,
        lr=LR, lr_decay=LR_DECAY,
        verbose=False,
    )

    net.eval()
    with torch.no_grad():
        chi = net(x_all_t).cpu().numpy()   # (N_ANC, K)

    sd = chi.std(axis=0)
    ev = result.get("eigenvalues", np.array([]))
    print(f"  Seed {seed}: loss={result['losses'][-1]:.5f}  SD={sd.round(4)}  "
          f"eigenvalues={np.abs(ev[:3]).round(3) if len(ev) else 'N/A'}")
    chi_seeds.append(chi)

chi_seeds = np.array(chi_seeds)  # (5, N_ANC, 3)

np.savez(
    os.path.join(RESULTS_DIR, "alanine_power_method.npz"),
    chi_seeds  = chi_seeds,
    phi_anchors = phi_anc,
    psi_anchors = psi_anc,
)
print(f"\nSaved: results/alanine_power_method.npz  shape={chi_seeds.shape}")
