# -*- coding: utf-8 -*-
"""
Reproduce/inspect the EXISTING discrete transfer-operator eigenspectrum for
vacuum ADP (benchmark_v2/panel0/adp_eigvecs.npz) — the small-data regime — to
see where the C7eq<->alphaR (psi) process currently sits before deciding how
much more data / lower T is needed to separate it.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ev = np.load(os.path.join(HERE, "..", "benchmark_v2", "panel0", "adp_eigvecs.npz"))
print("keys:", ev.files)
evals = ev["eigenvalues"]          # (16,) — excludes stationary lambda=1
eigvecs = ev["eigvecs"]            # (1600,16) on 40x40 grid
occ = ev["occupied"].astype(bool)  # (1600,)
edges = ev["edges"]                # (41,)
TAU = 5.0
NB = 40

its = np.array([-TAU/np.log(abs(l)) if 0 < abs(l) < 1 else np.inf for l in evals])
print("\nEV  eigenvalue   ITS(ps)")
for i in range(min(10, len(evals))):
    print(f"  {i+1:2d}  {evals[i]:.4f}   {its[i]:8.1f}")
gaps = evals[:-1]/np.clip(evals[1:],1e-9,None)
print("\ngap ratios (consecutive):", "  ".join(f"{g:.1f}" for g in gaps[:6]))
print(f"occupied cells: {occ.sum()}/1600")

# figure: eigenvalues + top eigenvectors on Ramachandran
fig = plt.figure(figsize=(18, 8))
axw = fig.add_subplot(2, 4, 1)
axw.plot(range(1, len(evals)+1), evals, "o-")
axw.axhline(0, color="k", lw=.5); axw.set_title("transfer-op eigenvalues (excl. stationary)")
axw.set_xlabel("index"); axw.set_ylabel("Re λ")
def grid(v):
    g = np.full(1600, np.nan); g[occ] = v[occ]; return g.reshape(NB, NB).T
ext = [-180, 180, -180, 180]
for k in range(7):
    ax = fig.add_subplot(2, 4, k+2)
    vmax = np.nanmax(np.abs(grid(eigvecs[:, k])))
    im = ax.imshow(grid(eigvecs[:, k]), origin="lower", extent=ext, aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"EV{k+1}  λ={evals[k]:.3f}  ITS={its[k]:.0f}ps")
    ax.set_xlabel("φ"); ax.set_ylabel("ψ"); plt.colorbar(im, ax=ax, fraction=.046)
fig.suptitle("Existing vacuum-ADP discrete transfer operator (small-data regime, τ=5ps, 450K)")
plt.tight_layout()
out = os.path.join(HERE, "figures"); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "existing_vacuum_eigenspectrum.png"), dpi=110, bbox_inches="tight")
print("\nsaved figures/existing_vacuum_eigenspectrum.png")
