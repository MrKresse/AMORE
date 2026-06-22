# -*- coding: utf-8 -*-
"""
benchmark_v4 — ground-truth MSM for explicit-solvent alanine dipeptide
(mdshare 3x250ns, matching the VAMPnets setup).

Periodicity-aware discretization: k-means on (cosφ,sinφ,cosψ,sinψ) so the C7eq
basin at ψ≈±180° is not split by a grid boundary. Reversible MSM at lag
tau=50 ps; implied timescales + spectral gap; 6-state PCCA+. Saves gate
diagnostics (figures + reference npz) so we can judge MSM quality before ISOKANN.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

LAG = 50
NCLUST = 150
NSTATES = 6
DT_PS = 1.0

dih = np.load(os.path.join(DATA, "alanine-dipeptide-3x250ns-backbone-dihedrals.npz"))
trajs = [dih[k] for k in sorted(dih.files)]          # each (250000,2)=(phi,psi) rad
print("dihedral arrays:", [(k, dih[k].shape) for k in sorted(dih.files)])

feat = lambda a: np.stack([np.cos(a[:, 0]), np.sin(a[:, 0]),
                           np.cos(a[:, 1]), np.sin(a[:, 1])], axis=1).astype(np.float32)
X = [feat(t) for t in trajs]

km = KMeans(n_clusters=NCLUST, max_iter=80, fixed_seed=0, n_jobs=4).fit_fetch(np.concatenate(X))
dtrajs = [km.transform(x).astype(np.int64) for x in X]
centers_ang = np.column_stack([np.arctan2(km.cluster_centers[:, 1], km.cluster_centers[:, 0]),
                               np.arctan2(km.cluster_centers[:, 3], km.cluster_centers[:, 2])])  # (NCLUST,2)

counts = TransitionCountEstimator(lagtime=LAG, count_mode="sliding").fit_fetch(dtrajs)
active = counts.submodel_largest()
msm = MaximumLikelihoodMSM(reversible=True).fit_fetch(active)
nact = msm.n_states
sym = msm.count_model.state_symbols
print(f"\nMSM: {nact}/{NCLUST} active microstates, lag={LAG} ps")

its = msm.timescales(k=min(10, nact - 1)) * DT_PS
print("\nImplied timescales (ps):")
for i, t in enumerate(its):
    print(f"  ITS{i+2}: {t:9.1f} ps")
gaps = its[:-1] / its[1:]
print("\ngap ratios:", "  ".join(f"{g:.1f}" for g in gaps[:6]))

# ITS convergence vs lag
lags = [5, 10, 20, 35, 50, 75, 100, 150]
its_conv = []
for L in lags:
    try:
        c = TransitionCountEstimator(lagtime=L, count_mode="sliding").fit_fetch(dtrajs)
        m = MaximumLikelihoodMSM(reversible=True).fit_fetch(c.submodel_largest())
        its_conv.append(m.timescales(k=5) * DT_PS)
    except Exception:
        its_conv.append(np.full(5, np.nan))
its_conv = np.array(its_conv)

pcca = msm.pcca(NSTATES)
memb = pcca.memberships                       # (nact,6)
assign = memb.argmax(1)
pi = msm.stationary_distribution
print("\nPCCA+ 6-state stationary populations:")
for s in range(NSTATES):
    print(f"  state {s}: pi={pi[assign==s].sum():.3f}  (#micro={int((assign==s).sum())})")
evec = msm.eigenvectors_right(min(NSTATES, nact))

# per-cluster -> value maps (active clusters only); inactive -> nan
def clustervals(vals_active, fill=np.nan):
    g = np.full(NCLUST, fill); g[sym] = vals_active; return g
assign_c = clustervals(assign.astype(float), fill=-1)
pi_c = clustervals(pi, fill=0.0)
evec_c = np.column_stack([clustervals(evec[:, k]) for k in range(evec.shape[1])])
memb_c = np.column_stack([clustervals(memb[:, s], fill=0.0) for s in range(NSTATES)])

# scatter visualization on (phi,psi): subsample frames, color by cluster value
phi = np.concatenate([t[:, 0] for t in trajs]) * 180 / np.pi
psi = np.concatenate([t[:, 1] for t in trajs]) * 180 / np.pi
ci = np.concatenate(dtrajs)
ss = np.random.default_rng(0).choice(len(phi), size=40000, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(18, 9)); axes = axes.ravel()
sc = axes[0].scatter(phi[ss], psi[ss], c=np.log10(pi_c[ci[ss]] + 1e-6), s=3, cmap="viridis")
axes[0].set_title("log10 stationary π"); plt.colorbar(sc, ax=axes[0], fraction=.046)
sc = axes[1].scatter(phi[ss], psi[ss], c=assign_c[ci[ss]], s=3, cmap="tab10", vmin=0, vmax=9)
axes[1].set_title("PCCA+ 6 states"); plt.colorbar(sc, ax=axes[1], fraction=.046)
for j in range(2, 8):
    k = j - 2
    if k < evec.shape[1]:
        sc = axes[j].scatter(phi[ss], psi[ss], c=evec_c[ci[ss], k], s=3, cmap="coolwarm")
        ttl = "stationary (EV1)" if k == 0 else f"slow EV{k+1} (ITS={its[k-1]:.0f} ps)"
        axes[j].set_title(ttl); plt.colorbar(sc, ax=axes[j], fraction=.046)
for a in axes:
    a.set_xlabel("φ"); a.set_ylabel("ψ"); a.set_xlim(-180, 180); a.set_ylim(-180, 180)
fig.suptitle(f"benchmark_v4 MSM — ADP explicit solvent, τ={LAG} ps, "
             f"{nact} microstates (k-means cos/sin)", fontsize=13)
plt.tight_layout(); fig.savefig(os.path.join(FIG, "msm_ground_truth.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(range(2, 2 + len(its)), its, color="steelblue")
ax[0].axhline(LAG, color="r", ls="--", lw=1, label=f"lag={LAG} ps")
ax[0].set_yscale("log"); ax[0].set_xlabel("mode index"); ax[0].set_ylabel("ITS (ps)")
ax[0].set_title("implied timescales @ τ=50 ps"); ax[0].legend()
for i in range(its_conv.shape[1]):
    ax[1].plot(lags, its_conv[:, i], "o-", label=f"ITS{i+2}")
ax[1].plot(lags, lags, "k--", lw=1, label="ITS=τ")
ax[1].set_yscale("log"); ax[1].set_xlabel("lag τ (ps)"); ax[1].set_ylabel("ITS (ps)")
ax[1].set_title("ITS convergence"); ax[1].legend(fontsize=7)
plt.tight_layout(); fig.savefig(os.path.join(FIG, "msm_timescales.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

np.savez(os.path.join(DATA, "msm_reference.npz"),
         lag=LAG, nclust=NCLUST, cluster_centers=km.cluster_centers,
         centers_ang=centers_ang, state_symbols=sym, timescales=its,
         eigvecs_cluster=evec_c, memb_cluster=memb_c, assign_cluster=assign_c,
         stationary_cluster=pi_c)
print("\nsaved figures/msm_ground_truth.png, figures/msm_timescales.png, data/msm_reference.npz")
