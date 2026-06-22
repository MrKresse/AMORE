# -*- coding: utf-8 -*-
"""
benchmark_v4 scoring across THREE lag conditions (5 ps / 0.1 ps / multitau) at
300 K. Reference = the 0.1 ps transfer operator's RIGHT eigenvectors (the slow
CVs): EV2 = φ-flip (C7eq/αR↔C7ax), EV3 = ψ-process (C7eq↔αR). The 0.1 ps operator
resolves ψ cleanly (λ=0.985) — at 5 ps ψ is nearly relaxed (λ=0.27) and its
right-eigenvector is noisy — so 0.1 ps gives the cleanest lag-independent
reference shapes for all three conditions. Rates untrusted → Hungarian scoring.
Outputs BENCHMARK_V4_RESULTS.md + figures.
"""
import os, sys, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); RUNS = os.path.join(HERE, "runs_v4")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
ANCH = "T300_m50"; REFLAG = "T300_0p1"; NB = 40; NCELL = NB*NB; edges = np.linspace(-np.pi, np.pi, NB+1)
VARIANTS = ["isa", "gramschmidt", "svd_power", "gs_isa", "ssm_isa"]
LABEL = {"isa": "ISA (no warm-up)", "gramschmidt": "GramSchmidt", "svd_power": "SVD-Power",
         "gs_isa": "GramSchmidt→ISA", "ssm_isa": "ShiftScale→ISA"}
LAGS_ALL = [("5 ps", ""), ("0.1 ps", "_0p1"), ("multitau (5+0.1 ps)", "_mt")]
def _lag_complete(sfx):  # all 5 variants have 3 completed seeds (hide partial lags)
    return all(sum(os.path.exists(os.path.join(RUNS, v + sfx, f"seed_{s}", "chi_best.npy"))
                   for s in range(3)) >= 3 for v in VARIANTS)
LAGS = [(n, s) for (n, s) in LAGS_ALL if _lag_complete(s)]

phi0 = np.load(os.path.join(DATA, f"vac_phi0_{ANCH}.npy"))
psi0 = np.load(os.path.join(DATA, f"vac_psi0_{ANCH}.npy"))
def _cells(ph, ps):
    return (np.clip(np.digitize(ph, edges)-1, 0, NB-1)*NB + np.clip(np.digitize(ps, edges)-1, 0, NB-1))
ci = _cells(phi0, psi0)

def right_eigvecs(tag):
    ph0 = np.load(os.path.join(DATA, f"vac_phi0_{tag}.npy")); ps0 = np.load(os.path.join(DATA, f"vac_psi0_{tag}.npy"))
    pht = np.load(os.path.join(DATA, f"vac_phitau_{tag}.npy")); pst = np.load(os.path.join(DATA, f"vac_psitau_{tag}.npy"))
    a, b = _cells(ph0, ps0), _cells(pht, pst)
    C = np.zeros((NCELL, NCELL)); np.add.at(C, (a, b), 1.0)
    rs = C.sum(1, keepdims=True); occ = rs[:, 0] > 0
    T = np.zeros_like(C); T[occ] = C[occ]/rs[occ]; T = 0.999*T + 1e-3/NCELL
    v, R = eig(T); o = np.argsort(v.real)[::-1]
    return v[o].real, R[:, o].real
evals, evg = right_eigvecs(REFLAG)
ref_phi, ref_psi = evg[ci, 1], evg[ci, 2]; REFS = np.stack([ref_phi, ref_psi], 1)
# sanity: 0.1ps phi vs 5ps phi
_e5, _g5 = right_eigvecs(ANCH)
def _pr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a)&np.isfinite(b)
    a, b = a[m]-a[m].mean(), b[m]-b[m].mean(); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(a@b/(na*nb))
print("sanity |r| 0.1ps-φ vs 5ps-φ:", round(abs(_pr(evg[ci,1], _g5[ci,1])), 3))

def hungarian(chi):
    R = np.array([[abs(_pr(chi[:, j], REFS[:, s])) for s in range(2)] for j in range(chi.shape[1])])
    ri, cidx = linear_sum_assignment(-R); return R[ri, cidx].mean()
def maxr(chi, ref): return max(abs(_pr(chi[:, j], ref)) for j in range(chi.shape[1]))

def score_run(variant, suffix):
    rs, a2, a3, keff = [], [], [], []
    for sd in sorted(glob.glob(os.path.join(RUNS, variant + suffix, "seed_*"))):
        cp = os.path.join(sd, "chi_best.npy")
        if not os.path.exists(cp): continue
        chi = np.load(cp); rs.append(hungarian(chi)); a2.append(maxr(chi, ref_phi)); a3.append(maxr(chi, ref_psi))
        sh = os.path.join(sd, "chi_sd_history.npy")
        if os.path.exists(sh):
            s = np.load(sh)
            if s.ndim == 2 and len(s): keff.append(int((s[-1] > 0.05).sum()))
    if not rs: return None
    return (len(rs), np.mean(rs), np.std(rs) if len(rs) > 1 else np.nan,
            np.mean(a2), np.mean(a3), np.mean(keff) if keff else np.nan)

if __name__ == "__main__":
    its = np.array([-0.1/np.log(abs(l)) if 0 < abs(l) < 1 else np.inf for l in evals])
    L = ["# Benchmark v4 — multi-D ISOKANN on vacuum alanine dipeptide (300 K)", "",
         "Genuine multi-process vacuum ADP, three lag conditions. Reference = the "
         "**0.1 ps** transfer-operator slow eigenvectors (lag-independent CV shapes; "
         "0.1 ps resolves both cleanly):", "",
         f"- **φ-flip** C7eq/αR↔C7ax (EV2).",
         f"- **ψ-process** C7eq↔αR (EV3) — λ=0.985 at 0.1 ps (ITS≈{its[2]:.1f} ps), but "
         "only λ=0.27 at 5 ps (nearly relaxed). The fast process is far easier to see "
         "at short lag in the *operator* — the test is whether ISOKANN can learn it.", "",
         "k=3, 231 pairwise distances, one burst/anchor per lag (multitau = K=2: a 5 ps "
         "AND a 0.1 ps burst per anchor). Score = Hungarian mean |r| of the 3 χ to "
         "{EV2,EV3}; also max|r| vs φ/ψ; k_eff = χ with SD>0.05.", "",
         "| Variant | lag | seeds | r (Hung φ,ψ) | SD | max r vs φ | max r vs ψ | k_eff |",
         "|---------|-----|-------|--------------|----|-----------|-----------|-------|"]
    for v in VARIANTS:
        for lagname, sfx in LAGS:
            r = score_run(v, sfx)
            if r is None: continue
            n, rm, sr, p2, p3, ke = r
            srs = f"±{sr:.3f}" if not np.isnan(sr) else ""
            L.append(f"| {LABEL[v]} | {lagname} | {n} | {rm:.3f} | {srs} | {p2:.3f} | {p3:.3f} | {ke:.1f} |")
    L += ["", "## Reference (0.1 ps transfer-operator eigenvectors)", "",
          "![reference](figures/v4_reference.png)", ""]
    for lagname, sfx in LAGS:
        if not glob.glob(os.path.join(RUNS, "gramschmidt" + sfx, "seed_*")): continue
        L += [f"## χ maps + validation — {lagname}", ""]
        for v in VARIANTS:
            if os.path.exists(os.path.join(RUNS, v + sfx, "seed_0", "chi_best.npy")):  # gate on run, not fig
                L += [f"### {LABEL[v]} — {lagname}", "", f"![{v}{sfx}](figures/v4_chi_{v}{sfx}.png)", ""]
        L += [f"### validation loss — {lagname}", "", f"![val{sfx}](figures/v4_val{sfx}.png)", ""]
    with open(os.path.join(HERE, "BENCHMARK_V4_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # ── figures ──────────────────────────────────────────────────────────────
    ss = np.random.default_rng(0).choice(len(phi0), 30000, replace=False)
    P, Q = phi0[ss]*180/np.pi, psi0[ss]*180/np.pi
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, (arr, ttl) in enumerate([(evg[ci[ss], 0], "EV1 stationary"),
                                    (evg[ci[ss], 1], f"EV2 φ-flip (λ={evals[1]:.3f})"),
                                    (evg[ci[ss], 2], f"EV3 ψ (λ={evals[2]:.3f}, ITS≈{its[2]:.0f}ps)")]):
        s = ax[k].scatter(P, Q, c=arr, s=3, cmap="coolwarm"); ax[k].set_title(ttl)
        plt.colorbar(s, ax=ax[k], fraction=.046); ax[k].set_xlabel("φ"); ax[k].set_ylabel("ψ")
    fig.suptitle("benchmark_v4 reference — vacuum ADP 300 K, 0.1 ps transfer-operator (right eigenvectors)")
    plt.tight_layout(); fig.savefig(os.path.join(FIG, "v4_reference.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    for lagname, sfx in LAGS:
        for v in VARIANTS:
            cp = os.path.join(RUNS, v + sfx, "seed_0", "chi_best.npy")
            if not os.path.exists(cp): continue
            chi = np.load(cp)
            fig, ax = plt.subplots(1, 3, figsize=(13, 4))
            for j in range(3):
                s = ax[j].scatter(P, Q, c=chi[ss, j], s=3, cmap="coolwarm")
                ax[j].set_title(f"χ{j+1} (SD={chi[:, j].std():.2f})"); plt.colorbar(s, ax=ax[j], fraction=.046)
                ax[j].set_xlabel("φ"); ax[j].set_ylabel("ψ")
            fig.suptitle(f"{LABEL[v]} — {lagname}"); plt.tight_layout()
            fig.savefig(os.path.join(FIG, f"v4_chi_{v}{sfx}.png"), dpi=105, bbox_inches="tight"); plt.close(fig)
        # val curves (two panels: isotarget vs svd_power)
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 4.5))
        any_iso = False
        for v in ["isa", "gramschmidt", "gs_isa", "ssm_isa"]:
            vp = os.path.join(RUNS, v + sfx, "seed_0", "val_loss.npy")
            if os.path.exists(vp): axA.plot(np.load(vp), label=LABEL[v], alpha=.85); any_iso = True
        axA.axvline(500, color="gray", ls=":", lw=1, label="MIN_ITER=500")
        axA.set_yscale("log"); axA.set_xlabel("isotarget iteration"); axA.set_ylabel("val MSE")
        axA.set_title(f"isotarget variants — {lagname}");
        if any_iso: axA.legend(fontsize=8)
        vp = os.path.join(RUNS, "svd_power" + sfx, "seed_0", "val_loss.npy")
        if os.path.exists(vp): axB.plot(np.load(vp), "o-", ms=3, color="tab:purple")
        axB.set_xlabel("outer power iter (×50 epochs)"); axB.set_ylabel("MSE to [0,1] targets")
        axB.set_title(f"SVD-Power (fixed 100 outer) — {lagname}")
        plt.tight_layout(); fig.savefig(os.path.join(FIG, f"v4_val{sfx}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote BENCHMARK_V4_RESULTS.md + figures")
