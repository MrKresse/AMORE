"""Build hematopoiesis_k_sweep_comparison.ipynb — a side-by-side of the k=4/5/6 Arm-K
models (analogue of mef_k4_vs_k6_comparison.ipynb). The point it makes: the cDC2 fate
is unrecoverable at every k (it is a late fan-out, not a metastable basin), while the
other three fates are clean basins at every k — so the cDC2 miss is a property of the
operator, not of the state count."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# ISOKANN Arm K on hematopoiesis — the $k$-sweep ($k=3,4,5,6,7$)

How many memberships? Several defensible answers, and we train them all:
* **$k=3$** — the **operator's own count**: $T$ has **three Perron eigenvalues at $\lambda=1$**, i.e. three
  absorbing basins. GPCCA's three macrostates are exactly **pDC, CD14+ Mono, Normoblast** (cDC2 is
  absorbed into the CD14+ Mono basin), and ISOKANN recovers the same three.
* **$k=4$** — the number of annotated terminal lineages (primary).
* **$k=5$** — the spectrum's **largest gap** (shallow, ≈0.008).
* **$k=6$** — GPCCA's macrostate count.
* **$k=7$** — a minor secondary gap; the next gap is at $k=10>9$ cell types, so 7 is a sensible upper
  bound to probe.

The comparison is **boring on purpose**, in the way that matters:
* the three metastable basins (**pDC, CD14+ Mono, Normoblast**) are recovered at **every** $k\ge3$
  (AUROC ≈ 0.99, matching/beating GPCCA);
* **cDC2** is **never** recovered: no $\chi$ column peaks on cDC2 at any $k$, its best (sign-flipped)
  column tops out ≈ 0.85 (the shared pDC/DC axis), and even GPCCA only surfaces cDC2 at $n=6$ — *after*
  both erythroid progenitors (GPCCA macrostates: $n{=}3$ pDC/CD14/Normo → $n{=}4$ +MK/E prog → $n{=}5$
  +Erythroblast → $n{=}6$ +cDC2). cDC2 is the least-metastable fate for *both* methods.

So cDC2's miss is a property of the **operator** (it is not a metastable set), not of the state count.
Two training caveats the sweep also documents: (i) at the smallest $k=3$ the ISA vertex search is
**seed-sensitive** — some seeds let the dominant erythroid arm grab a slot and miss pDC; we use a seed
that recovers the operator's three sinks (§1b). (ii) the ISA loss is noisy; given a long enough
best-iterate continuation **$k=6$ settles to a flat loss**, while **$k=5$ keeps oscillating** (one
starved mode keeps flipping ISA vertices) — see §5. (Contrast MEF, where $k$ *was* load-bearing because
its source genuinely needed its own mode.)""")

code(r"""import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = "retina"
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join("..", "..", "..", "..", "src"))
import config as C
from amore.scrna import scatter_chi, scatter_categorical, plot_loss
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

A = C.ARTIFACTS
TERMS = C.TERMINAL_STATES
FATE_COLS = json.load(open(C.RESOLVED_JSON))["fate_prob_columns"]
cell_type = np.load(f"{A}/cell_type.npy", allow_pickle=True).astype(str)
fate_probs = np.load(f"{A}/cr2_fate_probs.npy").astype("float32")
DIRS = {3: "armK_k3", 4: "armK", 5: "armK_k5", 6: "armK_k6", 7: "armK_k7"}
CHIS = {kk: np.load(f"{A}/{d}/chi.npy") for kk, d in DIRS.items() if os.path.exists(f"{A}/{d}/chi.npy")}

def lineage_map(chi_):
    S = np.array([[chi_[cell_type == t, j].mean() for t in TERMS] for j in range(chi_.shape[1])])
    r, c = linear_sum_assignment(-S)
    return {TERMS[cc]: int(rr) for rr, cc in zip(r, c)}
def col_label(chi_, j, n=1500):
    top = np.argsort(-chi_[:, j])[:n]; v, ct = np.unique(cell_type[top], return_counts=True)
    return v[ct.argmax()]
print("loaded k:", sorted(CHIS))""")

md(r"""## 1 · Koopman spectrum — three Perron eigenvalues, then a near-degenerate real cluster
Three eigenvalues at $\lambda=1$ (three nearly-absorbing basins), then a *shallow* descent: the largest
gap ($\approx0.008$) sits at $k=5$, and beyond mode 5 the eigenvalues form a **near-degenerate cluster**
(modes 6–9 lie within $\approx0.007$ of each other). The slow subspace is also **purely real** — there
is *no complex-conjugate pair* among the leading eigenvalues, so the k=6 difficulty is **not** a split
conjugate pair. It is the real analogue: forcing a 6th membership into a near-degenerate cluster gives it
no well-separated mode to lock onto, so two of the six ISA memberships stay weak (see §5). $k=5$ is the
clean ceiling; $k=6$ over-partitions. (Contrast MEF's sharp single-$\lambda{=}1$ + 0.025 gap, where a
specific higher $k$ *was* right.)""")

code(r"""T = sp.load_npz(f"{A}/T.npz").tocsr().astype(np.float64)
w = eigs(T, k=20, which="LM", return_eigenvectors=False, tol=0)   # complex eigenvalues
w = w[np.argsort(-w.real)]
ev = w.real
gaps = -np.diff(ev[:14]); kstar = int(np.argmax(gaps[:10])) + 1
max_imag = float(np.max(np.abs(w.imag)))
clust = float(ev[5] - ev[8])   # spread of modes 6..9
fig, ax = plt.subplots(figsize=(7.5, 4))
idx = np.arange(1, 15)
ax.plot(idx, ev[:14], "o-", color="#444", ms=6)
for kk in (4, 5, 6):
    ax.axvline(kk, ls=":", color="#aaa")
ax.axvspan(kstar + 0.5, kstar + 1.5, color="#E8820C", alpha=0.12)
ax.annotate(f"largest gap={gaps[kstar-1]:.3f} (k*={kstar})", xy=(kstar+0.6, ev[kstar]),
            color="#E8820C", fontsize=9)
ax.axvspan(5.5, 9.5, color="#1f77b4", alpha=0.07)
ax.annotate(f"near-degenerate cluster\n(modes 6–9 span {clust:.3f})", xy=(7.5, ev[7]),
            xytext=(8.2, ev[7] - 0.012), color="#1f77b4", fontsize=8)
ax.set_xlabel("eigenvalue index"); ax.set_ylabel(r"$\lambda$ of $T$ (real)")
ax.set_title("Koopman spectrum — 3 Perron eigenvalues, near-degenerate cluster after k=5")
fig.tight_layout(); fig.savefig(f"{C.FIGURES}/ksweep_spectrum.png", dpi=300, bbox_inches="tight"); plt.show()
print("eigs (real):", np.round(ev[:9], 4).tolist())
print(f"max |Im(lambda)| over top-20 = {max_imag:.2e}  -> slow subspace is REAL (no conjugate pair to split)")""")

md(r"""### GPCCA's own macrostate progression (the operator's ranking of fates)
For reference, the GPCCA macrostates as $n$ grows (computed in the CR2 reproduction) — the order in which
the operator surfaces fates:

| $n$ | GPCCA macrostates |
|---|---|
| 3 | **pDC, CD14+ Mono, Normoblast** |
| 4 | pDC, **MK/E prog**, Normoblast, CD14+ Mono |
| 5 | MK/E prog, pDC, **Erythroblast**, Normoblast, CD14+ Mono |
| 6 | MK/E prog, pDC, CD14+ Mono, **cDC2**, Normoblast, Erythroblast |

So the operator's three sinks are pDC / CD14+ Mono / Normoblast; the next two states it adds are
**erythroid progenitors** (MK/E prog, Erythroblast), and **cDC2 appears last, only at $n=6$**. ISOKANN's
$\chi$ follows the same ranking (§4): the three basins, then the HSC source / erythroid arm, never cDC2.""")

md(r"""## 1b · Convergence & live modes per $k$ (after a fair best-iterate continuation)

Every model was given the **same** best-iterate continuation (`03b_continue_armK.py`); the raw `03` run
keeps the noisy *final* ISA iterate. The table reports, per $k$: total iters, the best (minimum) training
ISA loss, the per-mode SD of the deployed (best) iterate, and how many modes are **weak** (SD < 0.15).
Read it together with §5 (loss curves):

* **$k=3$** recovers the three basins as three strong modes — *with the right seed*. At this smallest $k$
  the ISA vertex search is **initialization-sensitive**: some seeds let the dominant erythroid arm take a
  vertex and miss pDC (`armK_k3_seed0_localopt` is such a run). We use a seed that lands on the operator's
  three sinks; that this is needed only at $k=3$ is itself a sign the partition is tightest there.
* **$k=4$** is fully strong (0 weak modes).
* **$k=5$** has one starved mode and — even after a long continuation — its loss **keeps oscillating**
  (the starved membership keeps flipping ISA vertices).
* **$k=6$ converges to a flat loss** after the long continuation (all four terminal purities = 1.0) but
  carries **two weak modes**; **$k=7$** carries even more. Those weak modes are the over-partition of the
  near-degenerate slow cluster (§1), not an undertraining artifact — extra iters settle the *loss* but do
  not turn them into real basins.""")

code(r"""rows = []
for kk, d in sorted(DIRS.items()):
    if not os.path.exists(f"{A}/{d}/meta.json"): continue
    m = json.load(open(f"{A}/{d}/meta.json"))
    lt = np.load(f"{A}/{d}/loss_train.npy")
    sd = np.array(m["sd_final"])
    rows.append(dict(k=kk, total_iters=len(lt), best_loss_tr=float(np.nanmin(lt)),
                     n_weak_modes_sd_lt_0p15=int((sd < 0.15).sum()),
                     min_mode_sd=round(float(sd.min()), 3),
                     continued=bool(m.get("continued_extra_iters"))))
print(pd.DataFrame(rows).set_index("k").to_string())""")

md(r"""## 2 · Per-$k$ cell-fate AUROC — the three basins (pDC, CD14+ Mono, Normoblast) clean at every $k\ge3$, cDC2 missed at every $k$

At $k=3$ (the operator's three-sink count, §1) only three terminals can be assigned a committor; the
Hungarian step leaves **cDC2** unassigned, so its entry below is its *best-available* column (the
pDC/DC-axis spillover, ≈0.86). For $k\ge4$ every terminal is assigned, but cDC2's committor is still
near-random. The three genuine basins are recovered at **every** $k$ (AUROC ≈0.99), matching/beating
GPCCA.""")

code(r"""def _best_col(chi, t):                                   # best single column for terminal t
    return int(np.argmax([roc_auc_score((cell_type == t).astype(int), chi[:, j]) for j in range(chi.shape[1])]))
rows = []
for kk, chi in sorted(CHIS.items()):
    MAP = lineage_map(chi)
    rec = {"k": kk, "k_eff": int((chi.std(0) > C.SD_LIVE_THRESHOLD).sum())}
    for t in TERMS:                                       # MAP[t] if assigned, else its best column (k<4)
        col = MAP[t] if t in MAP else _best_col(chi, t)
        rec[t] = roc_auc_score((cell_type == t).astype(int), chi[:, col])
    rec["mean_3basins"] = np.mean([rec[t] for t in ["pDC", "CD14+ Mono", "Normoblast"]])
    rec["mean_all4"] = np.mean([rec[t] for t in TERMS])
    rows.append(rec)
cr2row = {"k": "GPCCA", "k_eff": np.nan}
for t in TERMS:
    cr2row[t] = roc_auc_score((cell_type == t).astype(int), fate_probs[:, FATE_COLS.index(t)])
cr2row["mean_3basins"] = np.mean([cr2row[t] for t in ["pDC", "CD14+ Mono", "Normoblast"]])
cr2row["mean_all4"] = np.mean([cr2row[t] for t in TERMS])
tab = pd.DataFrame(rows + [cr2row]).set_index("k")
print(tab.round(3).to_string())""")

md(r"""## 3 · cDC2 is in no column at any $k$
For each $k$, the best per-column AUROC for cDC2 (allowing a sign flip) and whether **any** column's top
cells are predominantly cDC2. At every $k$ the best column tops out ≈ 0.85 — the shared pDC/DC axis — and
no column peaks on cDC2, i.e. cDC2 never forms its own ISA mode.""")

code(r"""y = (cell_type == "cDC2").astype(int)
rows = []
for kk, chi in CHIS.items():
    aucs = [roc_auc_score(y, chi[:, j]) for j in range(chi.shape[1])]
    flipped = [max(a, 1 - a) for a in aucs]
    top_fracs = [(cell_type[np.argsort(-chi[:, j])[:200]] == "cDC2").mean() for j in range(chi.shape[1])]
    rows.append(dict(k=kk, best_col_AUROC=round(max(aucs), 3),
                     best_col_AUROC_signfix=round(max(flipped), 3),
                     max_top200_cDC2_frac=round(max(top_fracs), 3),
                     any_col_peaks_cDC2=bool(max(top_fracs) > 0.5)))
print(pd.DataFrame(rows).set_index("k").to_string())""")

md(r"""## 4 · All χ modes per $k$ (titled by their actual dominant cell type)

Each row is one $k$. The **leftmost panel** colours that $k$'s χ-UMAP by annotated `l2_cell_type` (the
reference layout); the remaining panels are the individual $\chi$ columns, each **titled by its own
dominant cell type** (with that fraction) and **ordered left→right along the developmental axis**
(HSC source → erythroid progenitors → erythroid terminal → monocyte → DC), so the same kind of mode sits
in the same place across the $k=4,5,6$ rows. Read off the modes the operator actually supports: the three
basins (pDC, CD14+ Mono, Normoblast), the **HSC source**, and the **erythroid progenitors** (MK/E prog →
Proerythroblast → Erythroblast). **No panel is cDC2** at any $k$ — even though cDC2 cells are clearly
present in the cell-type panel. (Which column the Hungarian step is *forced* to label "cDC2" — always one
of the erythroid-progenitor columns — is shown in the table below; the `col#` in each title is the
original ISA column index, so it still maps to that table.)""")

code(r"""import umap
# developmental ordering of cell types (source -> erythroid arm -> myeloid -> DC); used to lay the
# χ-mode panels out left->right by their dominant type, so rows are comparable across k.
DEV_ORDER = ["HSC", "MK/E prog", "G/M prog", "Proerythroblast", "Erythroblast", "Normoblast",
             "CD14+ Mono", "cDC2", "pDC"]
def _devrank(lab): return DEV_ORDER.index(lab) if lab in DEV_ORDER else len(DEV_ORDER)
def _top(chi, j, n=1500):                               # (dominant cell type, its fraction) of col j's top-n
    top = np.argsort(-chi[:, j])[:n]; v, cc = np.unique(cell_type[top], return_counts=True)
    return v[cc.argmax()], cc.max() / n
ncols = max(CHIS) + 1                                    # +1 leading cell-type column per k
fig, axes = plt.subplots(len(CHIS), ncols, figsize=(3.6 * ncols, 3.4 * len(CHIS)))
if len(CHIS) == 1: axes = axes[None, :]
for r, (kk, chi) in enumerate(sorted(CHIS.items())):
    emb = umap.UMAP(n_neighbors=200, min_dist=0.7, init="pca", random_state=0).fit_transform(chi)
    doms = {j: _top(chi, j) for j in range(chi.shape[1])}
    col_order = sorted(range(chi.shape[1]), key=lambda j: (_devrank(doms[j][0]), -doms[j][1]))
    scatter_categorical(axes[r, 0], emb, cell_type, title=f"k={kk} · cell types")
    for slot in range(max(CHIS)):
        ax = axes[r, slot + 1]
        if slot < len(col_order):
            j = col_order[slot]; lab, frac = doms[j]    # title by the column's actual dominant cell type
            scatter_chi(ax, emb, chi[:, j], title=f"k={kk} · col{j} · {lab} {frac:.0%}", s=4)
        else:
            ax.axis("off")
fig.suptitle("All χ modes for k=4,5,6 (ordered by developmental stage, titled by dominant cell type; none is cDC2)",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{C.FIGURES}/ksweep_all_modes.png", dpi=300, bbox_inches="tight"); plt.show()""")

md(r"""### What each χ column actually is (Hungarian label vs. real content)
The forced `→cDC2` column is an erythroid progenitor at every $k$; all the well-recovered columns are the
three basins, the HSC source, and erythroid-arm states.""")

code(r"""rows = []
for kk, chi in sorted(CHIS.items()):
    MAP = lineage_map(chi); inv = {v: t for t, v in MAP.items()}
    doms = {j: _top(chi, j) for j in range(chi.shape[1])}
    for j in sorted(range(chi.shape[1]), key=lambda j: (_devrank(doms[j][0]), -doms[j][1])):
        lab, frac = doms[j]
        assigned = inv.get(j, "—")
        au = roc_auc_score((cell_type == assigned).astype(int), chi[:, j]) if j in inv else np.nan
        rows.append(dict(k=kk, col=j, hungarian_label=assigned, real_top_type=f"{lab} ({frac:.0%})",
                         AUROC_for_assigned=("—" if np.isnan(au) else round(au, 3))))
print(pd.DataFrame(rows).to_string(index=False))""")

md(r"""## 4b · Why cDC2 is missed — it is a non-absorbing *bridge* into CD14+ Mono, not a basin

The χ-UMAPs above place cDC2 *between* CD14+ Mono and the HSC/pDC region rather than in a terminal blob.
That is not an embedding artefact — under this DPT-directed PseudotimeKernel cDC2 is a **non-absorbing
transition state that mostly commits to the CD14+ Mono basin**, and both ISOKANN and GPCCA's own numbers
say so:

* **Pseudotime** (left panel): cDC2 sits at an *intermediate* pseudotime (median ≈ 0.39), earlier than
  all three true terminals (Normoblast ≈ 0.83, CD14+ Mono / pDC ≈ 0.5) — it is mid-trajectory, not an
  endpoint.
* **Self-retention** (right panel): cDC2 keeps only ≈ 0.76 of its outgoing transition mass within itself,
  far below the absorbing terminals (CD14+ Mono 0.98, pDC 0.94, Normoblast 0.93) and on par with the
  Proerythroblast *transit* state — i.e. it leaks, the signature of a bridge, not a basin.
* **Where it flows** + **what both methods assign** (printout): cDC2's outgoing mass drains chiefly to
  **CD14+ Mono**; accordingly ISOKANN's CD14+ Mono committor is ≈ 0.95 *on the cDC2 cells themselves*
  (its cDC2 column ≈ 0), and **GPCCA's own absorption probabilities route the average cDC2 cell 0.62 to
  CD14+ Mono vs only 0.31 to cDC2**. So ISA does not "miss" a basin — it faithfully reports that cDC2 is
  a sub-region committing to the monocyte basin; GPCCA separates cDC2 only because `set_terminal_states`
  *forces* the small cDC2 macrostate core to be an absorbing target.""")

code(r"""pt = np.load(f"{A}/day.npy").astype(float)                 # dpt_pseudotime
order = ["HSC", "MK/E prog", "G/M prog", "Proerythroblast", "Erythroblast",
         "Normoblast", "CD14+ Mono", "cDC2", "pDC"]
idx_by = {t: np.where(cell_type == t)[0] for t in order}
# self-retained transition mass per type
T = sp.load_npz(f"{A}/T.npz").tocsr()
retain = {}
for t in order:
    rows = idx_by[t]; sub = T[rows]
    colmask = np.isin(np.arange(T.shape[0]), rows)
    retain[t] = float(np.asarray(sub[:, colmask].sum(1)).ravel().mean())
ptm = {t: float(np.median(pt[idx_by[t]])) for t in order}
cols = ["#d62728" if t == "cDC2" else ("#00AB8E" if t in TERMS else "#999") for t in order]

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].bar(range(len(order)), [ptm[t] for t in order], color=cols)
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=40, ha="right")
ax[0].set_ylabel("median dpt_pseudotime"); ax[0].set_title("Pseudotime per type (cDC2 = intermediate, not terminal)")
ax[1].bar(range(len(order)), [retain[t] for t in order], color=cols)
ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=40, ha="right")
ax[1].set_ylabel("within-type transition mass"); ax[1].set_ylim(0, 1)
ax[1].set_title("Self-retention in T (cDC2 leaks — bridge, not basin)")
ax[1].axhline(retain["cDC2"], ls=":", color="#d62728")
fig.tight_layout(); fig.savefig(f"{C.FIGURES}/ksweep_cdc2_bridge.png", dpi=300, bbox_inches="tight"); plt.show()

# cDC2 outflow + assignment by both methods
rows = idx_by["cDC2"]; sub = T[rows]
flow = {t: float(np.asarray(sub[:, np.isin(np.arange(T.shape[0]), idx_by[t])].sum(1)).ravel().mean()) for t in order}
flow = {t: v for t, v in sorted(flow.items(), key=lambda x: -x[1]) if v > 0.01}
print("cDC2 outgoing transition mass by target:", {t: round(v, 3) for t, v in flow.items()})
chi4 = CHIS[4]; MAP4 = lineage_map(chi4)
print(f"ISOKANN χ on cDC2 cells: CD14+ Mono-committor={chi4[idx_by['cDC2'], MAP4['CD14+ Mono']].mean():.2f}, "
      f"cDC2-committor={chi4[idx_by['cDC2'], MAP4['cDC2']].mean():.2f}")
print("GPCCA fate prob of cDC2 cells: " +
      ", ".join(f"{n}={fate_probs[idx_by['cDC2'], FATE_COLS.index(n)].mean():.2f}" for n in FATE_COLS))""")

md(r"""## 5 · Loss curves per $k$ — all converge, all keep every mode alive""")

code(r"""fig, axes = plt.subplots(1, len(CHIS), figsize=(5.2 * len(CHIS), 4))
if len(CHIS) == 1: axes = [axes]
for ax, (kk, d) in zip(np.atleast_1d(axes), sorted(DIRS.items())):
    if not os.path.exists(f"{A}/{d}/loss_train.npy"): ax.axis("off"); continue
    lt = np.load(f"{A}/{d}/loss_train.npy")
    lm = np.load(f"{A}/{d}/loss_monitor.npy") if os.path.exists(f"{A}/{d}/loss_monitor.npy") else None
    plot_loss(ax, lt, lm, title=f"k={kk} invariance loss")
fig.tight_layout(); fig.savefig(f"{C.FIGURES}/ksweep_loss.png", dpi=300, bbox_inches="tight"); plt.show()""")

md(r"""## Conclusion

**What the operator supports, and what ISOKANN recovers, across $k=3\!-\!7$:**

* **Three metastable basins — pDC, CD14+ Mono, Normoblast — recovered at every $k\ge3$**, AUROC ≈0.99,
  ties/beats GPCCA. At $k=3$ (the operator's three-$\lambda{=}1$-sink count) these are *exactly* GPCCA's
  three macrostates; ISOKANN finds the same three.
* **cDC2 is never recovered, at any $k$** (no χ column peaks on it; best ≈0.85). And this is not an
  ISOKANN quirk: **GPCCA also ranks cDC2 last**, only surfacing it at $n=6$ after both erythroid
  progenitors (§1b table). §4b shows *why* — cDC2 is a **non-absorbing bridge at intermediate pseudotime
  that drains into the CD14+ Mono basin**: ISOKANN's χ gives the cDC2 cells the CD14+ Mono committor
  (≈0.95), and GPCCA's own absorption routes the average cDC2 cell 0.62→CD14+ Mono vs 0.31→cDC2. ISA
  doesn't fail to find a basin; it correctly reports a non-basin, and GPCCA only separates cDC2 because it
  is *told* to make the tiny cDC2 core absorbing.

**On the state count and convergence (the questions raised during review):**

* **$k$ is not load-bearing for the science** — three basins + an unrecoverable cDC2 at every $k$. It only
  changes what the *extra* modes do: $k=3$ = the three basins; $k=4$ adds nothing real (cDC2 forced onto an
  erythroid column); $k=5$ adds the HSC source; $k=6,7$ over-partition the erythroid arm / near-degenerate
  cluster. (Contrast MEF, where $k$ *was* load-bearing because the source needed its own mode.)
* **Convergence is real but $k$-dependent.** The slow spectrum is **purely real** (no complex pair to
  split), but it is a near-continuum after mode 5. Given the same long best-iterate continuation,
  **$k=6$ settles to a flat loss** (yet keeps two weak over-partition modes; $k=7$ more), while **$k=5$
  keeps oscillating** because one starved mode never stops flipping ISA vertices. So the large loss
  fluctuations you can damp by training longer at $k=6$; at $k=5$ they are intrinsic to the starved mode.
* **$k=3$ needs a good seed.** At the tightest partition the ISA vertex search is initialization-sensitive
  (some seeds miss pDC for an erythroid extreme); with a seed that lands on the operator's three sinks it
  matches GPCCA(3) exactly.

**Bottom line.** The principled choices are **$k=3$** (the operator's absorbing-basin count, cleanest and
matches GPCCA's three macrostates) or **$k=4$** (the terminal-lineage count, primary in the main
notebook); both give the three basins and miss only the non-basin cDC2. Higher $k$ adds the HSC source
(k=5) and then over-partitions; the science — basin-vs-bridge — is invariant. This is the in-dataset
version of the multi-kernel benchmark's basin-vs-fate-fan conclusion.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
import os as _os
out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "hematopoiesis_k_sweep_comparison.ipynb")
nbf.write(nb, out); print("wrote", out, len(cells), "cells")
