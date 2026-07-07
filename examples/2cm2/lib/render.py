# -*- coding: utf-8 -*-
"""
render.py — per-metastable-state PyMOL renders for the 2cm2 ISOKANN models.

For each membership state i of a k-dim model we:
  * pick a **representative frame** (chi_i closest to 0.95) and a small **ensemble** of other
    frames with chi_i in [0.9, 1.0] (the conformations firmly committed to state i);
  * keep only the **protein + KB8 ligand** (water HOH and ions NA/CL dropped), make the
    complex whole across the periodic box, and **superpose** every frame on the representative
    by protein Calpha — so the state is structurally aligned, not chopped;
  * write `rep.pdb` (representative, opaque) and `ensemble.dcd` (the cloud) per state.

`render_states` then drives headless PyMOL (micromamba env `pymol`) to produce one PNG per
state: grey protein cartoon, orange KB8 sticks, white background, the representative fully
coloured with the ensemble as a faint transparent overlay, centred on the ligand.
"""
from __future__ import annotations
import os, sys, subprocess
import numpy as np
import torch as pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data                                                # noqa: E402

LIGAND = "KB8"
KEEP_SEL = f"protein or resname {LIGAND}"                  # drop HOH / NA / CL
MAMBA = "/home/htc/jkresse/.local/bin/micromamba"
MAMBA_ROOT = "/home/htc/jkresse/micromamba"


# ── representative + ensemble frame selection ────────────────────────────────
def state_frames(chi, i, lo=0.9, hi=1.0, target=0.95, n_ensemble=12):
    """Return (rep_anchor, ensemble_anchors) for state i.

    rep = frame with chi_i closest to `target` inside [lo,hi]; if the band is empty, fall
    back to the global argmax of chi_i.  ensemble = up to `n_ensemble` other band frames,
    evenly spread across the band so the cloud spans the whole [lo,hi] range."""
    chi = np.asarray(chi)
    ci = chi[:, i]
    band = np.where((ci >= lo) & (ci <= hi))[0]
    if len(band) == 0:
        rep = int(np.argmax(ci))
        return rep, np.array([], int)
    rep = int(band[np.argmin(np.abs(ci[band] - target))])
    others = band[band != rep]
    if len(others) > n_ensemble:
        order = others[np.argsort(ci[others])]
        pick = np.linspace(0, len(order) - 1, n_ensemble).round().astype(int)
        others = order[pick]
    return rep, others


# ── structure export (MDAnalysis) ────────────────────────────────────────────
def _pack_ligand(keep, box):
    """Shift the ligand into the same periodic image as the protein (minimum image of the
    COM separation).  unwrap() only makes each fragment whole; if the ligand is a separate
    fragment it can still sit a box-vector away, which after protein-CA alignment throws
    phantom ligand copies across the render.  Call every frame BEFORE aligning."""
    prot = keep.select_atoms("protein")
    lg = keep.select_atoms(f"resname {LIGAND}")
    if lg.n_atoms == 0 or prot.n_atoms == 0:
        return
    b = np.asarray(box[:3], float)
    shift = b * np.round((prot.center_of_mass() - lg.center_of_mass()) / b)
    lg.positions = lg.positions + shift


def export_state_structures(chi, k, out_dir, nstart=None, lo=0.9, hi=1.0,
                            target=0.95, n_ensemble=12, verbose=True):
    """Write rep.pdb + ensemble.dcd for every state under out_dir/state_i/.

    Frames are protein+ligand only, made whole across the box, and superposed on the
    representative by protein Calpha.  Returns a list of per-state info dicts."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    from MDAnalysis import transformations as trans

    nstart = data.NSTART if nstart is None else nstart
    os.makedirs(out_dir, exist_ok=True)
    u = mda.Universe(data.pdb_path(), data.dcd_path())
    keep = u.select_atoms(KEEP_SEL)
    # make the kept complex whole every frame (protein is already whole, but this also pulls
    # the ligand into the protein's periodic image so nothing is chopped).  unwrap needs
    # bonds, which a plain PDB lacks — guess them; fall back to no-unwrap if that fails.
    try:
        keep.guess_bonds()
        u.trajectory.add_transformations(trans.unwrap(keep))
        if verbose:
            print("  [render] PBC unwrap enabled (bonds guessed)", flush=True)
    except Exception as e:
        if verbose:
            print(f"  [render] unwrap skipped ({type(e).__name__}); protein already whole",
                  flush=True)

    ca = "protein and name CA"
    infos = []
    for i in range(k):
        rep, ens = state_frames(chi, i, lo, hi, target, n_ensemble)
        sd = os.path.join(out_dir, f"state_{i}")
        os.makedirs(sd, exist_ok=True)

        # reference = representative frame
        u.trajectory[int(nstart + rep)]
        ref = mda.Merge(keep)                              # static copy of the rep complex
        ref.atoms.positions = keep.positions
        ref_ca = ref.select_atoms(ca)
        ref.atoms.write(os.path.join(sd, "rep.pdb"))

        # ensemble, each superposed on the rep BY THE LIGAND (so the ligand poses collapse
        # onto one another — one crisp ligand with a faint halo of its own flexibility)
        frames = [rep] + list(ens)
        with mda.Writer(os.path.join(sd, "ensemble.dcd"), keep.n_atoms) as W:
            for a in frames:
                u.trajectory[int(nstart + a)]
                align.alignto(keep, ref, select=f"resname {LIGAND}")
                W.write(keep)
        ci = float(np.asarray(chi)[rep, i])
        infos.append(dict(state=i, rep_anchor=int(rep), rep_frame=int(nstart + rep),
                          rep_chi=ci, n_ensemble=len(frames), dir=sd))
        if verbose:
            print(f"  state {i}: rep frame {nstart+rep} (chi={ci:.3f}), "
                  f"{len(frames)} frames -> {sd}", flush=True)
    return infos


# ── PyMOL rendering (headless, micromamba env) ───────────────────────────────
_PML = r"""
from pymol import cmd
cmd.reinitialize()
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 1)                  # 4K is already high-res; AA1 keeps ray time sane
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_side_chain_helper", 1)
cmd.set("cartoon_transparency", 0.55)    # see the ligand through any protein in front
cmd.set("two_sided_lighting", 1)         # fix dark backface smudges on chopped cartoon
cmd.set("backface_cull", 0)
cmd.set("ray_interior_color", "grey80")  # clipped cartoon interiors grey, not black
cmd.set("ambient", 0.4)
cmd.set("specular", 0.15)
cmd.set_color("faintorange", [1.00, 0.83, 0.58])

sd   = r"{sd}"
png  = r"{png}"
pse  = r"{pse}"
lig  = "resn {lig}"

# ── representative conformation ──────────────────────────────────────────────
cmd.load(sd + "/rep.pdb", "conf")
cmd.hide("everything", "conf")
cmd.show("cartoon", "conf and polymer")           # semi-transparent grey context (chopped)
cmd.color("grey70", "conf and polymer")
cmd.color("grey80", "conf and polymer and chain B")
# pocket: protein residues within 5 A of the ligand, thin grey sticks
cmd.select("pocket", "byres (conf and polymer within 5 of (conf and " + lig + "))")
cmd.show("sticks", "pocket")
cmd.set("stick_radius", 0.10, "pocket")
cmd.color("grey60", "pocket and elem C")
cmd.util.cnc("pocket")
# THE one fat ligand — the representative pose, opaque orange
cmd.show("sticks", "conf and " + lig)
cmd.color("orange", "conf and " + lig)
cmd.util.cnc("conf and " + lig)
cmd.set("stick_radius", 0.35, "conf and " + lig)

# ── faint ensemble ligand halo (aligned on the ligand) ───────────────────────
cmd.load(sd + "/rep.pdb", "ens")
cmd.load_traj(sd + "/ensemble.dcd", "ens", state=1)
cmd.set("all_states", 1, "ens")
cmd.hide("everything", "ens")
cmd.show("sticks", "ens and " + lig)
cmd.color("faintorange", "ens and " + lig)        # very faint, opaque
cmd.set("stick_radius", 0.07, "ens and " + lig)

cmd.hide("everything", "hydro")                    # declutter

# ── camera: ligand centred and facing outward, protein behind ────────────────
cmd.orient("conf and " + lig)
cmd.zoom("conf and " + lig, buffer=4)
cmd.deselect()
cmd.save(pse)                                       # editable PyMOL session
cmd.ray(3840, 2880)                                 # 4K for zoom-in
cmd.png(png, dpi=300)
"""


def _render_one(sd, png, pse=None):
    pse = pse or os.path.splitext(png)[0] + ".pse"
    script = os.path.join(sd, "_render.py")
    with open(script, "w") as f:
        f.write(_PML.format(sd=sd, png=png, pse=pse, lig=LIGAND))
    env = dict(os.environ, MAMBA_ROOT_PREFIX=MAMBA_ROOT)
    r = subprocess.run([MAMBA, "run", "-n", "pymol", "pymol", "-cq", script],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0 and os.path.exists(png), r.stdout + r.stderr


def render_states(infos, out_dir, verbose=True):
    """Render one PNG per state from the exported structures.  Returns list of PNG paths."""
    pngs = []
    for info in infos:
        sd = info["dir"]
        png = os.path.join(out_dir, f"state_{info['state']}.png")
        ok, log = _render_one(sd, png)
        if verbose:
            tail = [l for l in log.splitlines() if "Ray:" in l or "ScenePNG" in l
                    or "Error" in l or "error" in l]
            print(f"  state {info['state']} -> {'OK' if ok else 'FAIL'} "
                  f"{os.path.basename(png)}  {tail[-1] if tail else ''}", flush=True)
        pngs.append(png)
    return pngs


# ── edge (transition-pathway) .pse ───────────────────────────────────────────
def export_edge_structures(chi, k, out_dir, n_path=60, tau_edge=0.8, force=False, verbose=True):
    """For every RELEVANT edge, write the zeroth-order pathway (on-edge frames ordered by
    s_ij) as a protein+ligand trajectory, subsampled to ~n_path frames and superposed on the
    FIRST (s-min) frame by protein Cα — so the protein is steady and the ligand's transition
    is what moves.  Writes edge_i-j/{path.dcd, rep.pdb}.  Returns per-edge info dicts.

    Per edge, the (slow) MDAnalysis extraction is skipped when path.dcd already exists and
    `force` is False — the info dict is still returned from the cheap chi-only computation."""
    import analysis as A
    os.makedirs(out_dir, exist_ok=True)
    chi = np.asarray(chi)
    rows = A.edge_table(chi)
    rare = set(A.rare_edges(rows))

    # cheap pass: which edges, subsampled order + s-range (no trajectory I/O)
    plans = []
    for r in rows:
        if not r["relevant"]:
            continue
        i, j = r["edge"]
        order, s = A.pathway_frames(chi, i, j, tau_edge=tau_edge)
        if len(order) < 3:
            continue
        if len(order) > n_path:
            pick = np.linspace(0, len(order) - 1, n_path).round().astype(int)
            order, s = order[pick], s[pick]
        plans.append(((i, j), order, s))

    need = [p for p in plans
            if force or not os.path.exists(os.path.join(out_dir, f"edge_{p[0][0]}-{p[0][1]}", "path.dcd"))]

    u = keep = None
    if need:
        import MDAnalysis as mda
        from MDAnalysis import transformations as trans
        u = mda.Universe(data.pdb_path(), data.dcd_path())
        keep = u.select_atoms(KEEP_SEL)
        try:
            keep.guess_bonds()
            u.trajectory.add_transformations(trans.unwrap(keep))
        except Exception:
            pass

    infos = []
    for (i, j), order, s in plans:
        ed = os.path.join(out_dir, f"edge_{i}-{j}")
        os.makedirs(ed, exist_ok=True)
        if force or not os.path.exists(os.path.join(ed, "path.dcd")):
            _extract_edge(u, keep, ed, order)
        infos.append(dict(edge=(i, j), dir=ed, n=len(order),
                          s_lo=float(s.min()), s_hi=float(s.max()),
                          rare=(i, j) in rare))
        if verbose:
            print(f"  edge {i}-{j}: {len(order)} frames "
                  f"(s {s.min():.2f}->{s.max():.2f}){' RARE' if (i,j) in rare else ''}",
                  flush=True)
    return infos


def _extract_edge(u, keep, ed, order):
    """Write edge_dir/{rep.pdb, path.dcd}: protein+ligand pathway, ligand packed to the
    protein image, superposed on the first frame by protein Cα."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    u.trajectory[int(data.NSTART + order[0])]
    _pack_ligand(keep, u.dimensions)
    ref = mda.Merge(keep); ref.atoms.positions = keep.positions
    ref.atoms.write(os.path.join(ed, "rep.pdb"))
    with mda.Writer(os.path.join(ed, "path.dcd"), keep.n_atoms) as W:
        for a in order:
            u.trajectory[int(data.NSTART + a)]
            _pack_ligand(keep, u.dimensions)
            align.alignto(keep, ref, select="protein and name CA")
            W.write(keep)


_EDGE_PML = r"""
from pymol import cmd
cmd.reinitialize()
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 1)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_side_chain_helper", 1)
cmd.set("cartoon_transparency", 0.55)
cmd.set("two_sided_lighting", 1)
cmd.set("backface_cull", 0)
cmd.set("ray_interior_color", "grey80")
cmd.set("ambient", 0.4)
cmd.set("specular", 0.15)
cmd.set_color("faintorange", [1.00, 0.83, 0.58])

ed  = r"{ed}"
pse = r"{pse}"
png = r"{png}"
lig = "resn {lig}"

# playable transition object (scrub states = walk the pathway j -> i)
cmd.load(ed + "/rep.pdb", "path")
cmd.load_traj(ed + "/path.dcd", "path", state=1)
cmd.set("all_states", 0, "path")
cmd.hide("everything", "path")
cmd.show("cartoon", "path and polymer")
cmd.color("grey70", "path and polymer")
cmd.select("pocket", "byres (path and polymer within 5 of (path and " + lig + "))")
cmd.show("sticks", "pocket")
cmd.set("stick_radius", 0.10, "pocket")
cmd.color("grey60", "pocket and elem C")
cmd.util.cnc("pocket")
cmd.show("sticks", "path and " + lig)
cmd.color("orange", "path and " + lig)
cmd.util.cnc("path and " + lig)
cmd.set("stick_radius", 0.30, "path and " + lig)

# faint halo = the whole ligand transition sweep (all pathway poses at once)
cmd.load(ed + "/rep.pdb", "sweep")
cmd.load_traj(ed + "/path.dcd", "sweep", state=1)
cmd.set("all_states", 1, "sweep")
cmd.hide("everything", "sweep")
cmd.show("sticks", "sweep and " + lig)
cmd.color("faintorange", "sweep and " + lig)
cmd.set("stick_radius", 0.06, "sweep and " + lig)

cmd.hide("everything", "hydro")
cmd.deselect()
cmd.orient("sweep and " + lig)          # frame the full ligand range
cmd.zoom("sweep and " + lig, buffer=4)
cmd.save(pse)
cmd.ray(3840, 2880)
cmd.png(png, dpi=300)
"""


def _render_edge_one(ed, pse, png):
    script = os.path.join(ed, "_edge.py")
    with open(script, "w") as f:
        f.write(_EDGE_PML.format(ed=ed, pse=pse, png=png, lig=LIGAND))
    env = dict(os.environ, MAMBA_ROOT_PREFIX=MAMBA_ROOT)
    r = subprocess.run([MAMBA, "run", "-n", "pymol", "pymol", "-cq", script],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0 and os.path.exists(pse), r.stdout + r.stderr


def build_edge_pses(k, use_pbc=True, out_dir=None, force=False, verbose=True, **kw):
    """One `.pse` (+ preview PNG) per relevant simplex edge of a cached k-dim model, same
    aesthetic as the state renders: see-through grey cartoon, orange KB8, 5 Å pocket sticks,
    plus the ligand transition sweep as a faint halo and a playable multi-state pathway."""
    import run
    out_dir = out_dir or os.path.join(HERE, "..", f"dim{k}", "edges")
    out_dir = os.path.abspath(out_dir)
    m = run.get_model(k, use_pbc=use_pbc, verbose=False)
    infos = export_edge_structures(m["chi"], k, out_dir, force=force, verbose=verbose, **kw)
    for info in infos:
        i, j = info["edge"]
        pse = os.path.join(out_dir, f"edge_{i}-{j}.pse")
        png = os.path.join(out_dir, f"edge_{i}-{j}.png")
        if not force and os.path.exists(pse):
            info["pse"] = pse; continue
        ok, log = _render_edge_one(info["dir"], pse, png)
        info["pse"] = pse if ok else None
        if verbose:
            print(f"  edge {i}-{j} -> {'OK' if ok else 'FAIL'} {os.path.basename(pse)}",
                  flush=True)
    return infos


def render_model_states(k, use_pbc=True, out_dir=None, force=False, verbose=True, **kw):
    """Full per-state render for a cached k-dim model: export structures then render PNGs.

    If all state PNGs already exist and `force` is False, the (slow) PyMOL step is skipped and
    the cached PNG paths are returned — so a notebook can call this cheaply on re-run."""
    import run
    out_dir = out_dir or os.path.join(HERE, "..", f"dim{k}", "states")
    out_dir = os.path.abspath(out_dir)
    pngs = [os.path.join(out_dir, f"state_{i}.png") for i in range(k)]
    if not force and all(os.path.exists(p) for p in pngs):
        if verbose:
            print(f"[render] all {k} state PNGs cached in {out_dir}", flush=True)
        return None, pngs
    m = run.get_model(k, use_pbc=use_pbc, verbose=False)
    infos = export_state_structures(m["chi"], k, out_dir, nstart=m["nstart"],
                                    verbose=verbose, **kw)
    pngs = render_states(infos, out_dir, verbose=verbose)
    return infos, pngs
