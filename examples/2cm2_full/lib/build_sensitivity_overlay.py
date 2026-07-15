# -*- coding: utf-8 -*-
"""
build_sensitivity_overlay.py -- PyMOL chi-sensitivity overlay: per-residue average
||grad s_ij||^2 (amore.chi.chi_sensitivity, EdgeCV), summed per residue, written into the
a-posteriori medoid path's B-factor column and rendered with PyMOL's viridis spectrum
coloring (matching the notebook heatmap's colormap) -- same playable-path +
ligand-transition-sweep-halo aesthetic as the standard edge .pse (render.py's _EDGE_PML),
but with the protein cartoon/pocket sticks colored by sensitivity instead of plain grey.

Uses the PLAIN per-window medoid path (build_medoid_string.py) -- the neighbor-pooled
"smoothed" variant was tried and dropped (see build_medoid_string.py's docstring:
degenerate repeated frames across consecutive windows, not genuine smoothness).

Requires medoid_string_results.pkl (build_medoid_string.py) for the medoid frame indices.
No standalone pymolgradient.py module -- built directly on render.py's existing PyMOL
plumbing (_pack_ligand, MAMBA/MAMBA_ROOT) instead.

Run (no GPU/OpenMM needed -- regular venv):
    python examples/2cm2_full/lib/build_sensitivity_overlay.py
"""
import os
import pickle
import sys
import subprocess

import numpy as np
import torch as pt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "2cm2", "lib"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import data                                                    # noqa: E402
import comfeat                                                  # noqa: E402
import analysis as A                                             # noqa: E402
import render                                                     # noqa: E402
from amore.chi import chi_sensitivity                              # noqa: E402
from amore.mep.simplex import EdgeCV                                # noqa: E402

data.NSTART, data.NEND, data.LAG = 0, 2979, 20
K = 3
MAX_FRAMES = 500
DEVICE = "cpu"
SCRATCH = os.environ.get("CM2_MFEP_SCRATCH", "/scratch/htc/jkresse/2cm2_mfep")

_SENS_PATH_PML = r"""
from pymol import cmd
cmd.reinitialize()
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 1)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_side_chain_helper", 1)
cmd.set("cartoon_transparency", 0.35)
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
# PyMOL has no built-in "viridis" spectrum palette (confirmed: cmd.spectrum's named-palette
# list is blue_red/rainbow/etc-family only) -- pass an explicit 6-stop hex approximation of
# matplotlib's viridis instead, matching the notebook heatmap's colormap for consistency.
VIRIDIS = "0x440154 0x414487 0x2a788e 0x22a884 0x7ad151 0xfde725"

# playable transition object (scrub states = walk the smoothed medoid path), cartoon +
# pocket sticks colored by per-residue chi-sensitivity (viridis, static across states --
# PyMOL's load_traj only animates coordinates, not b-factor)
cmd.load(ed + "/rep.pdb", "path")
cmd.load_traj(ed + "/path.dcd", "path", state=1)
cmd.set("all_states", 0, "path")
cmd.hide("everything", "path")
cmd.show("cartoon", "path and polymer")
cmd.spectrum("b", VIRIDIS, "path and polymer")
cmd.select("pocket", "byres (path and polymer within 7 of (path and " + lig + "))")
cmd.show("sticks", "pocket")
cmd.set("stick_radius", 0.10, "pocket")
cmd.spectrum("b", VIRIDIS, "pocket")
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
cmd.orient("(sweep and " + lig + ") or pocket")
cmd.zoom("(sweep and " + lig + ") or pocket", buffer=4)
cmd.save(pse)
cmd.ray(3840, 2880)
cmd.png(png, dpi=300)
"""


def render_sensitivity_path(ed, pse, png):
    script = os.path.join(ed, "_sens_path.py")
    with open(script, "w") as f:
        f.write(_SENS_PATH_PML.format(ed=ed, pse=pse, png=png, lig=render.LIGAND))
    env = dict(os.environ, MAMBA_ROOT_PREFIX=render.MAMBA_ROOT)
    r = subprocess.run([render.MAMBA, "run", "-n", "pymol", "pymol", "-cq", script],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0 and os.path.exists(pse), r.stdout + r.stderr


def write_medoid_path_with_sensitivity(u_full, order, atom_b, ed):
    """Same pattern as render._extract_edge (pack ligand, superpose on frame0 by protein
    CA, write rep.pdb + path.dcd) but also stamps atom_b into the topology's B-factor
    column before writing rep.pdb, so PyMOL's static (non-animated) b-factor spectrum
    colors the whole playable path by the same per-residue sensitivity value."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    u_full.trajectory[data.NSTART + int(order[0])]
    u_full.atoms.tempfactors = atom_b
    keep = u_full.select_atoms(render.KEEP_SEL)
    render._pack_ligand(keep, u_full.dimensions)
    ref = mda.Merge(keep)
    ref.atoms.positions = keep.positions
    ref.atoms.write(os.path.join(ed, "rep.pdb"))
    with mda.Writer(os.path.join(ed, "path.dcd"), keep.n_atoms) as W:
        for a in order:
            u_full.trajectory[data.NSTART + int(a)]
            render._pack_ligand(keep, u_full.dimensions)
            align.alignto(keep, ref, select="protein and name CA")
            W.write(keep)


def main():
    model, m = comfeat.load_trained_model_pocket(K, device=DEVICE)
    feat = comfeat.make_torch_featurizer_pocket(device=DEVICE)
    chi = np.asarray(m["chi"])

    with open(os.path.join(SCRATCH, "medoid_string_results.pkl"), "rb") as f:
        medoid_results = pickle.load(f)

    import MDAnalysis as mda
    u_full = mda.Universe(data.pdb_path(), data.dcd_path())

    def real_frame_positions_nm(idx):
        u_full.trajectory[data.NSTART + int(idx)]
        return (u_full.atoms.positions.astype(np.float64) / 10.0).flatten()

    u_top = mda.Universe(data.pdb_path())
    prot_ag = u_top.select_atoms("protein")
    lig_ag = u_top.select_atoms(f"resname {data.LIG_RESNAME}")
    resindex_full = np.full(u_top.atoms.n_atoms, -1, dtype=int)
    resindex_full[prot_ag.indices] = np.asarray(prot_ag.resindices)
    LIG_LABEL = int(resindex_full[prot_ag.indices].max()) + 1
    resindex_full[lig_ag.indices] = LIG_LABEL
    valid_atoms = resindex_full >= 0
    n_res_total = LIG_LABEL + 1

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           "dim3_pocket", "edges"))

    rows = A.edge_table(chi)
    for r in rows:
        i, j = r["edge"]
        if (i, j) not in medoid_results:
            print(f"edge {i}-{j}: no medoid string result -- skipping")
            continue
        order, s = A.pathway_frames(chi, i, j)
        if len(order) > MAX_FRAMES:
            pick = np.linspace(0, len(order) - 1, MAX_FRAMES).round().astype(int)
            order, s = order[pick], s[pick]
        print(f"edge {i}-{j}: {len(order)} frames for sensitivity")
        xyz = np.stack([real_frame_positions_nm(a) for a in order])
        xs_t = pt.as_tensor(xyz, dtype=pt.float32, device=DEVICE)
        _, avg_g2 = chi_sensitivity(EdgeCV(model, i, j), feat, xs_t, nbins=1)
        avg_g2 = avg_g2[0].detach().cpu().numpy()

        res_sens = np.zeros(n_res_total)
        ridx = resindex_full[valid_atoms]
        np.add.at(res_sens, ridx, avg_g2[valid_atoms])
        # PDB B-factor fields are %6.2f (2 decimal places) -- these raw sensitivity values
        # are ~1e-4 to 1e-3, far below that resolution, so they'd all silently round to
        # 0.00 (confirmed directly: this produced a uniformly flat cartoon on the first
        # pass). Rescale to 0-99 before writing so precision survives the file format;
        # cmd.spectrum auto-detects min/max on load so the absolute scale doesn't matter.
        res_sens_scaled = res_sens / res_sens.max() * 99.0
        atom_b = np.zeros(u_top.atoms.n_atoms)
        atom_b[valid_atoms] = res_sens_scaled[ridx]

        medoid_order = medoid_results[(i, j)]["plain_idx"]
        ed = os.path.join(out_dir, f"edge_{i}-{j}_sensitivity_medoid")
        os.makedirs(ed, exist_ok=True)
        write_medoid_path_with_sensitivity(u_full, medoid_order, atom_b, ed)

        pse = os.path.join(out_dir, f"edge_{i}-{j}_sensitivity_medoid.pse")
        png = os.path.join(out_dir, f"edge_{i}-{j}_sensitivity_medoid.png")
        ok, log = render_sensitivity_path(ed, pse, png)
        print(f"edge {i}-{j} sensitivity-on-medoid-path overlay -> {'OK' if ok else 'FAIL'}")
        if not ok:
            print(log)


if __name__ == "__main__":
    main()
