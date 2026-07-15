
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

ed  = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/edges/edge_1-2_sensitivity_medoid"
pse = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/edges/edge_1-2_sensitivity_medoid.pse"
png = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/edges/edge_1-2_sensitivity_medoid.png"
lig = "resn KB8"
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
