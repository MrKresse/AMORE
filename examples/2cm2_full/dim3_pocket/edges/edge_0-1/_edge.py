
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

ed  = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/edges/edge_0-1"
pse = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/edges/edge_0-1.pse"
png = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/edges/edge_0-1.png"
lig = "resn KB8"

# playable transition object (scrub states = walk the pathway j -> i)
cmd.load(ed + "/rep.pdb", "path")
cmd.load_traj(ed + "/path.dcd", "path", state=1)
cmd.set("all_states", 0, "path")
cmd.hide("everything", "path")
cmd.show("cartoon", "path and polymer")
cmd.color("grey70", "path and polymer")
cmd.select("pocket", "byres (path and polymer within 7 of (path and " + lig + "))")
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
# frame the full ligand sweep range AND the pocket together -- same reasoning as the
# state-render camera fix: pocket alone can sit just outside a ligand-only zoom buffer.
cmd.orient("(sweep and " + lig + ") or pocket")
cmd.zoom("(sweep and " + lig + ") or pocket", buffer=4)
cmd.save(pse)
cmd.ray(3840, 2880)
cmd.png(png, dpi=300)
