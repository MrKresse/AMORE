
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

sd   = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/states/state_2"
png  = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/states/state_2.png"
pse  = r"/home/htc/jkresse/AMORE/examples/2cm2_full/dim3_pocket/states/state_2.pse"
lig  = "resn KB8"

# ── representative conformation ──────────────────────────────────────────────
cmd.load(sd + "/rep.pdb", "conf")
cmd.hide("everything", "conf")
cmd.show("cartoon", "conf and polymer")           # semi-transparent grey context (chopped)
cmd.color("grey70", "conf and polymer")
cmd.color("grey80", "conf and polymer and chain B")
# pocket: protein residues within 7 A of the ligand, thin grey sticks (7, not 5: a
# genuinely-real ~5.2 A contact -- not a PBC artifact, see _pack_ligand -- was landing
# just outside a 5 A cutoff, selecting nothing)
cmd.select("pocket", "byres (conf and polymer within 7 of (conf and " + lig + "))")
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
# frame ligand + pocket together, not the ligand alone -- orienting/zooming on just the
# ligand guarantees IT is in view but not the (already-selected) nearby protein context,
# which for a looser pose can sit just outside a ligand-only zoom buffer despite being
# genuinely close (see the "pocket" cutoff widening above, same root issue).
cmd.orient("(conf and " + lig + ") or pocket")
cmd.zoom("(conf and " + lig + ") or pocket", buffer=4)
cmd.deselect()
cmd.save(pse)                                       # editable PyMOL session
cmd.ray(3840, 2880)                                 # 4K for zoom-in
cmd.png(png, dpi=300)
