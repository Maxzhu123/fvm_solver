import gmsh

gmsh.initialize()
gmsh.model.add("tracked_surfaces")

L = 1.0
cx, cy, cz = 0.5, 0.5, 0.5
r = 0.2

# ----------------------------
# Create geometry
# ----------------------------
box = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
sphere = gmsh.model.occ.addSphere(cx, cy, cz, r)

gmsh.model.occ.synchronize()

# Original boundary surfaces
box_surfs_old = gmsh.model.getBoundary(
    [(3, box)],
    oriented=False,
    recursive=False
)

sphere_surfs_old = gmsh.model.getBoundary(
    [(3, sphere)],
    oriented=False,
    recursive=False
)

# ----------------------------
# Fragment while tracking surfaces
# ----------------------------
objects = box_surfs_old + [(3, box)]
tools = sphere_surfs_old + [(3, sphere)]

out, out_map = gmsh.model.occ.fragment(
    objects,
    tools,
    removeObject=True,
    removeTool=True
)

gmsh.model.occ.synchronize()

inputs = objects + tools

# Helper: map old entity/entities to new entities
def mapped_entities(old_entities):
    result = []
    for old in old_entities:
        i = inputs.index(old)
        result.extend(out_map[i])
    # remove duplicates
    return list(dict.fromkeys(result))

# ----------------------------
# New surfaces corresponding to original box/sphere surfaces
# ----------------------------
box_surfs_new = [
    tag for dim, tag in mapped_entities(box_surfs_old)
    if dim == 2
]

sphere_surfs_new = [
    tag for dim, tag in mapped_entities(sphere_surfs_old)
    if dim == 2
]

print("Tracked box surface entities:", box_surfs_new)
print("Tracked sphere surface entities:", sphere_surfs_new)

# ----------------------------
# Determine final volume: box minus sphere
# ----------------------------
box_volume_new = [
    tag for dim, tag in mapped_entities([(3, box)])
    if dim == 3
]

sphere_volume_new = [
    tag for dim, tag in mapped_entities([(3, sphere)])
    if dim == 3
]

# The sphere volume is the part to remove.
# The remaining box volume is box fragments minus sphere fragments.
box_minus_sphere_vols = sorted(set(box_volume_new) - set(sphere_volume_new))

print("Box-derived volumes:", box_volume_new)
print("Sphere-derived volumes:", sphere_volume_new)
print("Final volume:", box_minus_sphere_vols)

# Remove the inside sphere volume, but keep its boundary surface
gmsh.model.occ.remove(
    [(3, v) for v in sphere_volume_new],
    recursive=False
)

gmsh.model.occ.synchronize()

# ----------------------------
# Physical groups
# ----------------------------
vol_phys = gmsh.model.addPhysicalGroup(3, box_minus_sphere_vols)
gmsh.model.setPhysicalName(3, vol_phys, "box_minus_sphere")

box_phys = gmsh.model.addPhysicalGroup(2, box_surfs_new)
gmsh.model.setPhysicalName(2, box_phys, "box_surfaces")

sphere_phys = gmsh.model.addPhysicalGroup(2, sphere_surfs_new)
gmsh.model.setPhysicalName(2, sphere_phys, "sphere_hole_surface")

# ----------------------------
# Mesh
# ----------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.04)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.08)

gmsh.model.mesh.generate(3)

# Mesh facets for each tracked group
print("\nBox surface mesh facets:")
for s in box_surfs_new:
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, s)
    print(f"surface {s}: {sum(len(e) for e in elem_tags)} facets")

print("\nSphere hole mesh facets:")
for s in sphere_surfs_new:
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, s)
    print(f"surface {s}: {sum(len(e) for e in elem_tags)} facets")

gmsh.write("tracked_box_sphere_hole.msh")
gmsh.finalize()