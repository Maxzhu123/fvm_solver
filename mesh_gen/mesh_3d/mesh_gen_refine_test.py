import numpy as np
import pyvista as pv
import tetgen
import os
import contextlib

L = 4
hole_radius = 0.45
hole_center = np.array([0.0, 0.0, 0.0])

@contextlib.contextmanager
def suppress_c_stdout():
    stdout_fd = os.dup(1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.close(stdout_fd)

def background_mesh(surface):
    # -----------------------------
    # This mesh must cover the full domain.
    # It does NOT need to match the final mesh.
    bg_tet = tetgen.TetGen(surface)

    print("Generating background mesh")
    with suppress_c_stdout():
        bg_tet.tetrahedralize(switches="pq1.2a0.2")

        bgmesh_raw = bg_tet.grid

        # Keep only linear tetrahedral cells
        bgmesh = bgmesh_raw.extract_cells(
            bgmesh_raw.celltypes == pv.CellType.TETRA
        ).clean()

    return bgmesh

def geometry():
    # -----------------------------
    # Main geometry: cube with sphere hole
    # -----------------------------

    cube = pv.Cube(
        center=(0, 0, 0),
        x_length=L,
        y_length=L,
        z_length=L,
    ).triangulate()

    sphere = pv.Sphere(
        center=hole_center,
        radius=hole_radius,
        theta_resolution=16,
        phi_resolution=16,
    ).triangulate()

    # sphere = pv.Cylinder(
    #     center=tuple(hole_center),
    #     radius=hole_radius,
    #     height=L*0.8,
    #     resolution=50,
    #     capping=True,
    # ).triangulate()

    surface = pv.merge([cube, sphere]).clean()
    return surface

def mesh_refine(points):
    dist = np.linalg.norm(points - hole_center, axis=1) - hole_radius

    h_min = 0.05
    h_max = 0.2
    transition = 0.5

    t = np.clip(dist / transition, 0.0, 1.0)
    sizes = h_min + (h_max - h_min) * t

    return sizes

def gen_refined_mesh(surface, bgmesh):
    tet = tetgen.TetGen(surface)
    tet.add_hole(hole_center)
    print("Tet done ")

    refine_kwargs = dict(order=1, mindihedral=30, minratio=1.5)
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    tet.tetrahedralize(
        bgmesh=bgmesh, **refine_kwargs, verbose=True, switches="q1.2V"
    )

    return tet

def main():
    surface = geometry()
    # -----------------------------
    # Background mesh
    bgmesh = background_mesh(surface)

    points = bgmesh.points
    sizes = mesh_refine(points)
    bgmesh.point_data["target_size"] = sizes

    tet = gen_refined_mesh(surface, bgmesh)
    mesh = tet.grid
    # -----------------------------
    # Visualize
    # -----------------------------
    from utils import plot_clip, plot_interactive

    print(mesh.cell_centers().points.shape)
    plot_interactive(mesh)

if __name__ == "__main__":
    main()