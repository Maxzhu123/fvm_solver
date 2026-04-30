"""Example: configurable 3D mesh generation with tetgen.
Demonstrates the pipeline:
  1. Define domain surfaces (Box3D, Sphere3D, Cylinder3D).
  2. Generate tetrahedral mesh via create_mesh_3d().
  3. Convert to pyvista UnstructuredGrid for visualisation.
"""
import numpy as np
import pyvista as pv
from mesh_gen.mesh_3d import Box3D, Sphere3D, Cylinder3D, create_mesh_3d
from mesh_gen.mesh_3d.utils import plot_interactive, plot_clip, plot_slice
from mesh_gen.mesh_gen_utils import MeshProps


def build_pyvista_grid(points, tetra):
    """Convert raw (points, tetra) arrays into a pyvista UnstructuredGrid."""
    n_tets = len(tetra)
    cells = np.hstack([np.full((n_tets, 1), 4), tetra]).ravel()
    cell_types = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, points)


def tet_volumes(points, tetra):
    """Compute absolute volume of each tetrahedron. Points (N,3), tetra (M,4)."""
    a = points[tetra[:, 0]]
    b = points[tetra[:, 1]]
    c = points[tetra[:, 2]]
    d = points[tetra[:, 3]]
    return np.abs(np.sum((b - a) * np.cross(c - a, d - a), axis=1)) / 6.0


def cube_sphere_example():
    """Box domain with a spherical hole at the centre."""
    print("=== Cube-with-sphere-hole mesh ===")
    mesh_props = MeshProps(min_area=0.05, max_area=0.1, lengthscale=2.0)
    coords = [
        Box3D(Xmin=[-1, -1, -1], Xmax=[1, 1, 1], dist_req=False, name="Farfield"),
        Sphere3D(center=[0, 0, 0], radius=0.45, hole=True, dist_req=True, name="NavierWall"),
    ]
    mesh_specs, marker_tags = create_mesh_3d(coords, mesh_props)
    (points, tetra), (_, face_markers), (int_faces, bound_faces) = mesh_specs
    print(f"  Points: {points.shape[0]}, Tets: {tetra.shape[0]}")
    print(f"  Interior faces: {int_faces.shape[0]}, Boundary faces: {bound_faces.shape[0]}")
    face_tag = [marker_tags[int(i)] for i in face_markers]
    from collections import Counter
    print(f"  Boundary tags: {Counter(face_tag)}")
    grid = build_pyvista_grid(points, tetra)
    grid.cell_data["volume"] = tet_volumes(points, tetra)
    print(f"  Volume range: [{grid.cell_data['volume'].min():.4f}, "
          f"{grid.cell_data['volume'].max():.4f}]")
    return grid


def pipe_example():
    """Box domain with a cylindrical hole along the z-axis."""
    print("\n=== Pipe (cylinder hole) mesh ===")
    mesh_props = MeshProps(min_area=0.025, max_area=0.1, lengthscale=0.5)
    L = 1.5
    coords = [
        Box3D(Xmin=[-L/2, -L/2, -L/2], Xmax=[L/2, L/2, L/2], dist_req=False, name="Farfield"),
        Cylinder3D(center=[0, 0, 0], radius=0.5, height=L * 0.8, hole=True, dist_req=True, name="NavierWall"),
    ]
    mesh_specs, marker_tags = create_mesh_3d(coords, mesh_props)
    (points, tetra), (_, face_markers), (int_faces, bound_faces) = mesh_specs
    print(f"  Points: {points.shape[0]}, Tets: {tetra.shape[0]}")
    print(f"  Interior faces: {int_faces.shape[0]}, Boundary faces: {bound_faces.shape[0]}")
    grid = build_pyvista_grid(points, tetra)
    grid.cell_data["volume"] = tet_volumes(points, tetra)
    # print(f"  Volume range: [{grid.cell_data['volume'].min():.4f}, "
    #       f"{grid.cell_data['volume'].max():.4f}]")
    return grid


def main():
    # grid_cube_sphere = cube_sphere_example()
    #
    # print("\nOpening interactive viewer (cube + sphere hole) ...")
    # plot_interactive(grid_cube_sphere)

    grid_pipe = pipe_example()
    print("\nOpening interactive viewer (pipe) ...")
    plot_interactive(grid_pipe)
    plot_slice(grid_pipe)


if __name__ == "__main__":
    main()
