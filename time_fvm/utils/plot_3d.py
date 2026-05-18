import numpy as np
import torch
from torch import Tensor
import pyvista as pv

def create_unstructured_grid(points: Tensor, cells: Tensor) -> pv.UnstructuredGrid:
    """
    Creates a PyVista UnstructuredGrid representation from points and cell connectivity.

    Parameters
    ----------
    points : shape (n_points, 3)
        Mesh vertex coordinates.
    cells : shape (n_cells, 4)
        Cell connectivity. (Tetrahedrons)
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    faces = []
    for cell in cells:
        cell = list(cell)
        faces.extend([len(cell), *cell])

    faces = np.asarray(faces, dtype=np.int64)

    # 10 is VTK_TETRA
    cell_types = np.full(len(cells), 10, dtype=np.uint8)

    mesh = pv.UnstructuredGrid(faces, cell_types, points)

    return mesh

def plot_interp_cell_3d(points: Tensor, cells: Tensor, cell_values: Tensor, *,
    cmap="viridis", show_edges=True, edge_color="black", scalar_bar_title="Cell value", opacity=0.5,
    window_size=(900, 700)):
    """
    Plot a 3D mesh in PyVista where each cell has a scalar value.

    Parameters
    ----------
    points : shape (n_points, 3)
        Mesh vertex coordinates.
    cells : shape (n_cells, 4)
        Cell connectivity. Each cell is a list of point indices.
    cell_values : shape (n_cells)
        Scalar value for each cell.
    cmap : str
        Matplotlib colormap name.
    show_edges : bool
        Whether to draw mesh edges.
    edge_color : str
        Edge color.
    scalar_bar_title : str
        Label for the colorbar.
    window_size : tuple[int, int]
        PyVista window size.
    """

    mesh = create_unstructured_grid(points, cells)

    cell_values = cell_values.cpu().numpy()

    if len(cells) != cell_values.shape[0]:
        raise ValueError("cell_values must have one value per cell")
    mesh.cell_data["values"] = cell_values

    plotter = pv.Plotter(window_size=window_size)
    plotter.add_mesh(
        mesh,
        scalars="values",
        preference="cell",
        cmap=cmap,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color,
        scalar_bar_args={"title": scalar_bar_title},
    )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

def plot_streamlines(points: Tensor, cells: Tensor, velocity: Tensor, *,
    cmap="viridis", show_edges=True, edge_color="black", opacity=0.3,
    window_size=(900, 700), n_points=100):
    """
    Plot 3D streamlines in PyVista for a given velocity field.

    Parameters
    ----------
    points : shape (n_points, 3)
        Mesh vertex coordinates.
    cells : shape (n_cells, 4)
        Cell connectivity. Each cell is a list of point indices.
    velocity : shape (n_cells, 3)
        Velocity vector field.
    cmap : str
        Matplotlib colormap name.
    show_edges : bool
        Whether to draw mesh edges.
    edge_color : str
        Edge color.

    opacity : float
        Opacity of the domain mesh.
    window_size : tuple[int, int]
        PyVista window size.
    n_points : int
        Number of seeding points for streamlines.
    tube_radius : float, optional
        Radius of the streamlines tubes. If None, simple lines are drawn.
    """

    mesh = create_unstructured_grid(points, cells)

    # Load velocity into mesh
    velocity = velocity.cpu().numpy()
    mesh.cell_data["velocity"] = velocity
    mesh_pt = mesh.cell_data_to_point_data()
    mesh_pt.set_active_vectors("velocity")

    # Add streamlines
    seed = pv.Plane(
        center=(-1., 0, 0),  # somewhere on/near inlet
        direction=(1, 0, 0),  # normal direction
        i_size=2, j_size=2, i_resolution=10, j_resolution=10,
    )
    stream = mesh_pt.streamlines_from_source(seed, vectors="velocity", integration_direction="both", max_length=2*mesh_pt.length)
    stream = stream.tube(radius=0.005*mesh_pt.length)

    # Make plot
    plotter = pv.Plotter(window_size=window_size)

    plotter.add_mesh(
        mesh,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color,
        color="white"
    )


    plotter.add_mesh(
        stream,
        scalars="velocity",
        cmap=cmap,
        scalar_bar_args={"title": "Speed"}
    )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
