import pyvista as pv
from pyvista.core.pointset import UnstructuredGrid
import numpy as np

def plot_interactive(grid: UnstructuredGrid):
    plotter = pv.Plotter()
    plotter.add_mesh_clip_plane(
        grid,
        normal="x",
        show_edges=True,
    )
    plotter.show_bounds()
    plotter.show()


def plot_slice(mesh: UnstructuredGrid):
    centers = mesh.cell_centers().points
    cell_ids = np.where(centers[:, 0] < 0)[0]

    left_cells = mesh.extract_cells(cell_ids)

    left_cells.plot(show_edges=True)


def plot_clip(grid: UnstructuredGrid):
    grid = grid.clip(normal="z", origin=(0, 0, 0.24))
    grid.plot(show_edges=True, show_axes=True)
