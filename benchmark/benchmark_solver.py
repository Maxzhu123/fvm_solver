from cprint import c_print
import torch
import numpy as np

from base_cfg import ARTEFACT_DIR
from time_fvm.mesh_utils.fvm_mesh import FVMMesh2D
from time_fvm.fvm_equation import FVMEquation, FluidConstitution2D
from time_fvm.config_fvm import ConfigFVM, ConfigEllipse
from time_fvm.time_solvers.integrators import get_solver
from time_fvm.run_fvm import generate_mesh, init_conds_ellipses, init_conds_nozzle


def main():
    import pickle
    np.random.seed(321)
    torch.manual_seed(321)

    cfg: ConfigFVM = ConfigEllipse()
    phy_setup = FluidConstitution2D(cfg, dim=2)

    new_mesh = False
    cfg.profile = True
    cfg.min_A = 5e-6
    cfg.max_A = 1e-5
    cfg.n_iter = 5000

    if new_mesh:
        c_print(f'Generating new mesh...', "green")
        prob_definition = generate_mesh(cfg)
        Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob_definition
        mesh = FVMMesh2D(Xs, tri_idx, all_edgs, bc_edge_mask, device=cfg.device)
        pickle.dump({'mesh': mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}, open(f"{ARTEFACT_DIR}/fvm_mesh.pkl", "wb"))
    else:
        c_print(f'Loading mesh', "green")
        save_dict = pickle.load(open(f"{ARTEFACT_DIR}/fvm_mesh.pkl", "rb"))
        mesh: FVMMesh2D = save_dict['mesh']
        edge_tag = save_dict['edge_tag']
        bound_edgs = save_dict['bound_edgs']
        c_print(f'Loaded mesh. {mesh.n_cells = }, {mesh.n_facets = }', "green")

    print(f'{mesh.volumes.min() = }')

    # Set up initial conditions.
    if cfg.problem_setup == "ellipse":
        bc_tags, Us_init = init_conds_ellipses(mesh, edge_tag, bound_edgs, phy_setup, cfg)
    elif cfg.problem_setup == "nozzle":
        bc_tags, Us_init = init_conds_nozzle(mesh, edge_tag, bound_edgs, phy_setup, cfg)
    else:
        raise ValueError(f'Unknown mode {cfg.problem_setup}')

    fvm_setup = FVMEquation(cfg, phy_setup, mesh, bc_tags, Us_init=Us_init)
    solver = get_solver(fvm_setup, cfg)
    solver.solve()

if __name__ == "__main__":
    # PYTHONPATH=/home/maccyz/Documents/FVM_solver:/home/maccyz/Documents/FVM_solver/tetgen:/home/maccyz/Documents/FVM_solver/tetgen/src
    print("Running fvm ")
    print()
    main()
