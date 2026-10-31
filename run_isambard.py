import argparse


# Solver imports stay inside the guard: the mesh generator starts a spawned worker that re-imports this file.
if __name__ == "__main__":
    from base_cfg import ARTEFACT_DIR
    from time_fvm.config_fvm import ConfigEllipse
    from time_fvm.run_fvm import main

    parser = argparse.ArgumentParser(description="Run the ellipse FVM problem on a cluster.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-iter", type=int, default=50000)
    parser.add_argument("--save-t", type=float, default=0.1, help="Simulation time between saved frames.")
    parser.add_argument("--new-mesh", action="store_true", help="Regenerate artefacts/fvm_mesh.pkl even if it exists.")
    args = parser.parse_args()

    ARTEFACT_DIR.mkdir(exist_ok=True)
    new_mesh = args.new_mesh or not (ARTEFACT_DIR / "fvm_mesh.pkl").exists()

    # The solver's only NaN check runs when it plots, so plot_t must stay finite.
    cfg = ConfigEllipse(device=args.device, n_iter=args.n_iter, save_t=args.save_t, plot_t=args.save_t)
    main(cfg, new_mesh=new_mesh)
