import torch
import time
from base_cfg import ARTEFACT_DIR

@torch.compile()
def run_projection(save_dict):
    Us_face = save_dict["Us_face"]
    U_facet_bc = save_dict["U_facet_bc"]
    cell_to_facet = save_dict["cell_to_facet"]
    cell_facet_signs = save_dict["cell_facet_signs"]
    bc_locations = save_dict["bc_locations"]
    bc_facet_side = save_dict["bc_facet_side"]

    n_comp = Us_face.shape[-1]

    # Code to benchmark
    U_face_all = torch.empty((664134, 2, n_comp), device="cuda")  # shape = [n_facets, 2, n_comp]
    U_face_all[cell_to_facet, cell_facet_signs] = Us_face
    U_face_all[bc_locations, bc_facet_side] = U_facet_bc

    # # Decompose components back
    # dim = 2
    # Vs_facet = U_face_all[:, :, :dim].contiguous()  # shape = [n_facets, facets=2, n_comp=dim]
    # rho_facet = U_face_all[:, :, dim].unsqueeze(-1).contiguous()  # shape = [n_facets, facets=2, dims=1]
    # T_facet = U_face_all[:, :, dim + 1].unsqueeze(-1).contiguous() # shape = [n_facets, facets=2, dims=1]
    U_face_mean = U_face_all.mean(dim=1)

    return U_face_mean
    # return U_face_all

def benchmark():
    # Load save variables
    save_dict = torch.load(f"{ARTEFACT_DIR}/facet_debug.pt")
    save_dict["cell_facet_signs"] = save_dict["cell_facet_signs"]
    save_dict["bc_facet_side"] = save_dict["bc_facet_side"]
    save_dict["cell_to_facet"] = save_dict["cell_to_facet"].int()
    save_dict["bc_locations"] = save_dict["bc_locations"].int()

    # Warmup
    warmup = 10
    iters = 1000

    for _ in range(warmup):
        run_projection(save_dict)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iters):
        run_projection(save_dict)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000 / iters
    print(f"avg time: {avg_ms:.4f} ms")

if __name__ == '__main__':
    benchmark()
