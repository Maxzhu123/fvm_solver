""" Fused P4 limiting and cell to facet projection. """

import torch
import triton
import triton.language as tl


@triton.autotune(configs=[
    triton.Config({'BLOCK': 64}, num_warps=2),
    triton.Config({'BLOCK': 128}, num_warps=4),
    triton.Config({'BLOCK': 256}, num_warps=4),
    triton.Config({'BLOCK': 256}, num_warps=8),
], key=['NC', 'C', 'D', 'L'])
@triton.jit
def reconstruct_kernel(U, G, Neigh, Disp, Eps, Facets, Sides, F, Limited,
                       NC: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
                       L: tl.constexpr, BLOCK: tl.constexpr):
    """ Limited P4 scheme for cell to facet interpolation.
        Computes dU and one phi_lim per cell and component, then writes
        Us_face = U_cent + dU * phi_lim and limited cell_grads.
        Reuses dU and phi_lim within one kernel.
        NC = n_cells, C = n_comp, D = dim, L = n_local.

        U: Us_cell_facet, shape = [n_cells + n_facets_bc, n_comp]
        G: cell_grads, shape = [n_cells, dim, n_comp]
        Neigh: neigh_combine, shape = [n_cells, n_local]
        Disp: cent_to_facet_disp, shape = [n_cells, n_local, dim, 1]
        Eps: eps_p, shape = [n_cells, 1, 1]
        Facets, Sides: cell_to_facet, cell_facet_signs, shape = [n_cells*n_local]
        F: U_face_all, shape = [n_facets, 2, n_comp]
        Limited: limited cell_grads, shape = [n_cells, dim, n_comp]
    """
    # One lane per cell and component
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < NC * C
    cell, comp = i // C, i % C
    cent = tl.load(U + i, mask=mask, other=0.)    # U_cent, shape = [BLOCK]

    # Select limiting neighbor values, including boundary values
    low, high = cent, cent    # U_min, U_max, shape = [BLOCK]
    for j in tl.static_range(L):
        neighbor = tl.load(Neigh + cell * L + j, mask=mask, other=0)
        value = tl.load(U + neighbor * C + comp, mask=mask, other=0.)
        low = tl.minimum(low, value)
        high = tl.maximum(high, value)

    # Reuse cell gradients for each local facet
    # Unrolled tuples keep values in registers
    gradients = ()
    for d in tl.static_range(D):
        gradients += (tl.load(G + cell * D * C + d * C + comp, mask=mask, other=0.),)
    increments = ()
    eps = tl.load(Eps + cell, mask=mask, other=0.)

    # Cell wide clamping: phi_lim = min(phi, dim=local facets)
    phi_cell = tl.full((BLOCK,), float('inf'), tl.float32)    # phi_lim, shape = [BLOCK]
    for j in tl.static_range(L):
        # Uncorrected update: dU = grad(U) . cent_to_facet_disp
        du = gradients[0] * tl.load(Disp + cell * L * D + j * D, mask=mask, other=0.)
        for d in tl.static_range(1, D):
            du += gradients[d] * tl.load(Disp + cell * L * D + j * D + d, mask=mask, other=0.)
        increments += (du,)

        # numerator = where(dU > 0, U_max, U_min) - U_cent
        delta = tl.where(du > 0., high, low) - cent    # shape = [BLOCK]
        a, b = tl.abs(delta), tl.abs(du)
        b2 = 2. * b
        a2 = a * a
        a_eps = a2 * a2 + eps

        # 4th order limiter, matching SlopeLimiter.p4
        S = b2 * (a2 + b2 * (b2 - a))
        phi = (a_eps + a * S) / (a_eps + b * (delta * delta * delta + S))
        phi = tl.where(a < b2, phi, 1.)
        phi_cell = tl.minimum(phi_cell, phi)

    # Limit cell gradients: cell_grads *= phi_lim
    # Save all components for face gradients and boundary conditions
    for d in tl.static_range(D):
        tl.store(Limited + cell * D * C + d * C + comp, gradients[d] * phi_cell, mask=mask)

    # Project to left and right face values: Us_face = U_cent + dU * phi_lim
    # Each cell writes its own facet side; no atomics needed
    for j in tl.static_range(L):
        facet = tl.load(Facets + cell * L + j, mask=mask, other=0)
        side = tl.load(Sides + cell * L + j, mask=mask, other=0)
        tl.store(F + facet * 2 * C + side * C + comp,
                 cent + increments[j] * phi_cell, mask=mask)


def limit_face_projection(Us_cell_facet, cell_grads, neigh_combine,
                          cent_to_facet_disp, eps_p, cell_to_facet,
                          cell_facet_signs, U_face_all):
    """ Compute P4 face values and limited cell gradients.
        Us_cell_facet.shape = [n_cells + n_facets_bc, n_comp]
        cell_grads.shape = [n_cells, dim, n_comp]
        U_face_all.shape = [n_facets, 2, n_comp], with BC sides already set.
        Writes cell-owned facet sides and returns limited cell_grads.
        Inputs and output use contiguous layouts.
    """
    n_cells, dim, n_comp = cell_grads.shape
    n_local = neigh_combine.shape[1]
    limited = torch.empty_like(cell_grads)
    grid = lambda meta: (triton.cdiv(n_cells * n_comp, meta['BLOCK']),)
    # Each cell-owned side is overwritten once; boundary sides are untouched.
    # Repeated autotuning launches therefore need no output restoration.
    reconstruct_kernel[grid](
        Us_cell_facet, cell_grads, neigh_combine,
        cent_to_facet_disp, eps_p, cell_to_facet, cell_facet_signs,
        U_face_all, limited, n_cells, n_comp, dim, n_local,
    )
    return limited
