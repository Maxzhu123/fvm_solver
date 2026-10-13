""" Fused normal face gradients, averaging and non-orthogonal correction. """

import triton
import triton.language as tl


@triton.autotune(configs=[
    triton.Config({'BLOCK': 64}, num_warps=2),
    triton.Config({'BLOCK': 128}, num_warps=4),
    triton.Config({'BLOCK': 256}, num_warps=4),
    triton.Config({'BLOCK': 512}, num_warps=8),
], key=['NF', 'C', 'D', 'K'])
@triton.jit
def gradient_kernel(U, G, Vals, Cols, Bias, Cells, XOrth, NHat, GV, GT, GN,
                    NF: tl.constexpr, C: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
                    SVM: tl.constexpr, SVK: tl.constexpr, SCM: tl.constexpr, SCK: tl.constexpr,
                    SB: tl.constexpr, BLOCK: tl.constexpr):
    """ Compute normal gradients, average adjacent cell gradients and correct grad_V.
        Writes grad_V, grad_T_n and normal gradients for all components.
        NF = n_facets, C = n_comp, D = dim, K = ELL entries per row.

        U: Us, shape = [n_cells, n_comp]
        G: limited cell_grads, shape = [n_cells, dim, n_comp]
        Vals, Cols: A_face_grad, shape = [n_facets*n_comp, K]
        Bias: b_face_grad, shape = [n_facets*n_comp]
        Cells: face_cell_indices, shape = [n_facets, 2]
        XOrth, NHat: X_orthog, normals_hat, shape = [n_facets, dim, 1]
        GV: grad_V, shape = [n_facets, dim, dim]
        GT: grad_T_n, shape = [n_facets]
        GN: normal face gradients, shape = [n_facets, n_comp]
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < NF * C
    facet, comp = i // C, i % C

    # Direct face gradient: grad_faces_n = A_face_grad @ Us.flatten() + b_face_grad
    # Preserve all components, including density
    dn = tl.zeros((BLOCK,), tl.float32)
    for kk in range(K):
        value = tl.load(Vals + i * SVM + kk * SVK, mask=mask, other=0.)
        col = tl.load(Cols + i * SCM + kk * SCK, mask=mask, other=0)
        u = tl.load(U + col, mask=mask, other=0.)
        dn += value * u
    dn += tl.load(Bias + i * SB, mask=mask, other=0.)
    tl.store(GN + i, dn, mask=mask)

    # Average limited cell gradients: grad_F_lstsq = 0.5 * (grad_L + grad_R)
    # Boundary facets use the adjacent cell on both sides
    grad_mask = mask & (comp != D)
    left = tl.load(Cells + 2 * facet, mask=grad_mask, other=0)
    right = tl.load(Cells + 2 * facet + 1, mask=grad_mask, other=0)
    averages, normals = (), ()
    orth, dot_n = tl.zeros((BLOCK,), tl.float32), tl.zeros((BLOCK,), tl.float32)
    for d in tl.static_range(D):
        gl = tl.load(G + left * D * C + d * C + comp, mask=grad_mask, other=0.)
        gr = tl.load(G + right * D * C + d * C + comp, mask=grad_mask, other=0.)
        average = .5 * (gl + gr)
        x = tl.load(XOrth + facet * D + d, mask=grad_mask, other=0.)
        n = tl.load(NHat + facet * D + d, mask=grad_mask, other=0.)
        averages += (average,)
        normals += (n,)
        orth += average * x    # grad_F_lstsq . X_orthog
        dot_n += average * n   # grad_F_lstsq . normals_hat

    # Correct normal derivative: dFdn_correct = grad_faces_n - grad_F_lstsq . X_orthog
    correction = dn - orth
    # Replace the normal part, retaining the tangential part:
    # grad_F = grad_F_lstsq + (dFdn_correct - grad_F_lstsq . normals_hat) * normals_hat
    for d in tl.static_range(D):
        value = averages[d] + (correction - dot_n) * normals[d]
        tl.store(GV + facet * D * D + d * D + comp, value, mask=mask & (comp < D))
    # Temperature normal derivative for conduction: grad_T_n = dFdn_correct[T]
    tl.store(GT + facet, correction, mask=mask & (comp == D + 1))


def compute_face_gradients(mesh, Us, cell_grads):
    """ Compute corrected face gradients from limited cell gradients.
        Us.shape = [n_cells, n_comp], cell_grads.shape = [n_cells, dim, n_comp].
        Returns grad_V [n_facets, dim, dim], grad_T_n [n_facets] and
        grad_faces_n [n_facets, n_comp]. Inputs use contiguous geometry layouts.
    """
    _, dim, n_comp = cell_grads.shape
    n_facets = mesh.face_cell_indices.shape[0]
    op = mesh.A_face_grad
    grad_V = Us.new_empty((n_facets, dim, dim))
    grad_T_n = Us.new_empty((n_facets,))
    grad_faces_n = Us.new_empty((n_facets, n_comp))
    grid = lambda meta: (triton.cdiv(n_facets * n_comp, meta['BLOCK']),)
    gradient_kernel[grid](
        Us, cell_grads, op.vals, op.cols, mesh.b_face_grad,
        mesh.face_cell_indices, mesh.X_orthog, mesh.normals_hat,
        grad_V, grad_T_n, grad_faces_n, n_facets, n_comp, dim, op.vals.shape[1],
        op.vals.stride(0), op.vals.stride(1), op.cols.stride(0), op.cols.stride(1),
        mesh.b_face_grad.stride(0),
    )
    return grad_V, grad_T_n, grad_faces_n
