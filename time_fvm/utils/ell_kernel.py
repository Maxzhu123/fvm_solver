"""Triton ELL-packed SpMM kernel and Python wrapper."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_B': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_B': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_B': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_B': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_B': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_B': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'B', 'K', 'HAS_BIAS', 'HAS_REF'],
)
@triton.jit
def ell_spmm_kernel(
    vals, cols, x, y, b,
    M: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_REF: tl.constexpr,
    REF_DIV: tl.constexpr,
    stride_vals_m: tl.constexpr,
    stride_vals_k: tl.constexpr,
    stride_cols_m: tl.constexpr,
    stride_cols_k: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_y_m: tl.constexpr,
    stride_y_b: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    bs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    mask_y = (rows[:, None] < M) & (bs[None, :] < B)
    acc = tl.zeros((BLOCK_M, BLOCK_B), tl.float32)

    # For zero-row-sum operators (e.g. the least-squares gradient) evaluate the
    # differences first: y_i = sum_k A_ik * (x[col_ik] - x[owner(i)]).  This is
    # algebraically identical (A @ 1 = 0) but annihilates constants exactly
    # instead of relying on catastrophic cancellation.
    if HAS_REF:
        owner = rows // REF_DIV
        mask_x = (rows[:, None] < M) & (bs[None, :] < B)
        x_ref = tl.load(
            x + owner[:, None] * stride_x_n + bs[None, :] * stride_x_b,
            mask=mask_x,
            other=0.0,
        )

    for kk in range(K):
        a_vals = tl.load(
            vals + rows * stride_vals_m + kk * stride_vals_k,
            mask=rows < M,
            other=0.0,
        )

        a_cols = tl.load(
            cols + rows * stride_cols_m + kk * stride_cols_k,
            mask=rows < M,
            other=0,
        )

        x_vals = tl.load(
            x + a_cols[:, None] * stride_x_n + bs[None, :] * stride_x_b,
            mask=(rows[:, None] < M) & (bs[None, :] < B),
            other=0.0,
        )

        if HAS_REF:
            acc += a_vals[:, None] * (x_vals - x_ref)
        else:
            acc += a_vals[:, None] * x_vals

    if HAS_BIAS:
        b_vals = tl.load(b + rows, mask=rows < M, other=0.0)
        acc += b_vals[:, None]

    tl.store(
        y + rows[:, None] * stride_y_m + bs[None, :] * stride_y_b,
        acc,
        mask=mask_y,
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=4),
    ],
    key=['M', 'K', 'HAS_BIAS', 'HAS_REF'],
)
@triton.jit
def _ell_spmv_kernel(
    vals, cols, x, y, b,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_REF: tl.constexpr,
    REF_DIV: tl.constexpr,
    stride_vals_m: tl.constexpr,
    stride_vals_k: tl.constexpr,
    stride_cols_m: tl.constexpr,
    stride_cols_k: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_y_m: tl.constexpr,
    stride_b_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """SpMV with optional bias. One block of rows per program."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < M

    # Gather x indices and values for all K nonzeros in this block of rows
    acc = tl.zeros((BLOCK_M,), tl.float32)
    if HAS_REF:
        owner = rows // REF_DIV
        x_ref = tl.load(x + owner * stride_x_n, mask=mask, other=0.0)

    for kk in range(K):
        a_val = tl.load(
            vals + rows * stride_vals_m + kk * stride_vals_k,
            mask=mask, other=0.0,
        )
        a_col = tl.load(
            cols + rows * stride_cols_m + kk * stride_cols_k,
            mask=mask, other=0,
        )
        x_val = tl.load(x + a_col * stride_x_n, mask=mask, other=0.0)
        if HAS_REF:
            acc += a_val * (x_val - x_ref)
        else:
            acc += a_val * x_val

    if HAS_BIAS:
        acc += tl.load(b + rows * stride_b_m, mask=mask, other=0.0)

    tl.store(y + rows * stride_y_m, acc, mask=mask)


# Separate tuning cache for caller-owned outputs. Restore y between tuning
# trials because b may alias y when accumulating into an existing buffer.
_ell_spmv_out_kernel = triton.autotune(
    configs=_ell_spmv_kernel.configs,
    key=['M', 'K', 'HAS_BIAS', 'HAS_REF'],
    restore_value=['y'],
)(_ell_spmv_kernel.fn)


def ell_spmv(vals, cols, x, b=None, *, out=None, ref_div=None):
    """
    vals: [M, K], cols: [M, K]
    x:    [N] or [N, 1]
    b:    [M] or None   (fused bias: y = A @ x + b)
    returns y: [M] or [M, 1] (matches x dimensionality)
    out: optional output buffer, overwritten with A @ x + b; may alias b,
         but must not overlap x, vals or cols.
    ref_div: if given, evaluate the differences-first (DIFF) form
         y_i = sum_k A_ik * (x[col_ik] - x[row_i // ref_div]).
         Intended for zero-row-sum operators (e.g. the least-squares
         gradient, ref_div = dim), where it is algebraically identical
         (A @ 1 = 0) but annihilates constants exactly instead of by
         cancellation.

    1D kernel: one block per row chunk, no wasted BLOCK_B dimension.
    Autotuned on (M, K, HAS_BIAS, HAS_REF).
    """
    M, K = vals.shape
    N = x.shape[0]
    # Pass the whole fresh output to Triton. Mutating a column view forces
    # torch.compile to insert unnecessary output-buffer copies.
    y = torch.empty((M,), device=x.device, dtype=x.dtype) if out is None else out
    HAS_BIAS = b is not None
    HAS_REF = ref_div is not None
    REF_DIV = ref_div or 1
    b_ptr = b if HAS_BIAS else x
    stride_b_m = b.stride(0) if HAS_BIAS else 0

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    kernel = _ell_spmv_kernel if out is None else _ell_spmv_out_kernel
    kernel[grid](
        vals, cols, x, y, b_ptr,
        M, N, K, HAS_BIAS, HAS_REF, REF_DIV,
        vals.stride(0), vals.stride(1),
        cols.stride(0), cols.stride(1),
        x.stride(0), y.stride(0), stride_b_m,
    )
    if out is not None:
        return out
    return y if x.ndim == 1 else y.unsqueeze(-1)


def ell_spmm(vals, cols, x, b=None, *, ref_div=None):
    """
    vals: [M, K], cols: [M, K]
    x:    [N, B]
    b:    [M] or None   (fused bias: y = A @ x + b)
    returns y: [M, B]
    ref_div: see ell_spmv. Row i's reference row is i // ref_div.

    Dispatches to ell_spmv when B == 1.
    Block sizes chosen automatically by @triton.autotune.
    """
    M, K = vals.shape
    N, B = x.shape

    if B == 1:
        return ell_spmv(vals, cols, x, b=b, ref_div=ref_div)

    HAS_BIAS = b is not None
    HAS_REF = ref_div is not None
    REF_DIV = ref_div or 1
    y = torch.empty((M, B), device=x.device, dtype=x.dtype)
    b_ptr = b if HAS_BIAS else x

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(B, meta['BLOCK_B']),
    )

    ell_spmm_kernel[grid](
        vals, cols, x, y, b_ptr,
        M, B, K, HAS_BIAS, HAS_REF, REF_DIV,
        vals.stride(0), vals.stride(1),
        cols.stride(0), cols.stride(1),
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    return y


def print_best_config():
    """Print the best config found by the autotuner, if any."""
    try:
        cfg = _ell_spmv_kernel.best_config
        if cfg is not None:
            print(f'Triton config: {cfg}')
        else:
            print("No best config yet — run the kernel first.")
    except Exception:
        print("Error getting best config.")


def csr_to_ell(csr: torch.Tensor, K: int | None = None):
    """Convert a PyTorch sparse CSR matrix to ELL format.

    Returns vals: [M, K], cols: [M, K].
    Padded entries use col=0, val=0.
    """
    assert csr.layout == torch.sparse_csr, "Input must be a sparse CSR tensor"

    crow = csr.crow_indices()
    col = csr.col_indices().to(torch.int32)
    val = csr.values()

    M = csr.shape[0]
    nnz_per_row = crow[1:] - crow[:-1]

    if K is None:
        K = int(nnz_per_row.max().item())

    if torch.any(nnz_per_row > K):
        raise ValueError("Some rows have more nonzeros than K")

    ell_cols = torch.zeros((M, K), device=col.device, dtype=torch.int32)
    ell_vals = torch.zeros((M, K), device=val.device, dtype=val.dtype)

    for k in range(K):
        mask = nnz_per_row > k
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        src_idx = crow[rows] + k
        ell_cols[rows, k] = col[src_idx]
        ell_vals[rows, k] = val[src_idx]

    return ell_vals, ell_cols
