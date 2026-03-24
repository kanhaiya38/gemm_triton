from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Expected rank-2 tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Triton GEMM expects CUDA tensors.")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Starter kernel expects contiguous inputs.")
    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match.")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Supported dtypes: fp16, bf16, fp32")
    return a.shape[0], b.shape[1], a.shape[1]


if triton is not None:

    @triton.jit
    def _gemm_kernel_tiled(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        TODO: implement optimized tiled GEMM.

        Suggested optimizations:
        - Keep BLOCK_* as multiples of 16 for tensor-core-friendly shapes.
        - Tune num_warps and num_stages by shape bucket.
        - Ensure masked loads/stores are correct on boundary tiles.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        row_mask = row_offsets < M
        col_mask = col_offsets < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in tl.range(0, K, BLOCK_K):
            tmp = tl.arange(0, BLOCK_K) + k
            a_offsets = row_offsets[:, None] * stride_am + tmp[None, :] * stride_ak
            a = tl.load(a_ptr + a_offsets, mask=row_mask[:, None] & (tmp[None, :] < K))

            b_offsets = col_offsets[None, :] * stride_bn + tmp[:, None] * stride_bk
            b = tl.load(b_ptr + b_offsets, mask=col_mask[None, :] & (tmp[:, None] < K))

            acc = tl.dot(a, b, acc)

        c_offsets = row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn
        c_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def triton_gemm_tiled(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    """
    TODO: launch tiled kernel.
        - Allocate / define output matrix
        - Set grid size
        - You may use _check_inputs function to get M, N, K dimensions
        - Call kernel implelemented above with appropriate parameters
    """

    M, N, K = _check_inputs(a, b)
    c = torch.empty((M, N), dtype=a.dtype, device="cuda")
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _gemm_kernel_tiled[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
