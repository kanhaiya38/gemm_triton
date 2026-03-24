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
    def _gemm_kernel_naive(
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
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        TODO: implement a naive GEMM kernel.

        Suggested mapping:
        - Use `pid_m = tl.program_id(0)` and `pid_n = tl.program_id(1)`.
        - Map lanes within a BLOCK_SIZE x BLOCK_SIZE output tile.
        - Accumulate over K with scalar loads from A and B.
        - Store results into C with masked writes for boundary tiles.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_offsets = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        row_mask = row_offsets < M
        col_mask = col_offsets < N

        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

        for k in tl.range(0, K):
            a_offsets = row_offsets * stride_am + k * stride_ak
            a = tl.load(a_ptr + a_offsets, mask=row_mask)

            b_offsets = col_offsets * stride_bn + k * stride_bk
            b = tl.load(b_ptr + b_offsets, mask=col_mask)

            acc += a[:, None] * b[None, :]

        c_offsets = row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn
        c_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def triton_gemm_naive(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_size: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    """
    TODO: launch the naive kernel and return output tensor C.
        - Use `_check_inputs(a, b)` to get `M, N, K`
        - Allocate output tensor `c`
        - Define a grid over output tiles
        - Launch `_gemm_kernel_naive`
    """
    M, N, K = _check_inputs(a, b)
    c = torch.empty((M, N), dtype=a.dtype, device="cuda")
    assert c.is_cuda
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    _gemm_kernel_naive[grid](
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
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
