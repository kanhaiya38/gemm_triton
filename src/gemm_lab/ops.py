from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from gemm_lab.kernels.gemm_naive import triton_gemm_naive
from gemm_lab.kernels.gemm_tiled import triton_gemm_tiled

KernelKind = Literal["naive", "tiled"]


@dataclass(frozen=True)
class GemmConfig:
    kernel: KernelKind = "tiled"
    allow_tf32: bool = True


def gemm(a: torch.Tensor, b: torch.Tensor, *, cfg: Optional[GemmConfig] = None) -> torch.Tensor:
    """
    GEMM: (M,K) x (K,N) -> (M,N)

    Contract:
    - CUDA only (for the lab)
    - a and b should be contiguous for the starter code
    - dtype: fp16/bf16/fp32
    """
    cfg = cfg or GemmConfig()

    if not a.is_cuda or not b.is_cuda:
        raise ValueError("This lab GEMM expects CUDA tensors.")

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Expected 2D tensors (M,K) and (K,N).")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")

    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Starter: a and b must be contiguous (students may extend).")

    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32

    if cfg.kernel == "naive":
        return triton_gemm_naive(a, b)
    if cfg.kernel == "tiled":
        return triton_gemm_tiled(a, b)

    raise ValueError(f"Unknown kernel kind: {cfg.kernel}")
