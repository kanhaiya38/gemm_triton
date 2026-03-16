from __future__ import annotations

import pytest
import torch

from gemm_lab.ops import GemmConfig, gemm


HAS_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


pytestmark = pytest.mark.skipif(not (HAS_CUDA and HAS_TRITON), reason="CUDA+Triton required")


@pytest.mark.parametrize("kernel", ["naive", "tiled"])
@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 192, 160), (64, 512, 128)])
def test_gemm_matches_torch(kernel: str, shape: tuple[int, int, int]) -> None:
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    out = gemm(a, b, cfg=GemmConfig(kernel=kernel))
    ref = torch.matmul(a, b).float()

    err = (out - ref).abs().max().item()
    assert err < 2e-1, f"kernel={kernel} max_abs={err}"
