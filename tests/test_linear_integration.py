from __future__ import annotations

import pytest
import torch

from gemm_lab.kernels.gemm_fused import FusedLinearReLU, fused_linear_relu
from gemm_lab.linear import MyLinear


HAS_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


@pytest.mark.skipif(not (HAS_CUDA and HAS_TRITON), reason="CUDA+Triton required")
def test_mylinear_matches_torch_linear() -> None:
    in_features, out_features = 128, 96
    x = torch.randn((64, in_features), device="cuda", dtype=torch.float16)

    ref = torch.nn.Linear(in_features, out_features, bias=True, device="cuda", dtype=torch.float16)
    got = MyLinear(in_features, out_features, bias=True, device="cuda", kernel="tiled").to(dtype=torch.float16)

    with torch.no_grad():
        got.weight.copy_(ref.weight)
        if got.bias is not None and ref.bias is not None:
            got.bias.copy_(ref.bias)

    y_ref = ref(x)
    y_got = got(x)
    assert torch.allclose(y_got, y_ref, atol=2e-1, rtol=2e-1)


@pytest.mark.skipif(not (HAS_CUDA and HAS_TRITON), reason="CUDA+Triton required")
def test_fused_function_matches_reference() -> None:
    x = torch.randn((64, 128), device="cuda", dtype=torch.float16).contiguous()
    w_t = torch.randn((128, 96), device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn((96,), device="cuda", dtype=torch.float16).contiguous()

    y_no_relu = fused_linear_relu(x, w_t, b, relu=False)
    ref_no_relu = torch.addmm(b, x, w_t)
    assert torch.allclose(y_no_relu, ref_no_relu, atol=2e-1, rtol=2e-1)

    y_relu = fused_linear_relu(x, w_t, b, relu=True)
    ref_relu = torch.relu(ref_no_relu)
    assert torch.allclose(y_relu, ref_relu, atol=2e-1, rtol=2e-1)


@pytest.mark.skipif(not (HAS_CUDA and HAS_TRITON), reason="CUDA+Triton required")
def test_fused_module_matches_reference_block() -> None:
    x = torch.randn((32, 128), device="cuda", dtype=torch.float16).contiguous()

    fused = FusedLinearReLU(128, 64, bias=True, relu=True, device="cuda").to(dtype=torch.float16)
    linear = torch.nn.Linear(128, 64, bias=True, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        fused.weight.copy_(linear.weight)
        if fused.bias is not None and linear.bias is not None:
            fused.bias.copy_(linear.bias)

    y_fused = fused(x)
    y_ref = torch.relu(linear(x))
    assert torch.allclose(y_fused, y_ref, atol=2e-1, rtol=2e-1)


def test_fused_on_cpu_raises_error() -> None:
    x = torch.randn((8, 16), dtype=torch.float32)
    w_t = torch.randn((16, 4), dtype=torch.float32)
    b = torch.randn((4,), dtype=torch.float32)

    with pytest.raises((RuntimeError, ValueError), match="Triton|CUDA"):
        _ = fused_linear_relu(x, w_t, b, relu=True)
