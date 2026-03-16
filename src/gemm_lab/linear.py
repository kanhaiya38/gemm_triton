from __future__ import annotations

import torch
from torch import nn

from gemm_lab.ops import GemmConfig, gemm


class MyLinear(nn.Module):
    """Linear layer backed by gemm_lab.ops.gemm."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: str = "cuda",
        kernel: str = "tiled",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        self.cfg = GemmConfig(kernel=kernel)  # type: ignore[arg-type]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1.0 / (self.weight.shape[1] ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features], weight.T: [in_features, out_features]
        y = gemm(x, self.weight.t().contiguous(), cfg=self.cfg).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y
