from __future__ import annotations

import torch

from gemm_lab.ops import GemmConfig, gemm


def shape_triplets(text: str) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for item in text.split(","):
        m, n, k = item.split("x")
        out.append((int(m), int(n), int(k)))
    return out


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_correctness(shape_list: list[tuple[int, int, int]], dtype: torch.dtype, kernel: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    cfg = GemmConfig(kernel=kernel)  # type: ignore[arg-type]
    for m, n, k in shape_list:
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b).float()
        out = gemm(a, b, cfg=cfg)
        rows.append({"m": m, "n": n, "k": k, "max_abs": max_abs_diff(out, ref)})
    return rows
