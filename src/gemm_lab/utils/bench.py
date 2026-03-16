from __future__ import annotations

import time
from collections.abc import Callable

import torch


def tflops(m: int, n: int, k: int, seconds: float) -> float:
    return (2.0 * m * n * k) / (seconds * 1e12)


def bench_once(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], a: torch.Tensor, b: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn(a, b)
    if a.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(a, b)
    if a.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters
