from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from gemm_lab.kernels.gemm_fused import fused_linear_relu
from gemm_lab.ops import GemmConfig, gemm
from gemm_lab.utils.bench import bench_once, tflops


def _build_bench_fns(
    bias: torch.Tensor,
    relu: bool,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    def post_op(out: torch.Tensor) -> torch.Tensor:
        out = out + bias
        return torch.relu(out) if relu else out

    def torch_linear(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, weight_t.t(), bias)
        return torch.relu(out) if relu else out

    return {
        "torch.linear": torch_linear,
        "fused_linear": lambda x, weight_t: fused_linear_relu(x, weight_t, bias, relu=relu),
        "tiled_gemm+bias+relu": lambda x, weight_t: post_op(gemm(
            x,
            weight_t,
            cfg=GemmConfig(kernel="tiled"),
        )),
        "naive_gemm+bias+relu": lambda x, weight_t: post_op(gemm(
            x,
            weight_t,
            cfg=GemmConfig(kernel="naive"),
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--relu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["torch.linear", "fused_linear", "tiled_gemm+bias+relu", "naive_gemm+bias+relu"],
        default=["torch.linear", "fused_linear", "tiled_gemm+bias+relu", "naive_gemm+bias+relu"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    a = torch.randn((args.m, args.k), device=args.device, dtype=dtype)
    b = torch.randn((args.k, args.n), device=args.device, dtype=dtype)
    bias = torch.randn((args.n,), device=args.device, dtype=dtype)

    bench_fns = _build_bench_fns(bias, args.relu)

    print(
        f"workload=gemm+bias{'+relu' if args.relu else ''} shape={args.m}x{args.n}x{args.k} "
        f"dtype={args.dtype} device={args.device} relu={args.relu}"
    )
    for name in args.cases:
        fn = bench_fns[name]
        sec = bench_once(fn, a, b, warmup=args.warmup, iters=args.iters)
        print(f"{name:12s} {tflops(args.m, args.n, args.k, sec):8.2f} TFLOP/s  ({sec*1e3:.3f} ms)")


if __name__ == "__main__":
    main()
