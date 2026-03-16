from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from gemm_lab.kernels.gemm_fused import FusedLinearReLU
from gemm_lab.linear import MyLinear


class MLPUnfused(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.fc1 = MyLinear(784, 256, device=device)
        self.fc2 = MyLinear(256, 128, device=device)
        self.fc3 = MyLinear(128, 10, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return self.fc3(x)


class MLPFused(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.fc1 = FusedLinearReLU(784, 256, relu=True, device=device)
        self.fc2 = FusedLinearReLU(256, 128, relu=True, device=device)
        self.fc3 = MyLinear(128, 10, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class MLPTorch(nn.Module):
    def __init__(self, device: str, dtype: torch.dtype):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=True, device=device, dtype=dtype)
        self.fc2 = nn.Linear(256, 128, bias=True, device=device, dtype=dtype)
        self.fc3 = nn.Linear(128, 10, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def copy_weights(unfused: MLPUnfused, fused: MLPFused, torch_mlp: MLPTorch) -> None:
    with torch.no_grad():
        fused.fc1.weight.copy_(unfused.fc1.weight)
        if fused.fc1.bias is not None and unfused.fc1.bias is not None:
            fused.fc1.bias.copy_(unfused.fc1.bias)
        torch_mlp.fc1.weight.copy_(unfused.fc1.weight)
        if unfused.fc1.bias is not None:
            torch_mlp.fc1.bias.copy_(unfused.fc1.bias)

        fused.fc2.weight.copy_(unfused.fc2.weight)
        if fused.fc2.bias is not None and unfused.fc2.bias is not None:
            fused.fc2.bias.copy_(unfused.fc2.bias)
        torch_mlp.fc2.weight.copy_(unfused.fc2.weight)
        if unfused.fc2.bias is not None:
            torch_mlp.fc2.bias.copy_(unfused.fc2.bias)

        fused.fc3.weight.copy_(unfused.fc3.weight)
        if fused.fc3.bias is not None and unfused.fc3.bias is not None:
            fused.fc3.bias.copy_(unfused.fc3.bias)
        torch_mlp.fc3.weight.copy_(unfused.fc3.weight)
        if unfused.fc3.bias is not None:
            torch_mlp.fc3.bias.copy_(unfused.fc3.bias)


def _mnist_loader(batch_size: int, num_batches: int, data_dir: str):
    total = batch_size * num_batches
    try:
        from torchvision import datasets, transforms

        ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        x = ds.data[:total].float().unsqueeze(1) / 255.0
        y = ds.targets[:total]
        return x, y
    except Exception as exc:
        raise RuntimeError(f"MNIST load failed: {exc}") from exc


def _bench(model: nn.Module, x: torch.Tensor, batch_size: int, warmup: int, iters: int) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x[:batch_size])
        if x.is_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(iters):
            s = i * batch_size
            e = s + batch_size
            _ = model(x[s:e])
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile", type=int, choices=[0, 1], default=0)
    parser.add_argument("--data-dir", type=str, default="./.data")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    x_cpu, _ = _mnist_loader(args.batch_size, args.iters, args.data_dir)
    x = x_cpu.view(x_cpu.shape[0], -1).to(device=args.device, dtype=dtype).contiguous()

    unfused = MLPUnfused(device=args.device).to(dtype=dtype)
    fused = MLPFused(device=args.device).to(dtype=dtype)
    torch_mlp = MLPTorch(device=args.device, dtype=dtype)
    copy_weights(unfused, fused, torch_mlp)

    if args.compile == 1:
        unfused = torch.compile(unfused)
        fused = torch.compile(fused)
        torch_mlp = torch.compile(torch_mlp)

    torch_ms = _bench(torch_mlp, x, args.batch_size, args.warmup, args.iters) * 1e3
    unfused_ms = _bench(unfused, x, args.batch_size, args.warmup, args.iters) * 1e3
    fused_ms = _bench(fused, x, args.batch_size, args.warmup, args.iters) * 1e3

    fused_vs_unfused = unfused_ms / fused_ms
    fused_vs_torch = torch_ms / fused_ms
    unfused_vs_torch = torch_ms / unfused_ms
    mode = "compile" if args.compile == 1 else "eager"
    print(f"mode={mode} dtype={args.dtype} batch={args.batch_size}")
    print(f"torch_mlp:   {torch_ms:.3f} ms")
    print(f"unfused_mlp: {unfused_ms:.3f} ms")
    print(f"fused_mlp:   {fused_ms:.3f} ms")
    print(f"fused_vs_unfused: {fused_vs_unfused:.3f}x")
    print(f"fused_vs_torch:   {fused_vs_torch:.3f}x")
    print(f"unfused_vs_torch: {unfused_vs_torch:.3f}x")


if __name__ == "__main__":
    main()
