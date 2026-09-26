# Triton GEMM Kernels: Naive, Tiled, and Fused Linear + Bias + ReLU

Three GPU matrix-multiplication kernels written in [OpenAI Triton](https://github.com/triton-lang/triton), benchmarked against cuBLAS (`torch.nn.Linear`) and profiled with NVIDIA Nsight Compute on an H200. The kernels are integrated into a PyTorch `nn.Module` and used for inference in a 3-layer MNIST MLP.

## Kernels

| Kernel | File | Design |
|---|---|---|
| Naive | [`gemm_naive.py`](src/gemm_lab/kernels/gemm_naive.py) | Each program computes a 16x16 output tile by iterating over K one element at a time and accumulating outer products of an A column and a B row. No tensor cores. |
| Tiled | [`gemm_tiled.py`](src/gemm_lab/kernels/gemm_tiled.py) | Loads `BLOCK_M x BLOCK_K` and `BLOCK_K x BLOCK_N` tiles and multiplies them with `tl.dot`, which lowers to tensor-core MMA instructions. Accumulates in fp32 with masked loads and stores for boundary tiles. |
| Fused | [`gemm_fused.py`](src/gemm_lab/kernels/gemm_fused.py) | The tiled GEMM with a fused epilogue: bias add and ReLU are applied to the fp32 accumulator in registers before the single store, removing the intermediate write and re-read of the GEMM output. Uses 2-stage software pipelining. |

`FusedLinearReLU` wraps the fused kernel as a drop-in `nn.Module`, and `MyLinear` exposes the naive and tiled kernels through the same interface.

## Results

All results: NVIDIA H200, fp16 inputs, fp32 accumulation. The workload is `relu(x @ W + b)`.

### GEMM throughput (TFLOP/s)

| M = N = K | cuBLAS (`torch.nn.Linear`) | Fused | Tiled | Naive |
|---:|---:|---:|---:|---:|
| 1024 | 100.0 | 74.1 | 51.7 | 4.9 |
| 2048 | 560.8 | 329.3 | 188.5 | 5.3 |
| 4096 | 682.0 | 343.0 | 197.6 | 5.4 |

At 1024³ every path is launch-latency bound (sub-0.05 ms runtimes), so throughput is low across the board. The gap to cuBLAS at larger sizes comes from missing optimizations listed under [Future work](#future-work).

### MNIST MLP inference (784 → 256 → 128 → 10, batch 256)

| Path | Latency | Speedup vs PyTorch |
|---|---:|---:|
| PyTorch (`nn.Linear` + `torch.relu`) | 0.183 ms | 1.00x |
| Unfused custom (`MyLinear` + `torch.relu`) | 0.176 ms | 1.04x |
| Fused hidden layers (`FusedLinearReLU`) | 0.153 ms | **1.19x** |

The output layer uses the unfused linear path since logits take no activation.

### Nsight Compute profile (4096³)

| Metric | Naive | Tiled |
|---|---:|---:|
| Duration | 33.70 ms | 0.78 ms |
| Compute throughput | 89.6% | 24.6% |
| Memory throughput | 80.5% | 91.8% |
| Effective memory bandwidth | 23.1 GB/s | 254.0 GB/s |
| Achieved occupancy | 98.7% | 48.5% |

- **Naive:** high compute utilization and occupancy are misleading. The SMs are saturated with scalar FMAs instead of tensor-core MMAs, and loads are serialized one element at a time across K.
- **Tiled:** memory-bound. `tl.dot` consumes tiles faster than L2 and DRAM refill them. 64 registers per thread cap theoretical occupancy at 50%, limiting latency hiding. The 90.7% L2 hit rate confirms effective tile reuse.

### Tuning experiments (tiled kernel, 4096³)

| Change | Throughput |
|---|---:|
| `num_stages` 1 → 2 (overlap next-tile loads with current `tl.dot`) | 218 → 331 TFLOP/s |
| `BLOCK_M = BLOCK_N` 64 → 128, `BLOCK_K` 32 → 64 | 342 → 454 TFLOP/s |

More pipeline stages beyond 2 give diminishing returns because each stage consumes shared memory and reduces resident blocks per SM. Larger tiles do not help at 1024³: a 128x128 tiling produces 64 blocks, leaving most of the H200's 132 SMs idle.

See [`REPORT.pdf`](REPORT.pdf) for the full write-up.

## Repository layout

```text
src/gemm_lab/
  kernels/
    gemm_naive.py      # scalar-accumulation GEMM
    gemm_tiled.py      # tensor-core tiled GEMM
    gemm_fused.py      # tiled GEMM + bias + ReLU epilogue, FusedLinearReLU module
  ops.py               # gemm() dispatcher
  linear.py            # MyLinear nn.Module
  utils/               # correctness checks, benchmarking helpers
tests/                 # pytest correctness and nn.Module integration tests
scripts/
  bench_gemm.py        # throughput comparison across all paths
  run_mlp_demo.py      # MNIST MLP inference benchmark
  profile_gemm.py      # Nsight Compute profiling
screenshots/           # test, benchmark, and demo output
```

## Usage

Requires an NVIDIA GPU with CUDA, Python 3.10+, PyTorch 2.3+, and Triton 2.3+.

```bash
pip install -r requirements.txt
pip install -e .

make test     # correctness tests against torch.matmul
make bench    # throughput at M=N=K=1024, 2048, 4096
make demo     # MNIST MLP inference (downloads MNIST on first run)
make profile  # Nsight Compute report (requires ncu)
```

Using the fused layer in a model:

```python
import torch
from gemm_lab.kernels import FusedLinearReLU

layer = FusedLinearReLU(784, 256).half()
y = layer(torch.randn(256, 784, device="cuda", dtype=torch.float16))
```

## Future work

- **Grouped (swizzled) program ordering** to improve L2 reuse across neighboring output tiles.
- **Autotuning** `BLOCK_*`, `num_warps`, and `num_stages` per shape with `triton.autotune`.
- **TMA loads and warp specialization** on Hopper to reduce the memory-bound stall observed in the tiled profile.
- **Split-K** for small-M shapes where the output grid underfills the GPU.
