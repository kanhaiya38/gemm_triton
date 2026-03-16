# Triton GEMM Lab

## Repo Tree

```text
triton-gemm-lab/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  Makefile

  src/
    gemm_lab/
      __init__.py
      ops.py
      linear.py
      kernels/
        __init__.py
        gemm_naive.py
        gemm_tiled.py
        gemm_fused.py
      utils/
        __init__.py
        correctness.py
        bench.py

  tests/
    test_gemm_correctness.py
    test_linear_integration.py

  scripts/
    bench_gemm.py
    run_mlp_demo.py
    profile_gemm.py
```

## Student Tasks

Implement and improve:
- `src/gemm_lab/kernels/gemm_naive.py`
- `src/gemm_lab/kernels/gemm_tiled.py`
- `src/gemm_lab/kernels/gemm_fused.py`

Skeleton behavior:
- The files above intentionally contain TODO stubs.
- They currently raise `NotImplementedError` until student code is added.

Keep correctness tests passing and report speedups plus profiling screenshots/analysis.

## Fused Path

`src/gemm_lab/kernels/gemm_fused.py` provides:
- `fused_linear_relu(x, weight_t, bias=None, relu=True)`
- `FusedLinearReLU(nn.Module)`

Error behavior:
- If Triton fused constraints are not met, it raises an explicit error.

## MNIST Demo

`run_mlp_demo.py` runs inference only on a 3-layer MLP (`784 -> 256 -> 128 -> 10`):
- torch baseline: `torch.nn.Linear` + `torch.relu`
- unfused custom: `MyLinear` + `torch.relu`
- hidden layers: fused (`linear + bias + relu`)
- output layer: non-fused linear logits

Data behavior:
- Attempts MNIST auto-download.
- Raises an explicit error if MNIST cannot be loaded.

## Commands

```bash
pip install -r requirements.txt
pip install -e .

make test     # Functionality & Build test
make bench    # Compare gemm+bias+relu across torch, fused, tiled, naive paths
make demo     # Run MLP with custom GeMM kernels
make profile  # ncu profiling 
```

## Deliverables

1. Kernel codes: gemm_naive.py, gemm_tiled.py, gemm_fused.py. All must pass correctness tests.
2. Short report (PDF, 1-2 pages). Include:
  - Screenshot of successful pytest
  - Screenshot of bench and demo results (Run experiments for M=N=K=1024, 2048, 4096)
  - Brief explanation of each kernel design
  - Brief profiling analysis (Memory BW, SM utilization) 
  - Any optimization attempts or improvement ideas

Download the generated NCU profiling report to your local machine.
To veiw the report in a GUI, install Nsight Compute on your laptop:
https://developer.nvidia.com/tools-overview/nsight-compute/get-started

