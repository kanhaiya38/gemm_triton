from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--relu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=str, default="profiles/gemm_profile")
    args = parser.parse_args()

    ncu = shutil.which("ncu")
    if ncu is None:
        raise SystemExit("ncu (Nsight Compute) not found in PATH")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ncu,
        "--set",
        "full",
        "-f",
        "--target-processes",
        "all",
        "--export",
        str(out),
        sys.executable,
        "scripts/bench_gemm.py",
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--relu" if args.relu else "--no-relu",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--cases",
        "naive_gemm+bias+relu",
        "tiled_gemm+bias+relu",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Profile written to {out}")


if __name__ == "__main__":
    main()
