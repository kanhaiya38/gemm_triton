.PHONY: test bench demo profile

test:
	pytest -q

bench:
	python scripts/bench_gemm.py --dtype fp16 --device cuda --m 1024 --n 1024 --k 1024
	python scripts/bench_gemm.py --dtype fp16 --device cuda --m 2048 --n 2048 --k 2048
	python scripts/bench_gemm.py --dtype fp16 --device cuda --m 4096 --n 4096 --k 4096

demo:
	python scripts/run_mlp_demo.py --dtype fp16 --compile 0

profile:
	python scripts/profile_gemm.py --dtype fp16 --device cuda
