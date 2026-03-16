.PHONY: test bench demo profile

test:
	pytest -q

bench:
	python scripts/bench_gemm.py --dtype fp16 --device cuda

demo:
	python scripts/run_mlp_demo.py --dtype fp16 --compile 0

profile:
	python scripts/profile_gemm.py --dtype fp16 --device cuda
