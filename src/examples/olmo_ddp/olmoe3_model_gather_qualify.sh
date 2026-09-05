#!/usr/bin/env bash
# Disposable two-GPU prototype only. Never loads/writes a training checkpoint.
set -euo pipefail
export OMP_NUM_THREADS=1
python src/examples/olmo_ddp/olmoe3_profile_topology.py /results/topology --gpus 2
python -m pytest -q src/test/examples/model_gather_bench_test.py -k nccl
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  src/examples/olmo_ddp/olmoe3_model_gather_bench.py
