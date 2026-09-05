#!/usr/bin/env bash
# Disposable two-B300 correctness test; no training checkpoints or upload state are touched.
set -euo pipefail
export OMP_NUM_THREADS=1
python src/examples/olmo_ddp/olmoe3_profile_topology.py /results/topology --gpus 2
python -m pytest -q src/test/nn/parallel/grad_add_fast_path_test.py -k nccl
OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE=1 python -m pytest -q \
  src/test/nn/parallel/distributed_test.py -k 'nccl and (fp32_grad_accumulation or no_sync)'
