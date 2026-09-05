#!/usr/bin/env bash
# Disposable two-GPU qualification only; no training/checkpoint/uploader state is touched.
set -euo pipefail
export OMP_NUM_THREADS=1
python src/examples/olmo_ddp/olmoe3_profile_topology.py /results/topology
python -m pytest -q src/test/nn/parallel/reduce_scatter_fast_path_test.py -k nccl
OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH=1 python -m pytest -q \
  src/test/nn/parallel/distributed_test.py -k 'reduce_scatter and nccl'
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  src/examples/olmo_ddp/olmoe3_reduce_scatter_bench.py
