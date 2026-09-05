#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
python src/examples/olmo_ddp/olmoe3_profile_topology.py /results/topology --gpus 2
python -m pytest -q src/test/nn/parallel/swiglu_pairwise_test.py
