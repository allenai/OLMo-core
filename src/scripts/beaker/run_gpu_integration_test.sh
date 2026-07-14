#!/bin/bash
set -eo pipefail

echo "Running 2-GPU integration tests on Beaker..."

uv run python src/scripts/beaker/launch_test.py \
  --num_gpus=2 \
  -- pytest -v -s -m gpu src/integration_tests/test_train_small_model.py::test_train_small_model_fsdp "$@"
