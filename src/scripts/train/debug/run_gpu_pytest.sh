#!/bin/bash
set -eo pipefail

TEST_PATH="${1:-src/test/nn/hf/qwen3_test.py}"

echo "Running GPU tests for: $TEST_PATH"
echo "Launching on Beaker..."

uv run python src/scripts/beaker/launch_test.py \
    --budget=ai2/oe-base \
    --num_gpus=1 \
    -- pytest -v "$TEST_PATH"
