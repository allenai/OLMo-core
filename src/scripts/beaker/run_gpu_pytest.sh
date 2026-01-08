#!/bin/bash
set -eo pipefail

TEST_PATH="${1:-src/test/nn/hf/golden_tests.py}"

echo "Running GPU tests for: $TEST_PATH"
echo "Launching on Beaker..."

uv run python src/scripts/beaker/launch_test.py \
    --budget=ai2/oe-base \
    --num_gpus=1 \
    --allow_dirty \
    -- pytest -v -s "$TEST_PATH"
