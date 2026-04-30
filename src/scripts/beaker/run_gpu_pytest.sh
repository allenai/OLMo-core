#!/bin/bash
set -eo pipefail

echo "Running GPU tests on Beaker..."

uv run python src/scripts/beaker/launch_test.py \
    --budget=ai2/oe-base \
    --num_gpus=1 \
    -- pytest -v -s "$@"
