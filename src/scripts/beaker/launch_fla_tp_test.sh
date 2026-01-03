#!/usr/bin/env bash

# Launch FLA TP (GPU) tests on Beaker using Gantry
# This script mirrors the test configuration from .github/workflows/main.yml

set -euo pipefail

# Configuration
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/OLMo-core}"
BEAKER_IMAGE="tylerr/olmo-core-tch291cu128-2025-11-25"
GPUS=2
BUDGET="ai2/oe-base"
PRIORITY="high"
TASK_TIMEOUT="8m"

# Check if gantry is installed
if ! command -v gantry &> /dev/null; then
    echo "Error: gantry is not installed. Install it with: uv tool install 'beaker-gantry>=3.1,<4.0'"
    exit 1
fi

echo "Launching FLA TP test on Beaker..."
echo "  Workspace: ${BEAKER_WORKSPACE}"
echo "  Image: ${BEAKER_IMAGE}"
echo "  GPUs: ${GPUS}"

# Launch the test using gantry
gantry run \
    --show-logs \
    --yes \
    --workspace "${BEAKER_WORKSPACE}" \
    --description "OLMo-core Test FLA TP (GPU)" \
    --beaker-image "${BEAKER_IMAGE}" \
    --budget "${BUDGET}" \
    --priority "${PRIORITY}" \
    --preemptible \
    --gpus "${GPUS}" \
    --task-timeout "${TASK_TIMEOUT}" \
    --host-networking \
    --gpu-type h100 \
    --system-python \
    --env 'TOKENIZERS_PARALLELISM=false' \
    --env 'CUBLAS_WORKSPACE_CONFIG=:16:8' \
    --env 'PYTHONPATH=./src/' \
    -- pytest -v --color=yes --durations=3 -m gpu \
        src/test/nn/transformer/model_test.py -k "test_tensor_parallel_fla_transformer"
