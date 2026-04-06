#!/bin/bash
# retry_train.sh — Wrapper that auto-restarts training on transient failures.
#
# Layer 2 safety net: if the @retriable decorator in io.py (Layer 1) exhausts
# all its HTTP retries and the training process crashes, this script restarts
# the entire command. OLMo-core's Trainer auto-resumes from the latest
# checkpoint in save_folder, so no extra flags are needed.
#
# Usage:
#   bash retry_train.sh <training_command_and_args...>
#
# Examples:
#   bash retry_train.sh python single_train_launch.py my_run --model_name olmo2_100M_moe_32_16
#   bash retry_train.sh torchrun --nproc-per-node=gpu single_train_launch.py my_run
#
# Environment variables:
#   MAX_RETRIES   — max restart attempts (default: 5)
#   RETRY_DELAY   — seconds to wait between retries (default: 30)

MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_DELAY=${RETRY_DELAY:-30}

if [ $# -eq 0 ]; then
    echo "Usage: bash retry_train.sh <training_command_and_args...>"
    exit 1
fi

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo "=== Attempt $attempt of $MAX_RETRIES ($(date)) ==="
    "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== Training completed successfully ==="
        exit 0
    fi

    echo "=== Training failed with exit code $EXIT_CODE ==="

    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
        echo "Waiting ${RETRY_DELAY}s before retry..."
        sleep "$RETRY_DELAY"
    fi
done

echo "=== All $MAX_RETRIES attempts exhausted, giving up ==="
exit 1
