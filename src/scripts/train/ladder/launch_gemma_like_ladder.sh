#!/usr/bin/env bash
#
# Launch all Gemma-like v2 ladder rungs up to 8B on Beaker.
#
# Usage:
#   bash src/scripts/train/ladder/launch_gemma_like_ladder.sh
#   TAG=exp1 bash src/scripts/train/ladder/launch_gemma_like_ladder.sh
#
# Environment variables:
#   TAG    Optional tag inserted after the prefix (e.g. TAG=exp1 → gl-olmo-v2-exp1-260m)
#
# Each rung is launched asynchronously (--launch.follow=false) so this script
# does not block waiting for jobs to finish. Monitor jobs via the Beaker UI or:
#   gantry follow <experiment-id>

set -euo pipefail

SCRIPT="src/scripts/train/ladder/gemma_like_ladder.py"
CLUSTER="ai2/jupiter"
PRIORITY="urgent"
PREFIX="gl-olmo-v2"
CHINCHILLA_MULTIPLE="4.0"
TAG="${TAG:-}"

# Format chinchilla multiple for run name (e.g. 4.0 → 4xC)
CHINCHILLA_SUFFIX="$(echo "${CHINCHILLA_MULTIPLE}" | sed 's/\.0$//')xC"

SIZES=(
    # 260m
    # 709m
    # 1p3b
    # 2b
    4b
    # 8b
)

for size in "${SIZES[@]}"; do
    if [[ -n "${TAG}" ]]; then
        run_name="${PREFIX}-${TAG}-${size}-${CHINCHILLA_SUFFIX}"
    else
        run_name="${PREFIX}-${size}-${CHINCHILLA_SUFFIX}"
    fi
    echo "=== Launching ${run_name} ==="
    python "${SCRIPT}" launch "${run_name}" "${CLUSTER}" \
        --launch.priority="${PRIORITY}" \
        --launch.follow=false \
        --trainer.callbacks.wandb.enabled=true \
        --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    echo ""
done

echo "All rungs launched. Monitor jobs in Beaker or with: gantry follow <experiment-id>"
