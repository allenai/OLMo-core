#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LADDER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(uv run python)
fi
CACHE_ARGS=()

if [[ "${REFRESH_CACHE:-0}" == "1" ]]; then
  CACHE_ARGS+=(--refresh-cache)
elif [[ "${REFRESH_STALE_CACHE:-1}" == "1" ]]; then
  CACHE_ARGS+=(--refresh-stale-cache)
fi

if [[ "${INCLUDE_RUNNING:-0}" == "1" ]]; then
  CACHE_ARGS+=(--include-running)
fi

"${PYTHON_CMD[@]}" "${LADDER_DIR}/plot_wandb_ladder.py" "${CACHE_ARGS[@]}"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/expert_granularity/plot_expert_granularity.py" "${CACHE_ARGS[@]}"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/total_sparsity/plot_total_sparsity.py" "${CACHE_ARGS[@]}"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/shared_expert/plot_shared_expert.py" "${CACHE_ARGS[@]}"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/dense_schedule/plot_dense_schedule.py" "${CACHE_ARGS[@]}"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/qwen3_like/plot_qwen3_like.py" "${CACHE_ARGS[@]}"
