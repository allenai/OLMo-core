#!/usr/bin/env bash
set -euo pipefail

# Temperature-based node healthchecks for Lambda/SLURM jobs.
#
# This script is intended to be run on *every node* before training starts.
# It will exit non-zero on failure to force the job to fail fast.
#
# Opt-out:
#   export OLMO_SKIP_HEALTHCHECKS=1

if [[ "${OLMO_SKIP_HEALTHCHECKS:-0}" == "1" ]]; then
  echo "[healthchecks][$(hostname)][node=${SLURM_NODEID:-?}] SKIP (OLMO_SKIP_HEALTHCHECKS=1)"
  exit 0
fi

log() {
  echo "[healthchecks][$(hostname)][node=${SLURM_NODEID:-?}] $*"
}

die() {
  echo "[healthchecks][$(hostname)][node=${SLURM_NODEID:-?}][ERROR] $*" 1>&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

log "Starting GPU temperature healthcheck..."

# Check for unusually hot GPUs (can indicate stuck workloads or cooling issues).
# Defaults are conservative to avoid false positives.
#
# Controls:
#   OLMO_SKIP_GPU_TEMP_CHECK=1   -> skip this check
#   OLMO_GPU_TEMP_WARN_C=50      -> warn at/above this temperature (C)
#   OLMO_GPU_TEMP_FAIL_C=55      -> fail at/above this temperature (C)
# NOTE(tylerr): these thresholds might be too strict, but typically temps for idle GPUs are around 30C,
# and we've been seeing random starting temps for idle GPUs in the range of 55-65C. Working theory is
# that these GPUs are getting throttled and slowing down jobs.
if [[ "${OLMO_SKIP_GPU_TEMP_CHECK:-0}" != "1" ]]; then
  have_cmd nvidia-smi || die "nvidia-smi not found in PATH; required for GPU temperature check"

  warn_c="${OLMO_GPU_TEMP_WARN_C:-50}"
  fail_c="${OLMO_GPU_TEMP_FAIL_C:-55}"

  if [[ ! "$warn_c" =~ ^[0-9]+$ ]] || [[ ! "$fail_c" =~ ^[0-9]+$ ]]; then
    die "Invalid temp thresholds: OLMO_GPU_TEMP_WARN_C='${warn_c}', OLMO_GPU_TEMP_FAIL_C='${fail_c}' (expected integers in C)"
  fi
  if [[ "$fail_c" -lt "$warn_c" ]]; then
    die "Invalid temp thresholds: fail (${fail_c}C) < warn (${warn_c}C)"
  fi

  temps="$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "$temps" ]] || die "Unable to query GPU temperatures via nvidia-smi"

  hottest=-1
  hottest_idx="?"
  while IFS=, read -r idx temp; do
    idx="$(echo "${idx:-}" | tr -d '[:space:]')"
    temp="$(echo "${temp:-}" | tr -d '[:space:]')"
    if [[ ! "$temp" =~ ^[0-9]+$ ]]; then
      die "Could not parse temperature for GPU ${idx}: '${temp}'"
    fi
    if [[ "$temp" -gt "$hottest" ]]; then
      hottest="$temp"
      hottest_idx="$idx"
    fi
    if [[ "$temp" -ge "$fail_c" ]]; then
      die "GPU ${idx} temperature is ${temp}C (>= ${fail_c}C fail threshold) on node ${SLURM_NODEID:-?}"
    fi
    if [[ "$temp" -ge "$warn_c" ]]; then
      log "WARNING: GPU ${idx} temperature is ${temp}C (>= ${warn_c}C warn threshold) on node ${SLURM_NODEID:-?}"
    fi
  done <<<"$temps"

  [[ "$hottest" -ge 0 ]] || die "No GPU temperatures returned by nvidia-smi"
  log "GPU temperature check complete; hottest GPU=${hottest_idx} at ${hottest}C (warn=${warn_c}C fail=${fail_c}C)"
else
  log "Skipping GPU temperature check (OLMO_SKIP_GPU_TEMP_CHECK=1)"
fi

log "Healthcheck passed."
