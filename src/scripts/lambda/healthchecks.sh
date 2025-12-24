#!/usr/bin/env bash
set -euo pipefail

# B200 node healthchecks for Lambda/SLURM jobs.
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

log "Starting GPU healthchecks..."

have_cmd nvidia-smi || die "nvidia-smi not found in PATH; GPU drivers/tooling missing"

# Basic GPU inventory / driver info.
driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]' || true)"
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
gpu_count="$(echo "$gpu_names" | sed '/^\s*$/d' | wc -l | tr -d '[:space:]')"

[[ -n "$driver_ver" ]] || die "Unable to read NVIDIA driver version via nvidia-smi"
[[ "$gpu_count" =~ ^[0-9]+$ ]] || die "Unable to parse GPU count"
[[ "$gpu_count" -gt 0 ]] || die "No GPUs detected by nvidia-smi"

log "Driver: ${driver_ver}; GPUs detected: ${gpu_count}"

# Check for unusually hot GPUs (can indicate stuck workloads or cooling issues).
# Defaults are conservative to avoid false positives.
#
# Controls:
#   OLMO_SKIP_GPU_TEMP_CHECK=1   -> skip this check
#   OLMO_GPU_TEMP_WARN_C=65      -> warn at/above this temperature (C)
#   OLMO_GPU_TEMP_FAIL_C=70      -> fail at/above this temperature (C)
if [[ "${OLMO_SKIP_GPU_TEMP_CHECK:-0}" != "1" ]]; then
  warn_c="${OLMO_GPU_TEMP_WARN_C:-65}"
  fail_c="${OLMO_GPU_TEMP_FAIL_C:-70}"

  if [[ ! "$warn_c" =~ ^[0-9]+$ ]] || [[ ! "$fail_c" =~ ^[0-9]+$ ]]; then
    die "Invalid temp thresholds: OLMO_GPU_TEMP_WARN_C='${warn_c}', OLMO_GPU_TEMP_FAIL_C='${fail_c}' (expected integers in C)"
  fi
  if [[ "$fail_c" -lt "$warn_c" ]]; then
    die "Invalid temp thresholds: fail (${fail_c}C) < warn (${warn_c}C)"
  fi

  temps="$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$temps" ]]; then
    log "WARNING: Unable to query GPU temperatures via nvidia-smi; skipping temp check"
  else
    hottest=-1
    hottest_idx="?"
    while IFS=, read -r idx temp; do
      idx="$(echo "${idx:-}" | tr -d '[:space:]')"
      temp="$(echo "${temp:-}" | tr -d '[:space:]')"
      if [[ ! "$temp" =~ ^[0-9]+$ ]]; then
        log "WARNING: Could not parse temperature for GPU ${idx}: '${temp}'"
        continue
      fi
      if [[ "$temp" -gt "$hottest" ]]; then
        hottest="$temp"
        hottest_idx="$idx"
      fi
      if [[ "$temp" -ge "$fail_c" ]]; then
        die "GPU ${idx} temperature is ${temp}C (>= ${fail_c}C fail threshold)"
      fi
      if [[ "$temp" -ge "$warn_c" ]]; then
        log "WARNING: GPU ${idx} temperature is ${temp}C (>= ${warn_c}C warn threshold)"
      fi
    done <<<"$temps"
    if [[ "$hottest" -ge 0 ]]; then
      log "GPU temperature check complete; hottest GPU=${hottest_idx} at ${hottest}C (warn=${warn_c}C fail=${fail_c}C)"
    fi
  fi
else
  log "Skipping GPU temperature check (OLMO_SKIP_GPU_TEMP_CHECK=1)"
fi

# Quick check for suspicious pre-existing allocations.
if have_cmd nvidia-smi; then
  mem_lines="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$mem_lines" ]]; then
    # Warn if any GPU already has >2GiB used before we start (often means stuck processes).
    while IFS=, read -r idx used total; do
      idx="$(echo "${idx:-}" | tr -d '[:space:]')"
      used="$(echo "${used:-}" | tr -d '[:space:]')"
      total="$(echo "${total:-}" | tr -d '[:space:]')"
      if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$used" -gt 2048 ]]; then
        log "WARNING: GPU ${idx} already using ${used} MiB / ${total} MiB before start"
      fi
    done <<<"$mem_lines"
  fi
fi

log "All healthchecks passed."
