#!/usr/bin/env bash

set -euo pipefail

# Source helper functions.
. ./src/scripts/lambda/utils.sh

# Node healthchecks for Lambda/SLURM jobs.
#
# Checks:
# - GPU temperature
# - GPU idle (process + memory)
#
# This script is intended to be run on *every node* before training starts.
# It will exit non-zero on failure to force the job to fail fast.
#
# Opt-out:
#   export OLMO_SKIP_HEALTHCHECKS=1

if [[ "${OLMO_SKIP_HEALTHCHECKS:-0}" == "1" ]]; then
  log_info "SKIP (OLMO_SKIP_HEALTHCHECKS=1)"
  exit 0
fi

log_info "Starting GPU temperature healthcheck..."

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
      die "GPU ${idx} temperature is ${temp}C (>= ${fail_c}C fail threshold) on node $(hostname)"
    fi
    if [[ "$temp" -ge "$warn_c" ]]; then
      log_warning "GPU ${idx} temperature is ${temp}C (>= ${warn_c}C warn threshold) on node $(hostname)"
    fi
  done <<<"$temps"

  [[ "$hottest" -ge 0 ]] || die "No GPU temperatures returned by nvidia-smi"
  log_info "GPU temperature check complete; hottest GPU=${hottest_idx} at ${hottest}C (warn=${warn_c}C fail=${fail_c}C)"
else
  log_info "Skipping GPU temperature check (OLMO_SKIP_GPU_TEMP_CHECK=1)"
fi

# Ensure GPUs are idle: no compute processes, and low/empty memory usage.
#
# Controls:
#   OLMO_SKIP_GPU_IDLE_CHECK=1      -> skip this check entirely
#   OLMO_GPU_MEM_WARN_MIB=200       -> warn at/above this memory.used (MiB)
#   OLMO_GPU_MEM_FAIL_MIB=500       -> fail at/above this memory.used (MiB)
#   OLMO_GPU_PROC_ALLOW_REGEX='...' -> allow listed compute process_name(s) (bash regex)
log_info "Starting GPU idle (process + memory) healthcheck..."
if [[ "${OLMO_SKIP_GPU_IDLE_CHECK:-0}" != "1" ]]; then
  have_cmd nvidia-smi || die "nvidia-smi not found in PATH; required for GPU idle check"

  mem_warn_mib="${OLMO_GPU_MEM_WARN_MIB:-200}"
  mem_fail_mib="${OLMO_GPU_MEM_FAIL_MIB:-500}"
  proc_allow_re="${OLMO_GPU_PROC_ALLOW_REGEX:-}"

  if [[ ! "$mem_warn_mib" =~ ^[0-9]+$ ]] || [[ ! "$mem_fail_mib" =~ ^[0-9]+$ ]]; then
    die "Invalid memory thresholds: OLMO_GPU_MEM_WARN_MIB='${mem_warn_mib}', OLMO_GPU_MEM_FAIL_MIB='${mem_fail_mib}' (expected integers in MiB)"
  fi
  if [[ "$mem_fail_mib" -lt "$mem_warn_mib" ]]; then
    die "Invalid memory thresholds: fail (${mem_fail_mib}MiB) < warn (${mem_warn_mib}MiB)"
  fi

  mems="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "$mems" ]] || die "Unable to query GPU memory usage via nvidia-smi"

  while IFS=, read -r idx mem_used; do
    idx="$(echo "${idx:-}" | tr -d '[:space:]')"
    mem_used="$(echo "${mem_used:-}" | tr -d '[:space:]')"
    if [[ ! "$mem_used" =~ ^[0-9]+$ ]]; then
      die "Could not parse memory.used for GPU ${idx}: '${mem_used}'"
    fi
    if [[ "$mem_used" -ge "$mem_fail_mib" ]]; then
      die "GPU ${idx} memory.used is ${mem_used}MiB (>= ${mem_fail_mib}MiB fail threshold) on node $(hostname)"
    fi
    if [[ "$mem_used" -ge "$mem_warn_mib" ]]; then
      log_warning "GPU ${idx} memory.used is ${mem_used}MiB (>= ${mem_warn_mib}MiB warn threshold) on node $(hostname)"
    fi
  done <<<"$mems"

  procs="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$procs" ]]; then
    bad=0
    while IFS=, read -r gpu_uuid pid pname used_mem; do
      gpu_uuid="$(echo "${gpu_uuid:-}" | tr -d '[:space:]')"
      pid="$(echo "${pid:-}" | tr -d '[:space:]')"
      pname="$(echo "${pname:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
      used_mem="$(echo "${used_mem:-}" | tr -d '[:space:]')"

      if [[ -n "$proc_allow_re" ]] && [[ "$pname" =~ $proc_allow_re ]]; then
        log_info "Allowed GPU compute process found (matches OLMO_GPU_PROC_ALLOW_REGEX): gpu_uuid=${gpu_uuid} pid=${pid} name='${pname}' used_memory=${used_mem}MiB"
        continue
      fi

      log_error "Unexpected GPU compute process found: gpu_uuid=${gpu_uuid} pid=${pid} name='${pname}' used_memory=${used_mem}MiB"
      bad=1
    done <<<"$procs"

    [[ "$bad" == "0" ]] || die "Found running GPU compute processes; GPUs are not idle"
  else
    log_info "No running GPU compute processes found."
  fi
else
  log_info "Skipping GPU idle check (OLMO_SKIP_GPU_IDLE_CHECK=1)"
fi

log_info "Healthcheck passed."
