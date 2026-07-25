#!/usr/bin/env bash

# Render or submit the canonical GDN2 and KDA sweeps in Cx-first order. The
# existing canonical GDN2 Cx8/1.6e-3 stability run is reused and never emitted
# by this launcher.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}
SUBMIT=0
if [[ ${1:-} == "--submit" ]]; then
  SUBMIT=1
  shift
fi
if (( $# )); then
  echo "usage: $0 [--submit]" >&2
  exit 2
fi

launch_wave() {
  local architecture=$1
  local cx=$2
  shift 2
  local manifest output record experiment
  if [[ ${architecture} == "kda" ]]; then
    manifest="${SCRIPT_DIR}/manifests/275m_kda_reference_lr_sweep.yaml"
  else
    manifest="${SCRIPT_DIR}/manifests/275m_gdn2_canonical_lr_sweep.yaml"
  fi
  output="${SCRIPT_DIR}/generated/275m_${architecture}_canonical_lr_sweep_cx${cx}.yaml"
  record="${SCRIPT_DIR}/generated/275m_canonical_gdn2_kda_lr_sweep_submissions.json"
  experiment="jacobm-275m-${architecture}-canonical-cx${cx}-lr-sweep"
  local command=(
    "${PYTHON}" "${SCRIPT_DIR}/launch_hybrid_scale_smokes.py"
    --manifest "${manifest}"
    --output "${output}"
    --record "${record}"
  )
  local task
  for task in "$@"; do
    command+=(--task "${task}")
  done
  if (( SUBMIT )); then
    command+=(--submit --experiment-name "${experiment}")
  fi
  "${command[@]}"
}

all_lrs=(lr4e-4 lr8e-4 lr1p6e-3 lr3p2e-3)
for cx in 1 2 4 8; do
  tasks=()
  for lr in "${all_lrs[@]}"; do
    tasks+=("cx${cx}-${lr}")
  done
  launch_wave kda "${cx}" "${tasks[@]}"

  if (( cx == 8 )); then
    # Reuse work 01KYBVM2N2D3DM67S8HWARJP6C for the omitted 1.6e-3 cell.
    tasks=(cx8-lr4e-4 cx8-lr8e-4 cx8-lr3p2e-3)
  fi
  launch_wave gdn2 "${cx}" "${tasks[@]}"
done
