#!/usr/bin/env bash
# Wait until a Beaker experiment finalizes (all jobs terminal), then run a launch command.
#
# Usage:
#   launch_after_experiment.sh <experiment_id> <command...>
#
# Example (gate a base-model run on the same-method Qwen3-4B run so they never overlap):
#   src/scripts/train/Qwen3/launch_after_experiment.sh 01KTEJ... \
#     python src/scripts/train/Qwen3/Qwen3-4B-base-dense-dolma3longmino.py \
#     launch q4b-base-dense-dolma3longmino ai2/jupiter-cirrascale-2
#
# Notes:
#   * Fires on *completion* (finalized), regardless of the counterpart's exit code.
#   * Polls every POLL_SECONDS (default 120). Requires the `beaker` CLI to be authenticated.
set -uo pipefail

EXP="${1:?usage: launch_after_experiment.sh <experiment_id> <command...>}"
shift
POLL_SECONDS="${POLL_SECONDS:-120}"
PY="${PYTHON_BIN:-python3}"

echo "[gate] waiting for beaker experiment ${EXP} to finalize, then: $*"
while true; do
  state=$(beaker experiment get "${EXP}" --format json 2>/dev/null | "${PY}" -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("UNKNOWN"); sys.exit(0)
j = d[0] if isinstance(d, list) else d
jobs = j.get("jobs", [])
if not jobs:
    print("PENDING"); sys.exit(0)
# Finalized when every job has a "finalized" timestamp.
if all(job.get("status", {}).get("finalized") for job in jobs):
    codes = [job.get("status", {}).get("exitCode") for job in jobs]
    print("DONE exit=%s" % ",".join(str(c) for c in codes))
else:
    print("RUNNING")
')
  case "${state}" in
    DONE*) echo "[gate] ${EXP} finalized (${state#DONE }); launching now."; break ;;
    *)     echo "[gate] ${EXP}: ${state}; sleeping ${POLL_SECONDS}s"; sleep "${POLL_SECONDS}" ;;
  esac
done

exec "$@"
