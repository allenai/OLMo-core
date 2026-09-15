#!/bin/bash
# Block until a Beaker experiment finalizes, emitting one line per status change.
#   bash debug/ds64_fast8k/poll_fast8k.sh <experiment-id> [<experiment-id> ...]
# Prints "<id> <state>" whenever a job's state changes, and exits when ALL are finalized.
# NEVER pkill anything; this only reads.
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
declare -A prev
todo=("$@")
while [ ${#todo[@]} -gt 0 ]; do
  left=()
  for ex in "${todo[@]}"; do
    st=$(beaker experiment get "$ex" --format json 2>/dev/null | python -c '
import json,sys
try:
    j=json.load(sys.stdin); j=j[0] if isinstance(j,list) else j
    s=j["jobs"][-1]["status"] if j.get("jobs") else {}
    if s.get("finalized"): print("FINAL rc=%s" % s.get("exitCode"))
    elif s.get("started"): print("RUNNING")
    else: print("QUEUED")
except Exception: print("UNKNOWN")')
    [ "$st" != "${prev[$ex]:-}" ] && { echo "$ex $st"; prev[$ex]="$st"; }
    case "$st" in FINAL*) ;; *) left+=("$ex") ;; esac
  done
  todo=("${left[@]}")
  [ ${#todo[@]} -gt 0 ] && sleep 60
done
echo "ALL FINALIZED"
