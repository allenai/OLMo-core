#!/bin/bash
# Blocking driver for the fast8k sweep: poll -> eval finished runs -> parity-probe them -> collect.
#
#   nohup bash debug/ds64_fast8k/orchestrate_fast8k.sh > debug/ds64_fast8k/orchestrator.log 2>&1 &
#
# It is a LOOP OVER IDEMPOTENT COMMANDS, not a state machine: `launch_fast8k.py status` refreshes
# state.json from Beaker, and `eval` / `parity` skip anything already launched and DEFER anything
# whose training has not finished (a checkpoint only exists after fit()). So the loop can be killed
# and restarted at any point, and a job that finishes between two ticks is picked up on the next.
#
# Parity is launched as soon as TRAINING finishes, not after the ladder eval: it reads the same
# model-only export and is a separate 1-GPU job, so serialising them would only add queue time.
#
# Never `pkill -f` anything here -- this repo runs several concurrent Beaker drivers and a pattern
# kill takes out the others. Stop it by pid, or let it exit on its own.
set -uo pipefail
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
export PYTHONPATH=src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
L=debug/ds64_fast8k/launch_fast8k.py
INTERVAL="${F8K_POLL_INTERVAL:-600}"     # 10 min
MAX_TICKS="${F8K_MAX_TICKS:-144}"        # 24 h

for tick in $(seq 1 "$MAX_TICKS"); do
  echo "=== tick $tick $(date +%H:%M:%S) ==="
  timeout 900 $PY -u $L status || true
  timeout 1800 $PY -u $L eval 2>&1 | grep -vE "^\[skip\]" || true
  timeout 1800 $PY -u $L parity 2>&1 | grep -vE "^\[skip\]" || true
  # Are we done?  Every run needs train DONE/FAILED, an eval, and at least one parity job, all final.
  if timeout 300 $PY - <<'PYX'
import json, sys
st = json.load(open("debug/ds64_fast8k/state.json"))
runs = st["runs"]
done = True
for name, r in runs.items():
    if r.get("state") not in ("DONE", "FAILED"):
        done = False
    e = st.get("evals", {}).get(name, {})
    if r.get("state") == "DONE" and e.get("state") not in ("DONE", "FAILED"):
        done = False
    ps = [v for v in st.get("parity", {}).values() if v.get("run") == name]
    if r.get("state") == "DONE" and (not ps or any(p.get("state") not in ("DONE", "FAILED") for p in ps)):
        done = False
sys.exit(0 if done else 1)
PYX
  then
    echo "=== ALL FINAL at tick $tick ==="
    break
  fi
  sleep "$INTERVAL"
done
timeout 1800 $PY debug/ds64_fast8k/collect_fast8k.py || true
echo "=== ORCHESTRATOR EXIT $(date +%H:%M:%S) ==="
