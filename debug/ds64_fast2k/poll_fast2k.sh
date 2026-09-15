#!/bin/bash
# Blocking poller: refresh state.json, launch the 2k eval for each finished run, stop when every
# run has a finished eval.  bash debug/ds64_fast2k/poll_fast2k.sh [max_cycles] [sleep_s]
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
N=${1:-200}; S=${2:-120}
for i in $(seq 1 "$N"); do
  echo "=== cycle $i $(date +%T) ==="
  $PY debug/ds64_fast2k/launch_fast2k.py status
  DONE=$($PY - <<'EOF'
import json
s=json.load(open('debug/ds64_fast2k/state.json'))
print(",".join(n for n,r in s['runs'].items() if r.get('state')=='DONE' and n not in s['evals']))
EOF
)
  if [ -n "$DONE" ]; then
    echo "--- launching evals: $DONE"
    $PY debug/ds64_fast2k/launch_fast2k.py eval --runs "$DONE"
  fi
  ALL=$($PY - <<'EOF'
import json
s=json.load(open('debug/ds64_fast2k/state.json'))
runs=s['runs']; ev=s['evals']
bad=[n for n,r in runs.items() if r.get('state') not in ('DONE','FAILED')]
bad+= [n for n,r in runs.items() if r.get('state')=='DONE' and ev.get(n,{}).get('state') not in ('DONE','FAILED')]
print("ALLDONE" if not bad else "PENDING %d"%len(bad))
EOF
)
  echo "$ALL"
  [ "$ALL" = "ALLDONE" ] && break
  sleep "$S"
done
