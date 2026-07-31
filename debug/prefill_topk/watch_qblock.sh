#!/bin/bash
# Poll the two qblock accuracy jobs; print per-rung f1 as each finishes.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
IDS="01KYV50JVC1H9BB7R8R1F8APEP:qblock10pct 01KYV524RSD79RE7MT0CC3KCHY:qblock25pct 01KYV6H3W6DAHHMMVMYSFP9ACV:union10pct"
REMAIN="$IDS"
while [ -n "$REMAIN" ]; do
  NEXT=""
  for pair in $REMAIN; do
    id="${pair%%:*}"; name="${pair##*:}"
    st=$(beaker experiment get "$id" --format json 2>/dev/null | python -c "
import json,sys
try:
    e=json.load(sys.stdin)[0]; s=e['jobs'][-1].get('status',{})
    print('TERMINAL' if any(k in s for k in ('failed','canceled','finalized','exitCode')) else 'running')
except Exception: print('running')
" 2>/dev/null)
    if [ "$st" = "TERMINAL" ]; then
      echo "[$name] finished"
      beaker experiment logs "$id" 2>/dev/null | grep -oE "ladder:contradiction@[0-9]+k\] f1=[0-9.]+|Traceback|Error|rc=[0-9]+" | tail -8
    else
      NEXT="$NEXT $pair"
    fi
  done
  REMAIN="$(echo $NEXT)"
  [ -n "$REMAIN" ] && sleep 180
done
echo "BOTH QBLOCK JOBS DONE"
