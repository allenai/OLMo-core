#!/usr/bin/env bash
# Poll the 16 CTC-suite eval jobs and emit one line per state CHANGE.
# Emits on every terminal state, not just success: a silent monitor and a crashlooping sweep look
# identical, and this runs unattended overnight.
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
IDS=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/hybridish_sft/sweep_ids.txt
prev=""
while true; do
  cur=$(while read -r tag id; do
    [ -n "$id" ] || continue
    st=$(beaker experiment get "$id" --format json 2>/dev/null | python -c "
import json,sys
try:
    j=json.load(sys.stdin)[0]['jobs']
    if not j: print('queued'); raise SystemExit
    s=j[-1].get('status',{})
    print(('exit'+str(s['exitCode'])) if 'exitCode' in s else ('running' if s.get('started') else 'queued'))
except Exception: print('unknown')" 2>/dev/null)
    echo "$tag $st"
  done < "$IDS")
  diff <(echo "$prev") <(echo "$cur") | grep '^>' | sed 's/^> /[sweep] /'
  prev="$cur"
  done_n=$(echo "$cur" | grep -c 'exit')
  [ "$done_n" -ge 16 ] && { echo "[sweep] ALL 16 TERMINAL"; break; }
  sleep 120
done
