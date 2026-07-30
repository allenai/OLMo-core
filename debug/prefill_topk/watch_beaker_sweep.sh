#!/bin/bash
# Poll the 4 prefill-top-k Beaker jobs and print one line per job as it reaches a terminal state,
# followed by its per-rung f1s. (Shell here is zsh -- run this via `bash`, not `source`.)
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
IDS="01KYTB8FCFAV4RY3Q71AZFK1DC:baseline 01KYTB9BZ1TV6FDBDFK1JSD23C:topk10pct 01KYTBA7JS7KG6R0N1PE05XFRP:topk25pct 01KYTBB3BFT2DNYRJQBGHHWEZQ:topk50pct"
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
except Exception:
    print('running')
" 2>/dev/null)
    if [ "$st" = "TERMINAL" ]; then
      echo "[$name] finished ($id)"
      beaker experiment logs "$id" 2>/dev/null | grep -oE "\[contradiction[_a-z0-9]*\] [a-z0-9_]+=[0-9.]+ \(n=[0-9]+\)|rc=[0-9]+" | tail -8
    else
      NEXT="$NEXT $pair"
    fi
  done
  REMAIN="$(echo $NEXT)"
  [ -n "$REMAIN" ] && sleep 180
done
echo "ALL 4 PREFILL-TOPK JOBS DONE"
