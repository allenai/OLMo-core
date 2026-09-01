#!/bin/bash
# Report FINISHED ladder evals that returned fewer rungs than they were asked for.
#
# A rung whose file is absent under EVAL500_ROOT is skipped with a warning and the job still exits
# 0, so a partially-synced bundle produces a short ladder that looks like a complete one. This
# compares --ladder-rungs against the number of `[ladder:...]` result lines, for exited jobs only.
set -uo pipefail
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
EL=debug/outlier_lengthmix_scaling/launches
short=0
for f in $EL/ev*-tsl-*.log; do
  [ -f "$f" ] || continue
  r=$(basename "$f" .log | sed -E 's/^ev[0-9]*-//')
  ex=$(grep -oE 'beaker\.org/ex/[A-Z0-9]{26}' "$f" | tail -1 | sed 's|.*/||')
  [ -z "$ex" ] && continue
  st=$(timeout 15 beaker experiment get "$ex" --format json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); s=(d[0]['jobs'][-1] if d[0].get('jobs') else {}).get('status',{}); print(s.get('exitCode') if 'exited' in s else 'live')" 2>/dev/null)
  [ "$st" != "0" ] && continue                       # only finished jobs can be short
  exp=$(grep -oE '\-\-ladder-rungs [0-9a-zA-Z,]+' "$f" | head -1 | awk '{print $2}' | tr ',' '\n' | wc -l)
  got=$(timeout 20 beaker experiment logs "$ex" 2>/dev/null | grep -cE "\[ladder:.*(f1|score|kendall)")
  if [ "$got" -lt "$exp" ]; then
    echo "SHORT $r  expected=$exp got=$got  ($ex)"
    short=$((short+1))
  fi
done
echo "$short finished eval(s) returned a short ladder"
