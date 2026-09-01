#!/bin/bash
# Fire the ladder eval for every finished training arm that does not have one yet.
#
#   bash debug/taskscale_lengthmix/fire_evals.sh [max_to_fire]
#
# Per-task eval wiring lives here, in one place, because it is the part that silently produces an
# empty result when it is wrong: the wrong bundle root or a rung label the bundle does not carry
# MISSING-skips every rung and still exits 0.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
L=debug/taskscale_lengthmix/launches
EL=debug/outlier_lengthmix_scaling/launches
MAX=${1:-4}
n=0

cfg () {  # task -> "rungs|bundle-root|extra-flags"
  case $1 in
    oolong)        echo "2k,8k,16k,32k|_eval_bundle_eval500_v2_clean|" ;;
    contradiction) echo "2k,8k,16k,32k|_eval_bundle_eval500_v3|--ladder-version v3" ;;
    xabsence)      echo "4k,8k,16k,32k|taskscale_lengthmix/eval_rungs|" ;;
    grouping)      echo "2k,8k,16k,32k|taskscale_lengthmix/eval_rungs|" ;;
    reorder)       echo "2k,4k,8k,16k|taskscale_lengthmix/eval_rungs|" ;;
    textgroups)    echo "2k,4k,8k,16k|taskscale_lengthmix/eval_rungs|" ;;
    absence)       echo "2k,4k,8k|taskscale_lengthmix/eval_rungs|" ;;
    *) echo ""; ;;
  esac
}

for f in $L/tsl-full-*.log $L/tsl-slm-*.log $L/tsl4-*.log $L/tslS-*.log $L/tslJ-*.log; do
  [ -f "$f" ] || continue
  ex=$(grep -oE 'beaker\.org/ex/[A-Z0-9]{26}' "$f" 2>/dev/null | tail -1 | sed 's|.*/||')
  [ -z "$ex" ] && continue
  st=$(timeout 15 beaker experiment get "$ex" --format json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); s=(d[0]['jobs'][-1] if d[0].get('jobs') else {}).get('status',{}); print(s.get('exitCode') if 'exited' in s else 'live')" 2>/dev/null)
  [ "$st" != "0" ] && continue
  sd=$(timeout 30 beaker experiment logs "$ex" 2>/dev/null | grep -oE "ctc_suite/ckpts/tsl-[A-Za-z0-9-]+-4b-[0-9T-]+" | tail -1 | sed 's|.*/||')
  [ -z "$sd" ] && continue
  run=$(echo "$sd" | sed -E 's/-[0-9]{8}T[0-9]{6}-[0-9]{4}$//')
  ls $EL/ev*-$run.log >/dev/null 2>&1 && continue          # already evaluated
  task=$(echo "$run" | sed -E 's/^tsl-(full|slm)-([a-z]+)-.*/\2/')
  spec=$(cfg "$task"); [ -z "$spec" ] && { echo "SKIP $run (no eval config for task=$task)"; continue; }
  rungs=${spec%%|*}; rest=${spec#*|}; root=${rest%%|*}; extra=${rest#*|}
  # sparse arms cannot run on L40S -- the landmark prefill kernel wants 104KB of shared memory
  case "$run" in *-slm-*) cl=ai2/saturn ;; *) cl=ai2/neptune ;; esac
  echo "FIRE $run task=$task rungs=$rungs cluster=$cl"
  timeout 200 $PY -u debug/outlier_lengthmix_scaling/beaker_native_lengthmix_eval.py \
    "$run" "$sd" --ladder-tasks "$task" --ladder-rungs "$rungs" --cluster "$cl" \
    --eval500-root "$root" $extra > "$EL/ev-$run.log" 2>&1
  echo "   -> $(grep -oE 'beaker\.org/ex/[A-Z0-9]{26}' $EL/ev-$run.log | tail -1 | sed 's|.*/||')"
  n=$((n+1)); [ "$n" -ge "$MAX" ] && break
done
echo "fired $n eval(s)"
