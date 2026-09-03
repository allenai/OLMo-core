#!/bin/bash
# Overnight watchdog: start the model-scale ladder orchestrators as soon as each repaired base
# lands on weka (the prep jobs print PREP_DONE). Idempotent; safe to rerun.
#   FIX_JOB=<ex> NINE_JOB=<ex> setsid nohup bash debug/flop_scaling/start_scale_when_ready.sh >> debug/flop_scaling/start_scale.log 2>&1 &
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core; cd $REPO
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
FIX_JOB="${FIX_JOB:?}"; NINE_JOB="${NINE_JOB:?}"
CL="ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale,ai2/saturn-cirrascale"
done_fix=0; done_nine=0
start_scale() { # $1 = scale
  tag="s$(echo $1 | tr -d .)"
  if ps -eo args | grep "[o]rchestrate_scale.py" | grep -q "FS_SCALE=$1 " || [ -f debug/flop_scaling/orchestrate_${tag}_state.json ]; then echo "$(date '+%m-%d %H:%M') $1 already started"; return; fi
  FS_SCALE=$1 FS35_CLUSTER="$CL" setsid nohup env FS_SCALE=$1 $PY debug/flop_scaling/orchestrate_scale.py >> debug/flop_scaling/orchestrate_${tag}.log 2>&1 < /dev/null &
  echo "$(date '+%m-%d %H:%M') started orchestrate_scale for $1"
}
while [ $done_fix = 0 ] || [ $done_nine = 0 ]; do
  if [ $done_fix = 0 ]; then
    L=$(beaker experiment logs $FIX_JOB 2>/dev/null)
    if echo "$L" | grep -q PREP_DONE; then done_fix=1; start_scale 0.8b; start_scale 2b
    elif echo "$L" | grep -qE "!!!|Traceback"; then echo "$(date '+%m-%d %H:%M') fix job FAILED"; done_fix=1; fi
  fi
  if [ $done_nine = 0 ]; then
    L=$(beaker experiment logs $NINE_JOB 2>/dev/null)
    if echo "$L" | grep -q PREP_DONE; then done_nine=1; start_scale 9b
    elif echo "$L" | grep -qE "!!!|Traceback|download failed"; then echo "$(date '+%m-%d %H:%M') 9b job FAILED"; done_nine=1; fi
  fi
  sleep 180
done
echo "$(date '+%m-%d %H:%M') watchdog exit"
