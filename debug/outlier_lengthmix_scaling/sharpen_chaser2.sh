#!/bin/bash
# Sharpen chaser v2: ALL evals on Beaker (user directive: no mooney). Train DONE -> weka-native eval.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
D=$REPO/debug/outlier_lengthmix_scaling
STATE=$D/chaser_state; LOGD=$D/launches
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"; export PYTHONPATH="$REPO/src"
JOBS="
lmx-full-q32k16000-qd-4b|qdmatch_nq|32k,64k
lmx-full-qmixs320M-qd-4b|qdmatch_nq|2k,8k,16k,32k
lmx-slm-qmixs320M-qd-4b|qdmatch_nq|2k,8k,16k,32k
lmx-full-nmixs16M-nq-4b|nq|2k,8k,16k,32k
lmx-full-nmixs32M-nq-4b|nq|2k,8k,16k,32k
lmx-full-nmixs48M-nq-4b|nq|2k,8k,16k,32k
lmx-slm-nmixs48M-nq-4b|nq|2k,8k,16k,32k
lmx-full-nqD32k4000-4b|nq|32k,64k
lmx-full-nqD64k2000-4b|nq|32k,64k
"
exid () { grep -Eo 'beaker.org/ex/[A-Z0-9]+' "$LOGD/$1.log" 2>/dev/null | head -1 | sed 's|.*/||'; }
state () {
  beaker experiment get "$1" --format json 2>/dev/null | python3 -c "
import json,sys
try: e=json.load(sys.stdin)[0]
except Exception: print('UNKNOWN'); raise SystemExit
st=(e.get('jobs') or [{}])[-1].get('status',{})
print('CANCELED' if st.get('canceled') else ('DONE:'+str(st.get('exitCode'))) if st.get('finalized') else 'RUNNING' if st.get('started') else 'SCHEDULED')"
}
savedir () { beaker experiment spec "$2" 2>/dev/null | grep -Eo "${1}-[0-9]{8}T[0-9]{6}-[0-9]{4}" | head -1; }
fire_native () {
  local RUN=$1 LT=$2 RUNGS=$3 EX=$4 SD
  SD=$(savedir "$RUN" "$EX"); [ -z "$SD" ] && return 1
  timeout 300 $PY debug/outlier_lengthmix_scaling/beaker_native_lengthmix_eval.py \
    "$RUN" "$SD" --ladder-tasks "$LT" --ladder-rungs "$RUNGS" > "$LOGD/nev-$RUN.log" 2>&1
  grep -q 'submitted:' "$LOGD/nev-$RUN.log"
}
for i in $(seq 1 200); do
  echo "$JOBS" | while IFS='|' read -r RUN LT RUNGS; do
    [ -z "$RUN" ] && continue
    SF="$STATE/$RUN.shp2.state"
    ST=$(cat "$SF" 2>/dev/null || echo TRAIN)
    case "$ST" in
      EVAL*|GAVE_UP*) continue ;;
      TRAIN)
        EX=$(exid "$RUN"); [ -z "$EX" ] && continue
        S=$(state "$EX")
        if [[ "$S" == "DONE:0" ]]; then
          fire_native "$RUN" "$LT" "$RUNGS" "$EX" && echo "EVAL:$(grep -Eo 'submitted: .*' "$LOGD/nev-$RUN.log" | head -1)" > "$SF" || echo "GAVE_UP:eval-launch" > "$SF"
        elif [[ "$S" == DONE:* || "$S" == CANCELED ]]; then
          echo "GAVE_UP:train-$S" > "$SF"; echo "[shp2] $RUN train ended badly: $S"
        fi ;;
    esac
  done
  N=$(grep -l '^EVAL' "$STATE"/*.shp2.state 2>/dev/null | wc -l)
  echo "[shp2 tick $i] $N/9 chained $(date '+%T')"
  [ "$N" -ge 9 ] && break
  sleep 300
done
echo "[shp2] exiting"
