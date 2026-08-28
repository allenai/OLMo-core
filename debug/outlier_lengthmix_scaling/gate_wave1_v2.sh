#!/bin/bash
# Gate v2: wait for (weka sync3 DONE:0) && (either markerfix twin DONE:0, cancel the other),
# then launch wave 1 (LR sweep + check2a).
set -uo pipefail
SYNC=01M127VQTQMWZYFJ9GFMX8TSQE
FIX_GPU=01M1254B2PYH372QDCHYAYSXVF
FIX_CPU=01M1284WDX7GYHNYQ6BH6Y3YPY
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

state () {
  beaker experiment get "$1" --format json 2>/dev/null | python3 -c "
import json,sys
try:
    e=json.load(sys.stdin)[0]
except Exception:
    print('UNKNOWN'); raise SystemExit
st=(e.get('jobs') or [{}])[-1].get('status',{})
print('CANCELED' if st.get('canceled') else ('DONE:'+str(st.get('exitCode'))) if st.get('finalized') else 'RUNNING' if st.get('started') else 'SCHEDULED')"
}
cancel_exp () {
  JOB=$(beaker experiment get "$1" --format json 2>/dev/null | python3 -c "
import json,sys
e=json.load(sys.stdin)[0]
print((e.get('jobs') or [{}])[-1].get('id',''))")
  [ -n "$JOB" ] && beaker job cancel "$JOB" >/dev/null 2>&1 && echo "[gate] canceled twin job $JOB"
}

FIX_OK=""
for i in $(seq 1 120); do
  S=$(state "$SYNC"); G=$(state "$FIX_GPU"); C=$(state "$FIX_CPU")
  echo "[gate $i] sync=$S fixgpu=$G fixcpu=$C $(date '+%T')"
  if [ -z "$FIX_OK" ]; then
    if [ "$G" = "DONE:0" ]; then FIX_OK=gpu; cancel_exp "$FIX_CPU";
    elif [ "$C" = "DONE:0" ]; then FIX_OK=cpu; cancel_exp "$FIX_GPU"; fi
  fi
  [ "$S" = "DONE:0" ] && [ -n "$FIX_OK" ] && break
  if [[ "$S" == DONE:* && "$S" != "DONE:0" ]]; then echo "[gate] sync FAILED ($S)"; exit 1; fi
  if [[ "$G" == DONE:* && "$G" != "DONE:0" && "$C" == DONE:* && "$C" != "DONE:0" ]]; then
    echo "[gate] both markerfix twins failed"; exit 1; fi
  sleep 60
done
[ "$(state "$SYNC")" = "DONE:0" ] && [ -n "$FIX_OK" ] || { echo "[gate] timed out"; exit 1; }

echo "[gate] green (markerfix via $FIX_OK) -- launching wave 1"
bash debug/outlier_lengthmix_scaling/launch_lr_sweep.sh

export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
LOGD=debug/outlier_lengthmix_scaling/launches; mkdir -p "$LOGD"
for variant in full sparselandmark; do
  vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  RUN="lmx-${vtag}-p8k4000-4b"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 16384 --lr 5e-5 $PACK_FLAG \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/p8k_4000" \
    --base-checkpoint "$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim" \
    --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
done
wait
echo "[gate] wave 1 all submitted"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*.log 2>/dev/null || echo "no errors in launch logs"
