#!/bin/bash
# Gate on (weka sync2 DONE) && (markerfix DONE), then launch wave 1:
# check#1 LR sweep (6 runs) + check#2(a) p8k_4000 at 5e-5 (2 runs).
set -uo pipefail
SYNC_EXP=$1; FIX_EXP=$2
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

state () {
  beaker experiment get "$1" --format json 2>/dev/null | python3 -c "
import json,sys
e=json.load(sys.stdin)[0]
st=(e.get('jobs') or [{}])[-1].get('status',{})
print('CANCELED' if st.get('canceled') else ('DONE:'+str(st.get('exitCode'))) if st.get('finalized') else 'RUNNING' if st.get('started') else 'SCHEDULED')"
}

for i in $(seq 1 90); do
  S=$(state "$SYNC_EXP"); F=$(state "$FIX_EXP")
  echo "[gate $i] sync=$S markerfix=$F $(date '+%T')"
  [ "$S" = "DONE:0" ] && [ "$F" = "DONE:0" ] && break
  case "$S$F" in *CANCELED*|*DONE:1*|*DONE:2*) echo "[gate] FAILURE state -- aborting"; exit 1;; esac
  sleep 60
done
S=$(state "$SYNC_EXP"); F=$(state "$FIX_EXP")
[ "$S" = "DONE:0" ] && [ "$F" = "DONE:0" ] || { echo "[gate] timed out sync=$S fix=$F"; exit 1; }

echo "[gate] both done -- launching wave 1"
bash debug/outlier_lengthmix_scaling/launch_lr_sweep.sh

export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
LOGD=debug/outlier_lengthmix_scaling/launches
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
echo "[gate] wave 1 submitted; logs in $LOGD"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*.log 2>/dev/null || echo "no errors in launch logs"
