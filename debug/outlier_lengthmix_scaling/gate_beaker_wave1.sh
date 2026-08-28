#!/bin/bash
# Poll weka markerfix (nep2); on DONE:0 launch the Beaker LR sweep + check2a p8k_4000 pair.
set -uo pipefail
FIX=01M128RAF2SS07FTTB1SYRM947
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
for i in $(seq 1 60); do
  S=$(state "$FIX"); echo "[bgate $i] nep2=$S $(date '+%T')"
  [ "$S" = "DONE:0" ] && break
  [[ "$S" == DONE:* || "$S" == CANCELED ]] && { echo "[bgate] nep2 FAILED ($S)"; exit 1; }
  sleep 45
done
[ "$(state "$FIX")" = "DONE:0" ] || { echo "[bgate] timeout"; exit 1; }

echo "[bgate] weka markerfix ready -- launching Beaker LR sweep + p8k_4000 pair"
bash debug/outlier_lengthmix_scaling/launch_lr_sweep.sh
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
LOGD=debug/outlier_lengthmix_scaling/launches
mkdir -p "$LOGD"
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
echo "[bgate] Beaker wave 1 submitted"
