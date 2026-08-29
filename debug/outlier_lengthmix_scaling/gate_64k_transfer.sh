#!/bin/bash
# 64k transfer test (128k fallback): wait for n440 build (3484415), sync, launch 3 trains 1-node.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
for i in $(seq 1 120); do
  ST=$(sacct -j 3484417 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[64kgate $i] build=$ST $(date '+%T')"
  [[ "$ST" == "COMPLETED" ]] && break
  [[ "$ST" == FAILED* || "$ST" == CANCELLED* || "$ST" == TIMEOUT ]] && { echo "[64kgate] build FAILED"; exit 1; }
  sleep 120
done
[[ "$(sacct -j 3484417 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')" == "COMPLETED" ]] || exit 1
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync13 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 480
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
launch () { # run arm variant seq lr extra...
  local RUN=$1 ARM=$2 variant=$3 SEQ=$4 LR=$5; shift 5
  echo "[64kgate] launch $RUN"
  timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len "$SEQ" --lr "$LR" "$@" \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-64k \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 4
}
launch lmx-full-p64k1500-4b   p64k_1500   full 66560 5e-6
launch lmx-full-mixs64k96M-4b mix_s64k96M full 65536 5e-6 --pack
launch lmx-slm-mixs64k96M-4b  mix_s64k96M sparselandmark 67584 1e-5
wait
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*64k*.log 2>/dev/null || echo "[64kgate] launched clean"
