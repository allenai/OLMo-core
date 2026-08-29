#!/bin/bash
# Wait for s96/s160 arm build (3483891), weka-sync, launch 5 trains (incl. sparse 64M seed-2 replicate).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
for i in $(seq 1 150); do
  ST=$(sacct -j 3483891 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[s160gate $i] build=$ST $(date '+%T')"
  [[ "$ST" == "COMPLETED" ]] && break
  [[ "$ST" == FAILED* || "$ST" == CANCELLED* || "$ST" == TIMEOUT ]] && { echo "[s160gate] build FAILED"; exit 1; }
  sleep 120
done
[[ "$(sacct -j 3483891 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')" == "COMPLETED" ]] || exit 1
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync10 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 480
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
launch () { # arm variant lr run extra
  local ARM=$1 variant=$2 lr=$3 RUN=$4 EXTRA=${5:-}
  local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  echo "[s160gate] launch $RUN"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 65536 --lr "$lr" $PACK_FLAG $EXTRA \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
}
launch mix_s96M  full           5e-6 lmx-full-mixs96M-4b
launch mix_s96M  sparselandmark 1e-5 lmx-slm-mixs96M-4b
launch mix_s160M full           5e-6 lmx-full-mixs160M-4b
launch mix_s160M sparselandmark 1e-5 lmx-slm-mixs160M-4b
launch mix_s64M  sparselandmark 1e-5 lmx-slm-mixs64M-s2-4b "--seed 2"
wait
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*mixs{96,160,64M-s2}*.log 2>/dev/null || echo "[s160gate] launched clean"
