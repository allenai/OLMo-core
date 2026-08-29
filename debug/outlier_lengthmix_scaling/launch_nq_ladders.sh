#!/bin/bash
# NQ pure-length ladders: dense 2k/8k + sparse high-data probes. Arms on S3 (agent build); sync then launch.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"; export PYTHONPATH="$REPO/src"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync11 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 480
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
launch () { # arm seq variant lr
  local ARM=$1 SEQ=$2 variant=$3 lr=$4
  local vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  local RUN="lmx-${vtag}-${ARM//_/}-4b"
  echo "[nqlad] launch $RUN"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task retrieval --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len "$SEQ" --lr "$lr" $PACK_FLAG \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
}
for A in nq2k_1250 nq2k_2500 nq2k_5000 nq2k_10000 nq2k_20000; do launch $A 4096 full 5e-6; done
wait
for A in nq8k_1000 nq8k_2000 nq8k_4000 nq8k_8000; do launch $A 16384 full 5e-6; done
launch nq2k_20000 4096  sparselandmark 1e-5
launch nq8k_8000  16384 sparselandmark 1e-5
wait
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*nq{2,8}k*.log 2>/dev/null || echo "[nqlad] launched clean"
