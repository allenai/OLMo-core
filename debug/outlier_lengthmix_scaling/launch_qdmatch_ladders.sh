#!/bin/bash
# qdmatch_nq pure-length ladders: dense 2k/8k ladders + sparse high-data probes.
# Reuses smoke results for q2k_5000/q8k_4000 (already trained). Arms already on weka (sync6).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"; export PYTHONPATH="$REPO/src"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
launch () { # arm seq variant lr
  local ARM=$1 SEQ=$2 variant=$3 lr=$4
  local vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  local RUN="lmx-${vtag}-${ARM//_/}-qd-4b"
  echo "[qdlad] launch $RUN"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task qdmatch --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len "$SEQ" --lr "$lr" $PACK_FLAG \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
}
launch q2k_1250  4096  full 5e-6
launch q2k_2500  4096  full 5e-6
launch q2k_10000 4096  full 5e-6
launch q2k_20000 4096  full 5e-6
launch q8k_1000  16384 full 5e-6
launch q8k_2000  16384 full 5e-6
launch q8k_8000  16384 full 5e-6
launch q2k_20000 4096  sparselandmark 1e-5
launch q8k_8000  16384 sparselandmark 1e-5
wait
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*q{2,8}k*-qd-4b.log 2>/dev/null || echo "[qdlad] launched clean"
