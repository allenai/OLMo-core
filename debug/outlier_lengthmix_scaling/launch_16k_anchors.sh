#!/bin/bash
# 16k-length anchor runs to pin K(16k) for the N(0.9) extrapolation.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"; export PYTHONPATH="$REPO/src"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
launch () { # arm variant lr
  local ARM=$1 variant=$2 lr=$3
  local vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  local RUN="lmx-${vtag}-${ARM//_/}-lropt-4b"
  echo "[16k] launch $RUN (lr=$lr)"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 32768 --lr "$lr" $PACK_FLAG \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
}
launch p16k_250  full 5e-6
launch p16k_1000 full 5e-6
launch p16k_4000 full 5e-6
launch p16k_4000 sparselandmark 1e-5
wait
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*p16k*.log 2>/dev/null || echo "no errors in launch logs"
