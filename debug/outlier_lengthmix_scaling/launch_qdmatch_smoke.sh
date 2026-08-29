#!/bin/bash
# qdmatch_nq smoke pair-of-pairs: q2k_5000 (seq 4096) + q8k_4000 (seq 16384), full@5e-6 / slm@1e-5.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"; export PYTHONPATH="$REPO/src"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches; mkdir -p "$LOGD"
sleep 300   # let outlier-lm-weka-sync6 land the q* arms before jobs can start
launch () { # arm seq variant lr
  local ARM=$1 SEQ=$2 variant=$3 lr=$4
  local vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  local RUN="lmx-${vtag}-${ARM//_/}-qd-4b"
  echo "[qd] launch $RUN (lr=$lr seq=$SEQ)"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task qdmatch --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len "$SEQ" --lr "$lr" $PACK_FLAG \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
}
launch q2k_5000 4096  full 5e-6
launch q2k_5000 4096  sparselandmark 1e-5
launch q8k_4000 16384 full 5e-6
launch q8k_4000 16384 sparselandmark 1e-5
wait
echo "[qd] smoke submitted"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*-qd-4b.log 2>/dev/null || echo "no errors in launch logs"
