#!/bin/bash
# Wave 2: check#2 (b,c + FLOP-matched control) + check#3 2k scaling ladder, both variants, lr 5e-5.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"; export PYTHONPATH="$REPO/src"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches; mkdir -p "$LOGD"
LR="${LR:-5e-5}"

launch () { # arm seqlen
  local ARM=$1 SEQ=$2
  for variant in full sparselandmark; do
    local vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
    local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
    local RUN="lmx-${vtag}-${ARM//_/}-4b"
    echo "=== launch $RUN (arm=$ARM seq=$SEQ lr=$LR) ==="
    timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
      --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
      --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len "$SEQ" --lr "$LR" $PACK_FLAG \
      --global-batch 8 --micro-batch-instances 1 \
      --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
      --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
      launch > "$LOGD/${RUN}.log" 2>&1 &
    sleep 3
  done
}
launch m8k_mix 16384
launch p8k_5000 16384
launch p8k_8000 16384
launch p2k_1250 4096
launch p2k_2500 4096
launch p2k_10000 4096
launch p2k_20000 4096
wait
echo "wave2 submitted; logs in $LOGD"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*.log 2>/dev/null || echo "no errors in launch logs"
