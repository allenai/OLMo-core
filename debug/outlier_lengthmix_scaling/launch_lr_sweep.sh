#!/bin/bash
# Stage 0.5: LR sweep / smoke -- 3 LRs x {full, sparselandmark} on the 2k-context 5000-example
# arm (625 steps at global_batch 8). Per-arch winners become the grid LRs; runs double as the
# train-chain smoke test. Launch AFTER the S3->weka sync of arms/ and the markerfix base exist.
#
#   bash debug/outlier_lengthmix_scaling/launch_lr_sweep.sh          # launch all 6
#   DRY=1 bash .../launch_lr_sweep.sh                                # dry_run each
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
export PYTHONPATH="$REPO/src"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CMD=launch; [ "${DRY:-0}" = "1" ] && CMD=dry_run
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
DATA="$WEKA_ROOT/outlier_lengthmix/arms/lr2k5000"
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches; mkdir -p "$LOGD"

lr_tag () { case "$1" in 2e-5) echo lr2e5;; 5e-5) echo lr5e5;; 1.2e-4) echo lr1p2e4;; *) exit 2;; esac; }

n=0
for variant in full sparselandmark; do
  vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  PACK_FLAG=""
  [ "$variant" = "full" ] && PACK_FLAG="--pack"
  for lr in 2e-5 5e-5 1.2e-4; do
    tag=$(lr_tag "$lr")
    RUN="lmx-${vtag}-${tag}-4b"
    echo "=== [$CMD] $RUN (variant=$variant lr=$lr) ==="
    timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
      --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
      --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 4096 --lr "$lr" $PACK_FLAG \
      --global-batch 8 --micro-batch-instances 1 \
      --data-root "$DATA" --base-checkpoint "$BASE" \
      --wandb-group outlier-lengthmix-lr \
      "$CMD" > "$LOGD/${RUN}.log" 2>&1 &
    n=$((n+1)); sleep 3
  done
done
wait
echo "submitted/attempted $n; check $LOGD/*.log (launch follower killed by timeout is EXPECTED; the Beaker job keeps running)"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*.log 2>/dev/null && echo "^^ CHECK THESE" || echo "no errors in launch logs"
