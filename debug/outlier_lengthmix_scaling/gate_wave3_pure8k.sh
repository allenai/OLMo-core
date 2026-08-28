#!/bin/bash
# Wave 3 (pure-length, per user directive 2026-08-28): pure-8k data-scaling ladder at per-arch
# optimal LRs (full 5e-6, slm 1e-5). Waits for the mooney->S3 arm push (slurm 3482952), syncs
# S3->weka, then launches 14 Beaker trains: p8k_{250,1000,2000,4000,8000,16000,32000} x 2 archs.
# Run names carry -lropt- so they don't collide with wave-2's 2e-5 copies.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
PUSHJOB=3482952

for i in $(seq 1 60); do
  ST=$(sacct -j $PUSHJOB --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[w3gate $i] push job $PUSHJOB state=$ST $(date '+%T')"
  [[ "$ST" == "COMPLETED" ]] && break
  [[ "$ST" == FAILED* || "$ST" == CANCELLED* || "$ST" == TIMEOUT ]] && { echo "[w3gate] push FAILED"; exit 1; }
  sleep 60
done
ST=$(sacct -j $PUSHJOB --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
[[ "$ST" == "COMPLETED" ]] || { echo "[w3gate] push timeout (state=$ST)"; exit 1; }

echo "[w3gate] syncing wave-3 arms to weka"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync5 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 480

PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches; mkdir -p "$LOGD"

launch () { # arm variant lr
  local ARM=$1 variant=$2 lr=$3
  local vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  local RUN="lmx-${vtag}-${ARM//_/}-lropt-4b"
  echo "[w3gate] launch $RUN (lr=$lr)"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 16384 --lr "$lr" $PACK_FLAG \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
}
for ARM in p8k_250 p8k_1000 p8k_2000 p8k_4000 p8k_8000 p8k_16000 p8k_32000; do
  launch "$ARM" full 5e-6
  launch "$ARM" sparselandmark 1e-5
done
wait
echo "[w3gate] wave-3 pure-8k submitted"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*lropt*.log 2>/dev/null || echo "no errors in launch logs"
