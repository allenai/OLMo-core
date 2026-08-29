#!/bin/bash
# Wait for mix-arm build (sbatch 3483442), weka-sync, launch 28 mix trains (14 arms x 2 archs).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
for i in $(seq 1 150); do
  ST=$(sacct -j 3483442 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[mixgate $i] arm build state=$ST $(date '+%T')"
  [[ "$ST" == "COMPLETED" ]] && break
  [[ "$ST" == FAILED* || "$ST" == CANCELLED* || "$ST" == TIMEOUT ]] && { echo "[mixgate] build FAILED"; exit 1; }
  sleep 120
done
ST=$(sacct -j 3483442 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
[[ "$ST" == "COMPLETED" ]] || { echo "[mixgate] timeout"; exit 1; }
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync9 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 600
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
ARMS="mix_u16M mix_u32M mix_u64M mix_u128M mix_s16M mix_s32M mix_s64M mix_l16M mix_l32M mix_l64M mix_l128M mix_t16M mix_t32M mix_t64M"
for ARM in $ARMS; do
  for variant in full sparselandmark; do
    vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
    PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
    LR=5e-6; [ "$variant" = "sparselandmark" ] && LR=1e-5
    RUN="lmx-${vtag}-${ARM//_/}-4b"
    echo "[mixgate] launch $RUN"
    timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
      --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
      --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 65536 --lr "$LR" $PACK_FLAG \
      --global-batch 8 --micro-batch-instances 1 \
      --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
      --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
      launch > "$LOGD/${RUN}.log" 2>&1 &
    sleep 3
  done
  wait
done
echo "[mixgate] 28 mix trains submitted"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*mix*.log 2>/dev/null || echo "no errors in launch logs"
