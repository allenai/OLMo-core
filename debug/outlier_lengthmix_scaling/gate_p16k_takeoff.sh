#!/bin/bash
# Wait for n111-ext arm build (sbatch 3483438), weka-sync, launch sparse 16k takeoff trains.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
for i in $(seq 1 120); do
  ST=$(sacct -j 3483438 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[p16kT $i] arm build state=$ST $(date '+%T')"
  [[ "$ST" == "COMPLETED" ]] && break
  [[ "$ST" == FAILED* || "$ST" == CANCELLED* || "$ST" == TIMEOUT ]] && { echo "[p16kT] build FAILED"; exit 1; }
  sleep 120
done
ST=$(sacct -j 3483438 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
[[ "$ST" == "COMPLETED" ]] || { echo "[p16kT] timeout"; exit 1; }
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync8 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 480
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
for ARM in p16k_8000 p16k_16000; do
  RUN="lmx-slm-${ARM//_/}-lropt-4b"
  echo "[p16kT] launch $RUN"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant sparselandmark --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 32768 --lr 1e-5 \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
done
wait
grep -l "Traceback\|ERROR" "$LOGD"/lmx-slm-p16k{8000,16000}-lropt-4b.log 2>/dev/null || echo "[p16kT] launched clean"
