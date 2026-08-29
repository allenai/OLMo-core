#!/bin/bash
# Wait for the p32k_8000 arm build (mooney sbatch 3483437), weka-sync, launch the dense train.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
for i in $(seq 1 90); do
  ST=$(sacct -j 3483437 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[p32k8k $i] arm build state=$ST $(date '+%T')"
  [[ "$ST" == "COMPLETED" ]] && break
  [[ "$ST" == FAILED* || "$ST" == CANCELLED* || "$ST" == TIMEOUT ]] && { echo "[p32k8k] build FAILED"; exit 1; }
  sleep 120
done
ST=$(sacct -j 3483437 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
[[ "$ST" == "COMPLETED" ]] || { echo "[p32k8k] timeout"; exit 1; }
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync7 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 480
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
  --task outlier --variant full --model-scale 4b --model-family qwen3_5 \
  --run-name lmx-full-p32k8000-lropt-4b --num-nodes 1 --epochs 1 --seq-len 65536 --lr 5e-6 --pack \
  --global-batch 8 --micro-batch-instances 1 \
  --data-root "$WEKA_ROOT/outlier_lengthmix/arms/p32k_8000" \
  --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
  launch > debug/outlier_lengthmix_scaling/launches/lmx-full-p32k8000-lropt-4b.log 2>&1
grep -Eo 'beaker.org/ex/[A-Z0-9]+' debug/outlier_lengthmix_scaling/launches/lmx-full-p32k8000-lropt-4b.log | head -1
echo "[p32k8k] launched"
