#!/bin/bash
# 128k transfer test: dense pure-vs-mix at two matched budgets + sparse mix probe.
# Seq-lens per the 128k recon (pure dense 132096 unpacked; mix dense 131072 packed; sparse 134144).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"; export PYTHONPATH="$REPO/src"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync12 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 600
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
launch () { # run arm variant seq lr nodes extra...
  local RUN=$1 ARM=$2 variant=$3 SEQ=$4 LR=$5 NODES=$6; shift 6
  echo "[128k] launch $RUN"
  timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes "$NODES" --epochs 1 --seq-len "$SEQ" --lr "$LR" "$@" \
    --global-batch 8 --micro-batch-instances 1 --activation-checkpointing full \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-128k \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 4
}
launch lmx-full-p128k1500-4b  p128k_1500 full 132096 5e-6 2 --cp-degree 2
launch lmx-full-mixsx200M-4b  mix_sx200M full 131072 5e-6 2 --cp-degree 2 --pack
launch lmx-full-p128k500-4b   p128k_500  full 132096 5e-6 2 --cp-degree 2
launch lmx-full-mixsx64M-4b   mix_sx64M  full 131072 5e-6 2 --cp-degree 2 --pack
launch lmx-slm-mixsx200M-4b   mix_sx200M sparselandmark 134144 1e-5 1
wait
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*{128k,sx}*.log 2>/dev/null || echo "[128k] launched clean"
