#!/bin/bash
# DENSE control at the two sparse takeoff budgets.
#
# qdmatch sparse @160M took off on seed 1 (.448 @8k) and stayed on the floor on seed 3 (.002). The
# claim being made from that is "sparse near its threshold is BIMODAL across seeds". That claim
# needs a control: if DENSE at the same budget on the same data were also seed-unstable, the
# bimodality would be a property of the arm/budget, not of sparse attention. Dense at these budgets
# is far above its own threshold, so it should be boring -- and demonstrating that it is boring is
# what licenses the sparse-specific reading.
#
#   bash debug/outlier_lengthmix_scaling/launch_takeoff_dense_control.sh [seed]
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
CLUSTER=${CLUSTER:-ai2/jupiter-cirrascale-2}
SEED=${1:-3}
launch () { local ARM=$1 TASK=$2 RUN=$3
  timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task "$TASK" --variant full --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 65536 --lr 5e-6 --pack --seed "$SEED" \
    --cluster "$CLUSTER" --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1
  echo "$RUN -> $(grep -oE 'beaker\.org/ex/[A-Z0-9]{26}' $LOGD/${RUN}.log | tail -1 | sed 's|.*/||')"
}
launch nmix_s48M  retrieval "lmx-full-nmixs48M-nq-s${SEED}-4b"
launch qmix_s160M qdmatch   "lmx-full-qmixs160M-qd-s${SEED}-4b"
