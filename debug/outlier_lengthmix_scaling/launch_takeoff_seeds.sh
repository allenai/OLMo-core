#!/bin/bash
# Seed replicates at the two SPARSE TAKEOFF budgets.
#
# nq sparse jumps .023 -> .728 @8k between 32M and 48M tokens, and qdmatch sparse .002 -> .448 @8k
# between 64M and 160M. Those two steps are now the campaign's headline, and each rests on ONE run.
# A step that is really a seed fluke would look exactly the same, so re-run the post-takeoff arm at
# a second seed before the shape goes into the writeup. No new data: same tokenized arms.
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
launch () { local ARM=$1 TASK=$2 RUN=$3 SEED=$4
  timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task "$TASK" --variant sparselandmark --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 65536 --lr 1e-5 --seed "$SEED" \
    --cluster "$CLUSTER" --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1
  echo "$RUN -> $(grep -oE 'beaker\.org/ex/[A-Z0-9]{26}' $LOGD/${RUN}.log | tail -1 | sed 's|.*/||')"
}
launch nmix_s48M  retrieval lmx-slm-nmixs48M-nq-s3-4b  3
launch qmix_s160M qdmatch   lmx-slm-qmixs160M-qd-s3-4b 3
