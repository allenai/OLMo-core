#!/bin/bash
# Relaunch a SINGLE (task, budget, variant) arm -- launch_arms.sh always fires both variants of a
# budget, which is wrong when only one half of a pair died. Config is copied from launch_arms.sh
# verbatim so a relaunched cell stays comparable with the ones that succeeded first time.
#
#   bash debug/taskscale_lengthmix/relaunch_one.sh <task> <budgetM> <full|sparselandmark> [cluster]
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
TASK=${1:?task}; B=${2:?budgetM}; VARIANT=${3:?variant}
CLUSTER=${4:-ai2/jupiter-cirrascale-2}
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
ARMS="$WEKA/taskscale_lengthmix/arms_tokenized"
LOGD=debug/taskscale_lengthmix/launches
case "$VARIANT" in full) LR=5e-6; TAG=full; EXTRA=--pack ;; sparselandmark) LR=1e-5; TAG=slm; EXTRA= ;;
  *) echo "bad variant $VARIANT"; exit 1 ;; esac
RUN="tslR-${TAG}-${TASK}-s${B}M-4b"
timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
  --task "$TASK" --variant "$VARIANT" --model-scale 4b --model-family qwen3_5 \
  --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 65536 --lr "$LR" $EXTRA \
  --cluster "$CLUSTER" --global-batch 8 --micro-batch-instances 1 \
  --data-root "$ARMS/${TASK}_mix_s${B}M" \
  --base-checkpoint "$BASE" --wandb-group taskscale-lengthmix \
  launch > "$LOGD/${RUN}.log" 2>&1
echo "$RUN -> $(grep -oE 'beaker\.org/ex/[A-Z0-9]{26}' $LOGD/${RUN}.log | tail -1 | sed 's|.*/||')"
