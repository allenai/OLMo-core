#!/bin/bash
# Launch the dense + sparse-landmark data-scaling arms for one task on Beaker.
#
#   bash debug/taskscale_lengthmix/launch_arms.sh <task> [budgets...]   # e.g. oolong 20 40 80
#
# Config is deliberately byte-identical to the outlier/qdmatch/nq campaign
# (debug/outlier_lengthmix_scaling/relaunch_sharpen_12.sh): same base, seq-len, batch, and the
# LR pair the LR sweep picked per variant -- 5e-6 dense / 1e-5 sparse. Cross-task comparison of
# the sparse-vs-dense gap is the whole point of this wave, so nothing here should drift per task.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"

# CLUSTER: jupiter is the default, but it has been 100% allocated with a deep urgent queue for
# hours at a time -- ai2/saturn (A100-80GB) places our jobs in minutes and runs the same config.
CLUSTER=${CLUSTER:-ai2/jupiter-cirrascale-2}
TASK=${1:?task}; shift
BUDGETS=("$@"); [ ${#BUDGETS[@]} -eq 0 ] && BUDGETS=(20 40 80)
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
ARMS="$WEKA/taskscale_lengthmix/arms_tokenized"
LOGD=debug/taskscale_lengthmix/launches

launch () { local ARM=$1 VARIANT=$2 LR=$3 RUN=$4; shift 4
  echo "[tsl] launch $RUN ($ARM, $VARIANT)"
  timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task "$TASK" --variant "$VARIANT" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 65536 --lr "$LR" "$@" \
    --cluster "$CLUSTER" \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$ARMS/$ARM" \
    --base-checkpoint "$BASE" --wandb-group taskscale-lengthmix \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 4
}

for B in "${BUDGETS[@]}"; do
  ARM="${TASK}_mix_s${B}M"
  launch "$ARM" full           5e-6 "tsl-full-${TASK}-s${B}M-4b" --pack
  launch "$ARM" sparselandmark 1e-5 "tsl-slm-${TASK}-s${B}M-4b"
done
wait
echo "[tsl] $TASK: submitted ${#BUDGETS[@]} budgets x 2 variants $(date '+%T')"
