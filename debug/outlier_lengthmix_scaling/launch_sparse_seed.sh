#!/bin/bash
# Extra sparse seeds at an ARBITRARY lmx arm/budget, to map where the bimodality lives.
#
# Four seeds at each takeoff budget showed sparse is bimodal there (nq@48M 3-of-4, qdmatch@160M
# 1-of-4 clean). The open question that changes what we would recommend is whether training PAST
# the threshold buys reliability: if qdmatch@320M -- the point everyone treats as settled -- is
# also a coin flip, then no sparse number in this campaign is safe at one seed. If it is tight,
# the instability is confined to a window around the threshold and "just train past it" is real
# advice. The below-threshold side matters too: a floor that is reliably a floor is a different
# claim from one that occasionally takes off.
#
#   bash debug/outlier_lengthmix_scaling/launch_sparse_seed.sh <arm> <task> <run-prefix> <seed>
#   e.g.  ... qmix_s320M qdmatch lmx-slm-qmixs320M-qd 4
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
ARM=${1:?arm}; TASK=${2:?task}; PREFIX=${3:?run-prefix}; SEED=${4:?seed}
RUN="${PREFIX}-s${SEED}-4b"
timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
  --task "$TASK" --variant sparselandmark --model-scale 4b --model-family qwen3_5 \
  --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len 65536 --lr 1e-5 --seed "$SEED" \
  --cluster "$CLUSTER" --global-batch 8 --micro-batch-instances 1 \
  --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
  --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
  launch > "$LOGD/${RUN}.log" 2>&1
echo "$RUN -> $(grep -oE 'beaker\.org/ex/[A-Z0-9]{26}' $LOGD/${RUN}.log | tail -1 | sed 's|.*/||')"
