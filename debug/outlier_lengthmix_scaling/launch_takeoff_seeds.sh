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
# 2026-09-03, after the seed-3 pair came back: seed 3 on qdmatch@160M did not take off AT ALL
# (.002/.000/.001 vs seed 1's .448/.205/.034) while nq@48M took off but landed .30 higher at 32k.
# So the quantity to measure is the takeoff FRACTION, which needs more than two draws. Seeds 4 and
# 5 at both takeoff budgets take each to four seeds; 320M gets a second seed to test whether the
# post-takeoff point everyone treats as settled actually is.
#
#   bash .../launch_takeoff_seeds.sh          # the original seed-3 pair
#   bash .../launch_takeoff_seeds.sh 4        # one more seed at both takeoff budgets
SEED=${1:-}
if [ -n "$SEED" ]; then
  launch nmix_s48M  retrieval "lmx-slm-nmixs48M-nq-s${SEED}-4b"  "$SEED"
  launch qmix_s160M qdmatch   "lmx-slm-qmixs160M-qd-s${SEED}-4b" "$SEED"
  exit 0
fi
launch nmix_s48M  retrieval lmx-slm-nmixs48M-nq-s3-4b  3
launch qmix_s160M qdmatch   lmx-slm-qmixs160M-qd-s3-4b 3
