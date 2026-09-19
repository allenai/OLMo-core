#!/usr/bin/env bash
# Stage locally-built hybridish CTC SFT shards to weka, via S3.
#
# Beaker jobs cannot read this machine's /scratch, and weka is not mounted here -- so the transfer
# is the documented two-step: push to the S3 buffer from here, then run a gantry job that pulls
# S3 -> weka. Pushing to S3 alone does nothing for a training job; the sync is what makes the
# shards visible, and its absence shows up as a MISSING path at step 0.
#
#   bash src/scripts/data/hybridish/stage_shards_to_weka.sh /scratch/.../shards_long shards_long
set -euo pipefail

SRC="${1:?usage: stage_shards_to_weka.sh <local-shard-dir> <dest-name>}"
NAME="${2:?}"
S3="s3://ai2-llm/checkpoints/prasanns/ctc_hybridish_sft/${NAME}"
WEKA="/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft/${NAME}"
CLUSTER="${CLUSTER:-ai2/saturn}"

echo "[1/2] $SRC -> $S3"
aws s3 sync "$SRC" "$S3" --only-show-errors
aws s3 ls "$S3/" | sed 's/^/      /'

echo "[2/2] gantry sync $S3 -> $WEKA on $CLUSTER"
gantry run --name "stage-${NAME}" --workspace ai2/flex2 --budget ai2/oe-other \
  --cluster "$CLUSTER" --weka oe-training-default:/weka/oe-training-default \
  --cpus 2 --priority urgent --no-python --allow-dirty --timeout 0 --yes \
  -- bash -c "mkdir -p '$WEKA' && aws s3 sync '$S3' '$WEKA' --only-show-errors && ls -la '$WEKA'"
