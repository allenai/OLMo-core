#!/bin/bash
# Watch the mooney staging chain; when S3 staging (job id in $1) completes OK, fire the gantry
# S3->weka sync and wait for it; then run the local dry-run validation of the sparselandmark arm.
# Exits when done (the interactive session reviews + launches the LR sweep).
set -uo pipefail
S3JOB=$1
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
GANTRY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry

while true; do
  ST=$(sacct -j "$S3JOB" -n -o State 2>/dev/null | head -1 | tr -d ' ')
  case "$ST" in
    COMPLETED) echo "[watch] S3 staging done"; break ;;
    FAILED|CANCELLED*|TIMEOUT) echo "[watch] S3 staging $ST -- ABORT"; exit 1 ;;
    *) sleep 60 ;;
  esac
done

cd "$REPO"
echo "[watch] firing gantry S3->weka sync"
$GANTRY run --name outlier-lm-weka-sync -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune,ai2/ceres,ai2/saturn,ai2/jupiter --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c '
set -e
mkdir -p ~/.aws
printenv AWS_CREDS > ~/.aws/credentials
printenv AWS_CFG > ~/.aws/config
export AWS_PROFILE=S3
aws s3 sync s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  /weka/oe-training-default/ai2-llm/checkpoints/prasanns/outlier_lengthmix --only-show-errors
echo SYNC-OK
ls /weka/oe-training-default/ai2-llm/checkpoints/prasanns/outlier_lengthmix/arms/ | head
' 2>&1 | tail -3

echo "[watch] waiting 5 min for sync, then local dry-run validation"
sleep 300
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
for variant in sparselandmark full; do
  echo "=== dry-run $variant vs /net arm shard ==="
  PYTHONPATH=$REPO/src $PY src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py \
    --task outlier --data /net/mooney/data/prasann/outlier_lengthmix/arms_tokenized/lr2k5000 \
    --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --seq-len 4096 --epochs 1 --global-batch 8 --lr 5e-5 \
    --dry-run --dry-run-world-size 8 2>&1 | tail -12
done
echo "[watch] ALL DONE -- review dry-run output above, then bash debug/outlier_lengthmix_scaling/launch_lr_sweep.sh"
