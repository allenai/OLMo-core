#!/usr/bin/env bash
# Second half of the staging two-step: S3 -> weka, so Beaker jobs can read the checkpoints.
# Pushing to S3 alone does nothing for a Beaker job; its absence shows up as a MISSING path at step 0.
#
# Two things this script exists to get right, both of which failed first:
#   * The container has no AWS credentials. Without the --env-secret pair and the ~/.aws writes the
#     sync dies with "Unable to locate credentials" after a clean setup.
#   * gantry pins the job to a PUSHED commit, so a fresh local commit fails with "not our ref".
#     --ref pins to a commit already on the remote; this job runs no repo code, so any ref does.
# The remote command is written WITHOUT shell loops on purpose: nesting a loop variable through
# bash -c "..." quoting silently shipped a literal "$c" and synced into a directory of that name.
set -euo pipefail
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
REF="${REF:-$(git rev-parse origin/prasann/landmark)}"
S3=s3://ai2-llm/checkpoints/prasanns/ctc_hybridish_sft
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft

read -r -d '' CMD <<INNER || true
set -eu
mkdir -p ~/.aws
echo "\$AWS_CREDS" > ~/.aws/credentials
echo "\$AWS_CFG" > ~/.aws/config
rm -rf '$WEKA/\$c'
mkdir -p '$WEKA/sft_4to1_ml_hf' '$WEKA/sft_7to1_ml_hf'
aws s3 sync '$S3/sft_4to1_ml_hf' '$WEKA/sft_4to1_ml_hf' --only-show-errors
aws s3 sync '$S3/sft_7to1_ml_hf' '$WEKA/sft_7to1_ml_hf' --only-show-errors
du -sh '$WEKA/sft_4to1_ml_hf' '$WEKA/sft_7to1_ml_hf'
grep -o '"scalable_softmax": [a-z]*' '$WEKA/sft_4to1_ml_hf/config.json'
grep -o '"scalable_softmax": [a-z]*' '$WEKA/sft_7to1_ml_hf/config.json'
ls '$WEKA'
INNER

exec "$GANTRY" run --name sync-hybridish-sft-ckpts -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --gpus 0 --priority urgent \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --ref "$REF" --cpus 4 --no-python --allow-dirty --timeout 0 --yes -- bash -c "$CMD"
