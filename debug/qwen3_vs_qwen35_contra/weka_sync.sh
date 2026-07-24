#!/bin/bash
# Step 2 of weka staging: gantry job that syncs S3 -> weka (weka isn't reachable from Berkeley).
# Syncs the two 10k mix shards + the Qwen3 dense base. (Qwen3.5 base is already on weka.)
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -e
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/ctc_suite
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite
for d in bases/qwen3-4b-base-trainedmark shards/contra_mix_qwen3_10k shards/contra_mix_qwen35_10k; do
  echo "=== sync $d ==="
  aws s3 sync "$S3/$d" "$WK/$d"
done
echo "=== verify ==="
ls "$WK/bases/qwen3-4b-base-trainedmark/model_and_optim/.metadata" && echo QWEN3_BASE_OK
echo "qwen3 shard files: $(ls "$WK/shards/contra_mix_qwen3_10k" | wc -l)"
echo "qwen35 shard files: $(ls "$WK/shards/contra_mix_qwen35_10k" | wc -l)"
echo "WEKA_SYNC_DONE"
'

gantry run --name ctc-contra-mix-weka-sync -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
