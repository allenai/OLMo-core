#!/bin/bash
# Sync the 128k qwen3 (dense) mix shard S3 -> weka for the Beaker dense relaunch.
# (qwen3-4b-base-trainedmark base is already on weka from the 256k run; qwen35_128k only needed
#  if we ever run the hybrid on Beaker -- currently hybrid is local, so sync qwen3_128k only.)
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -e
export PATH=/opt/conda/bin:$PATH
if ! command -v aws >/dev/null 2>&1; then echo "installing awscli..."; python -m pip install -q awscli; fi
AWS=$(command -v aws); echo "using aws: $AWS"
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/ctc_suite
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite
for d in shards/contra_mix_qwen3_10k_128k shards/contra_mix_qwen35_10k_128k; do
  echo "=== sync $d ==="
  "$AWS" s3 sync "$S3/$d" "$WK/$d"
done
echo "qwen3_128k files:  $(ls "$WK/shards/contra_mix_qwen3_10k_128k" | wc -l)"
echo "qwen35_128k files: $(ls "$WK/shards/contra_mix_qwen35_10k_128k" | wc -l)"
echo "WEKA_SYNC_128K_DONE"
'

gantry run --name ctc-contra-mix128k-weka-sync -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
