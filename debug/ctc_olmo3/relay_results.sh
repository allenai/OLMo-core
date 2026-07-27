#!/bin/bash
# Relay the OLMo-3 rung results from weka -> S3 so Berkeley can pull them.
# beaker_eval.sh's inline relay assumed `aws` existed in the baked image; it does not
# (/opt/conda/bin/aws: No such file or directory), so the results sat on weka. weka_sync.sh gets
# this right by pip-installing awscli first -- same thing here.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -e
export PATH=/opt/conda/bin:$PATH
command -v aws >/dev/null 2>&1 || python -m pip install -q awscli
AWS=$(command -v aws); echo "using aws: $AWS"
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3/results
echo "=== result files on weka ==="
find "$WK" -name "rung_*.json" ! -name "*.raw.json" ! -name "*.generations.json" | sort
"$AWS" s3 sync "$WK" s3://ai2-llm/checkpoints/prasanns/ctc_olmo3/results
echo "RELAY_DONE"
'

gantry run --name ctc-olmo3-relay-$(date +%H%M%S) -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
