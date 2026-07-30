#!/bin/bash
# Audit the LIVE weka v2 eval bundle: per rung, eval_size + FEVER/wiki contamination + glob ambiguity.
#
# Runs on Beaker because the bundle the evals actually read lives on weka, which is unreachable from
# Berkeley -- and the local /scratch copies are a DIFFERENT build (different `n` in the filenames),
# so auditing those would answer the wrong question.
#
# The FEVER/wiki fingerprint set is built at Berkeley (the source corpora are on /scratch) and
# shipped via S3: debug/xlong_5task/ built s3://ai2-llm/checkpoints/prasanns/_audit/fever_wiki_hashes.txt.gz
#
# Answers, per (task, rung): does it exist, is eval_size >= 500, and is it PubMed-only?
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -e
export PATH=/opt/conda/bin:$PATH
command -v aws >/dev/null 2>&1 || python -m pip install -q awscli
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
aws s3 cp s3://ai2-llm/checkpoints/prasanns/_audit/fever_wiki_hashes.txt.gz /tmp/fw.txt.gz --only-show-errors
# The audit itself is a real file in the cloned repo -- no heredoc, so no shell/f-string escaping.
python -u debug/xlong_5task/audit_weka_ladder.py
'

gantry run --name xlong5-audit-weka-ladder -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
