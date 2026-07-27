#!/bin/bash
# Step 2 of weka staging for the OLMo-3 CTC arm: gantry job that syncs S3 -> weka (weka is not
# reachable from Berkeley). Syncs the marker-repaired Olmo-3 7B base, both 10k-scale shards, the
# patched marker tokenizer, and the 2k/4k/8k/16k eval rungs.
# S3-push alone is NOT enough: every input would log MISSING and the job would exit 0.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -e
export PATH=/opt/conda/bin:$PATH
if ! command -v aws >/dev/null 2>&1; then
  echo "installing awscli into conda env..."; python -m pip install -q awscli
fi
AWS=$(command -v aws); echo "using aws: $AWS"
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/ctc_olmo3
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3
for d in bases/olmo3-7b-base-fixmark shards/contradiction_train shards/qdmatch_hpqa_train tokenizer eval_rungs; do
  echo "=== sync $d ==="
  "$AWS" s3 sync "$S3/$d" "$WK/$d"
done
echo "=== verify ==="
ls -l "$WK/bases/olmo3-7b-base-fixmark/model_and_optim/.metadata" && echo OLMO3_BASE_OK
echo "contradiction shard files: $(ls "$WK/shards/contradiction_train" | wc -l)"
echo "qdmatch shard files: $(ls "$WK/shards/qdmatch_hpqa_train" | wc -l)"
echo "tokenizer files: $(ls "$WK/tokenizer" | wc -l)"
echo "eval rungs: $(find "$WK/eval_rungs" -name "*.jsonl" | wc -l)"
echo "WEKA_SYNC_DONE"
'

gantry run --name ctc-olmo3-weka-sync -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
