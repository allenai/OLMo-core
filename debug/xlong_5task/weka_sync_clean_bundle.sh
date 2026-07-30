#!/bin/bash
# Part 3 of 3: sync the CLEAN v2 bundle S3 -> weka, then re-audit it on weka.
#
# Parts 1-2: upload_clean_bundle.sbatch (the new rungs, from cubbins) + assemble_clean_bundle.sh
# (server-side copies of everything the audit already passed).
#
# The audit runs AFTER the sync and against WEKA, because weka is what eval reads -- an S3-side
# check would pass even if weka were empty, and an S3 push alone leaves every rung MISSING at eval
# time while the job still exits 0.
#
# Target: _eval_bundle_eval500_v2_clean, deliberately NOT an in-place fix of
# _eval_bundle_eval500_v2. That bundle is being read by eval jobs right now, and the corrected rungs
# carry DIFFERENT `n` in their filenames (the filler pool changed), so writing them alongside the old
# ones would leave two files matching one size glob and let sorted()[0] pick the rung.
#
# To use it: EVAL500=<...>/_eval_bundle_eval500_v2_clean (the runner exports it as EVAL500_ROOT).
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

CLEAN_S3=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean
CLEAN_WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean

echo "=== sync clean bundle S3 -> weka ==="
aws s3 sync "$CLEAN_S3" "$CLEAN_WK"
echo "sync rc=$?"
du -sh "$CLEAN_WK" || true

echo "=== re-audit the CLEAN bundle on weka ==="
python -u debug/xlong_5task/audit_weka_ladder.py --bundle "$CLEAN_WK"
'

gantry run --name xlong5-sync-audit-clean -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
