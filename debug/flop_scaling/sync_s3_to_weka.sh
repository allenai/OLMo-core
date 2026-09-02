#!/bin/bash
# Step 2 of the two-step data staging (beaker.md "Data: weka vs S3"): a gantry job on a weka node
# syncs the FLOP-scaling shards + the Qwen3-4B fixmark base from S3 to weka. Same aws-cli
# bootstrap as the proven relay jobs (debug/ctc_modelscale/relay_*_to_s3.sh). Incremental.
#   bash debug/flop_scaling/sync_s3_to_weka.sh
set -uo pipefail
PFX="${PFX:-flop_scaling}"   # flop_scaling (Qwen3-4B study) | flop_scaling35 (Qwen3.5 KV shards)
S3=s3://ai2-llm/checkpoints/prasanns/$PFX
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/$PFX
CMD='AWS=$(command -v aws || ls /opt/conda/bin/aws 2>/dev/null || true); '
CMD+='if [ -z "$AWS" ]; then pip install -q awscli && AWS=$(command -v aws); fi; '
CMD+='[ -n "$AWS" ] || { echo FATAL_NO_AWSCLI; exit 127; }; '
CMD+='mkdir -p ~/.aws && echo "$AWS_CREDS" > ~/.aws/credentials && echo "$AWS_CFG" > ~/.aws/config; export AWS_PROFILE=S3; '
CMD+="mkdir -p $WEKA/shards $WEKA/bases; "
CMD+="[ $PFX = flop_scaling ] && { \$AWS s3 sync $S3/bases/ $WEKA/bases/ --only-show-errors; echo base_files=\$(ls $WEKA/bases/q4b-dense-cpt-fixmark/model_and_optim 2>/dev/null | wc -l); }; "
CMD+="\$AWS s3 sync $S3/shards/ $WEKA/shards/ --only-show-errors; "
CMD+="for d in $WEKA/shards/*/; do echo \$(basename \$d) \$(ls \$d | wc -l) files; done; echo SYNC_DONE"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
exec gantry run --name "fs-sync-$PFX-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$CMD"
