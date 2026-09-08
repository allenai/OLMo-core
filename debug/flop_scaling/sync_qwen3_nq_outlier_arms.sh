#!/bin/bash
# Step 2 of the two-step data staging (beaker.md "Data: weka vs S3") for the dense Qwen3-4B
# NQ (48M) / outlier (160M) arms: a gantry CPU job on a weka node syncs the Qwen3-tokenized arms
# from S3 to weka, same aws-cli bootstrap as sync_s3_to_weka.sh / the ctc_modelscale relay jobs.
# Source: s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/arms_qwen3/{nmix_s48M,mix_s160M}
# Dest:   /weka/oe-training-default/ai2-llm/checkpoints/prasanns/outlier_lengthmix/arms_qwen3/<arm>
#   bash debug/flop_scaling/sync_qwen3_nq_outlier_arms.sh
set -uo pipefail
S3=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/arms_qwen3
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/outlier_lengthmix/arms_qwen3
CMD='AWS=$(command -v aws || ls /opt/conda/bin/aws 2>/dev/null || true); '
CMD+='if [ -z "$AWS" ]; then pip install -q awscli && AWS=$(command -v aws); fi; '
CMD+='[ -n "$AWS" ] || { echo FATAL_NO_AWSCLI; exit 127; }; '
CMD+='mkdir -p ~/.aws && echo "$AWS_CREDS" > ~/.aws/credentials && echo "$AWS_CFG" > ~/.aws/config; export AWS_PROFILE=S3; '
CMD+="mkdir -p $WEKA/nmix_s48M $WEKA/mix_s160M; "
CMD+="\$AWS s3 sync $S3/nmix_s48M/ $WEKA/nmix_s48M/ --only-show-errors; "
CMD+="\$AWS s3 sync $S3/mix_s160M/ $WEKA/mix_s160M/ --only-show-errors; "
CMD+="for d in $WEKA/*/; do echo \$(basename \$d) \$(ls \$d | wc -l) files; done; echo SYNC_DONE"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
exec gantry run --name "fs-sync-qwen3-nqoutlier-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$CMD"
