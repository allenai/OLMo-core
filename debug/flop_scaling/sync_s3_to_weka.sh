#!/bin/bash
# Step 2 of the two-step data staging (beaker.md "Data: weka vs S3"): a gantry job on a weka node
# syncs the FLOP-scaling shards (and, once, the Qwen3-4B base) from S3 to weka, where the Beaker
# training jobs read them. Validated template shape from beaker.md.
#   bash debug/flop_scaling/sync_s3_to_weka.sh            # all flop_scaling shards
#   ONLY="outlier_sh8M outlier_sh16M" bash debug/flop_scaling/sync_s3_to_weka.sh
set -uo pipefail
S3=s3://ai2-llm/checkpoints/prasanns/flop_scaling/shards
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/flop_scaling/shards
ONLY="${ONLY:-}"
if [ -n "$ONLY" ]; then
  WORK="mkdir -p $WEKA; for a in $ONLY; do aws s3 sync $S3/\$a $WEKA/\$a; done; ls -la $WEKA"
else
  BASE_S3=s3://ai2-llm/checkpoints/prasanns/flop_scaling/bases; BASE_WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/flop_scaling/bases
  WORK="mkdir -p $WEKA $BASE_WEKA; aws s3 sync $BASE_S3/ $BASE_WEKA/; ls $BASE_WEKA/*/model_and_optim | wc -l; aws s3 sync $S3/ $WEKA/; ls -la $WEKA; for d in $WEKA/*/; do echo \$d \$(ls \$d | wc -l) files; done"
fi
gantry run --name "fs-sync-s3-weka-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/neptune*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "export AWS_PROFILE=S3; $WORK"
