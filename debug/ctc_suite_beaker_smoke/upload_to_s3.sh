#!/bin/bash
set -uo pipefail
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/ctc_suite
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="

echo "--- outlier_train shard ---"
aws s3 sync /data/prasann/ctc_suite/shards/outlier_train "$S3/shards/outlier_train" --exclude "*" --include "token_ids_part_*.npy" --include "labels_mask_*.npy" --include "metadata.json"
echo "rc=$?"

echo "--- grouping_train shard ---"
aws s3 sync /data/prasann/ctc_suite/shards/grouping_train "$S3/shards/grouping_train" --exclude "*" --include "token_ids_part_*.npy" --include "labels_mask_*.npy" --include "metadata.json"
echo "rc=$?"

echo "--- 4b base ---"
aws s3 sync /data/prasann/ctc_suite/bases/q35-4b-base-modelonly "$S3/bases/q35-4b-base-modelonly"
echo "rc=$?"

echo "=== REMOTE LISTING ==="
aws s3 ls "$S3/shards/outlier_train/"
aws s3 ls "$S3/shards/grouping_train/"
aws s3 ls "$S3/bases/q35-4b-base-modelonly/model_and_optim/" | head
echo "=== DONE $(date '+%F %T') ==="
