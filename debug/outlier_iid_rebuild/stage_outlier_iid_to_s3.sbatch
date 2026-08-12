#!/bin/bash
# Publish the iid-rebuilt outlier SFT shard to S3 (weka sync follows via gantry).
#
# The rebuild lives only on mooney's node-local disk. Named single_task_ladders_v3/outlier so it is
# a drop-in for the v2 path every 5-task launcher already points at
# (DATA_ROOT=.../single_task_ladders_v2, OUTLIER_DATA_ROOT=$DATA_ROOT/outlier).
#
# WHAT CHANGED vs single_task_ladders_v2/outlier: majority runs are now built the way the v2 EVAL
# ladder builds them (--majority-mode articles: whole articles of U[4,14] chunks until full, K
# emergent) instead of sampling K then partitioning at 25 chunks/article. Verified K = 3/7/13/24 vs
# the eval's 3/7/13/25, docs-per-topic 8.3-8.9 vs 8.7-9.5. Same tokenizer/window/query_position as
# the shard it replaces (Qwen3-0.6B, 40960, both), so it drops straight in.
#
# ⚠ Anything trained on this needs the outlier eval's generation budget raised from 200 to 512:
# the target enumerates every majority topic and the doc IDs come LAST, so at n~220 the answer is
# ~160 tokens median / 182 max and truncates before the IDs. See
# records/contradiction-train-eval-non-iid.md §2b.
#SBATCH --job-name=outlier-iid-s3
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_iid_s3_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export AWS_PROFILE=S3
SRC=/data/prasann/outlier_iid/tokenized
DST=s3://ai2-llm/checkpoints/prasanns/single_task_ladders_v3/outlier

echo "=== [outlier-iid-s3] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$SRC" || { echo "!!! shard missing at $SRC"; exit 1; }
for f in token_ids_part_000000.npy labels_mask_000000.npy metadata.json; do
  [ -f "$SRC/$f" ] || { echo "!!! missing $f"; exit 1; }
done
cat "$SRC/metadata.json"

aws s3 sync "$SRC" "$DST" --only-show-errors || { echo "!!! s3 sync FAILED"; exit 1; }
echo "--- published ---"
aws s3 ls "$DST/"
echo "=== [outlier-iid-s3] DONE $(date '+%F %T') ==="
