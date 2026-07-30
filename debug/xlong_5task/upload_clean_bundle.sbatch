#!/bin/bash
# Push the newly built CLEAN rungs (cubbins /data) into the clean v2 bundle prefix on S3.
#
# Part 1 of 3 in assembling _eval_bundle_eval500_v2_clean:
#   1. (this) upload the rungs that only exist locally: contra 64k..2M pubmed-only, and
#      nq/outlier/rerank 64k/128k/256k rebuilt at eval_size=500.
#   2. assemble_clean_bundle.sh  -- server-side S3 copies for everything already clean on S3
#      (base 2k-32k rungs, nq/outlier/rerank 512k/1M/2M, all oolong rungs, the OOD probes).
#   3. weka_sync_clean_bundle.sh -- S3 -> weka, then re-audit.
#
# A NEW bundle rather than an in-place fix of _eval_bundle_eval500_v2, because that bundle is being
# read by eval jobs right now and the corrected rungs have DIFFERENT `n` in their filenames (the
# filler pool changed) -- writing them alongside the old ones would leave two files for one size and
# let sorted()[0] silently choose the rung.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=xl5-clean-up
#SBATCH --output=/data/prasann/xlong5/logs/clean_upload_%A.log
set -uo pipefail
export AWS_PROFILE=S3
SRC=/data/prasann/xlong5/eval_ultra_pubmed
S3=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
fail=0
for t in contra nq outlier rerank; do
  [ -d "$SRC/$t" ] || { echo "!! missing $SRC/$t"; fail=1; continue; }
  echo "--- $t ---"
  for f in "$SRC/$t"/*.jsonl; do
    n=$(wc -l < "$f")
    printf "    %-58s eval_size=%-5s %13s bytes\n" "$(basename "$f")" "$n" "$(stat -c %s "$f")"
    [ "$n" -ge 500 ] || { echo "      ^ BELOW the 500 floor -- refusing to stage"; fail=1; }
  done
done
[ "$fail" = 1 ] && { echo "ABORT: a rung is short or missing"; exit 3; }

for t in contra nq outlier rerank; do
  echo "=== upload $t -> $S3/$t ==="
  # .manifest.json sidecars go too: they pin which corpora each rung was built from.
  aws s3 sync "$SRC/$t" "$S3/$t" --only-show-errors
  echo "  rc=$? remote=$(aws s3 ls "$S3/$t/" | wc -l) objects"
done
echo "=== done $(date '+%F %T') ==="
