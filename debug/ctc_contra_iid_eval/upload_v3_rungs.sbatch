#!/bin/bash
#SBATCH --job-name=v3-rungs-s3
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/v3_rungs_s3_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home AWS_PROFILE=S3
echo "=== [v3-rungs-s3] START $(date '+%F %T') ==="
echo "--- contra v3 (base 2k-32k + xlong 64k-1M) ---"
du -sh /data/prasann/contra_iid_eval/v3_hub /data/prasann/contra_iid_eval/v3_xlong 2>/dev/null
echo "--- outlier scale-K (3k-1M) ---"
du -sh /data/prasann/outlier_xlong_scalek/outlier 2>/dev/null
for spec in "/data/prasann/contra_iid_eval/v3_hub:contra" \
            "/data/prasann/contra_iid_eval/v3_xlong:contra" \
            "/data/prasann/outlier_xlong_scalek/outlier:outlier"; do
  SRC="${spec%%:*}"; SUB="${spec##*:}"
  [ -d "$SRC" ] || { echo "!! missing $SRC"; continue; }
  echo "[$(date '+%T')] syncing $SRC -> $SUB"
  aws s3 sync "$SRC" "s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3/$SUB" \
    --exclude "*manifest*" --only-show-errors || echo "!! sync failed for $SRC"
  echo "[$(date '+%T')] done $SUB"
done
echo "--- published ---"
for t in contra outlier; do
  echo "[$t]"; aws s3 ls "s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3/$t/" | awk '{printf "  %12s %s\n",$3,$4}'
done
echo "=== [v3-rungs-s3] DONE $(date '+%F %T') ==="
