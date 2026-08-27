#!/bin/bash
# Stage the length-mix experiment data (arm shards + per-rung heldout CE shards + eval ladder)
# from mooney /data to S3, for the gantry S3->weka sync.
#SBATCH --job-name=outlier-lm-s3
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_lm_s3_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home
export AWS_PROFILE=S3
WORK=/data/prasann/outlier_lengthmix
DST=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix

echo "=== [lm-s3] START $(date '+%F %T') ==="
[ -s "$WORK/arms_tokenized/MANIFEST.json" ] || { echo "!!! arms not tokenized"; exit 1; }
aws s3 sync "$WORK/arms_tokenized" "$DST/arms" --only-show-errors || exit 1
for N in 14 28 57 111; do
  aws s3 sync "$WORK/tokenized/n${N}_heldout" "$DST/heldout/n${N}" --only-show-errors || exit 1
done
aws s3 sync "$WORK/eval_rungs/outlier/outlier" "$DST/eval_rungs/outlier" --only-show-errors || exit 1
echo "--- published ---"; aws s3 ls --recursive "$DST/" | tail -5
echo "=== [lm-s3] DONE $(date '+%F %T') ==="
