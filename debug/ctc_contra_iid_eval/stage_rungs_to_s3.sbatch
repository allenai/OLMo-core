#!/bin/bash
# Move the IID rungs mooney -> cubbins via S3.
#
# The rungs are generated on mooney (that node holds the Qwen2.5-7B generator cache and the
# training file), but the contradiction serving copies live only on cubbins. /net is the slow NFS
# layer and is explicitly not a way to share data between nodes, so this follows the same S3
# two-step prep_task.sh uses for checkpoints. ~270MB total.
#SBATCH --job-name=contra-iid-stage
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/contra_iid_stage_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export AWS_PROFILE=S3
SRC=/data/prasann/contra_iid_eval/rungs
DST=s3://ai2-llm/checkpoints/prasanns/_transfer/contra_iid_rungs

echo "=== [stage] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$SRC" || { echo "!!! nothing to stage at $SRC"; exit 1; }

aws s3 sync "$SRC" "$DST" --only-show-errors || { echo "!!! s3 sync FAILED"; exit 1; }
# also carry the provenance so the numbers stay self-describing
for f in /data/prasann/contra_iid_eval/disjointness_report.json \
         /data/prasann/contra_iid_eval/iid_rung_audit.json; do
  [ -f "$f" ] && aws s3 cp "$f" "$DST/$(basename "$f")" --only-show-errors
done

echo "--- staged ---"
aws s3 ls "$DST/"
echo "=== [stage] DONE $(date '+%F %T') ==="
