#!/bin/bash
# Publish the contradiction v3 (train/eval IID) rungs to S3 as _eval_bundle_eval500_v3/contra/.
#
# v3 is a ONE-TASK bundle on purpose. Only contradiction differs from v2; nq / outlier / oolong /
# rerank were audited 2026-08-11 and are already in-distribution, so eval_lc_native.py reads those
# four from the v2 bundle and only redirects contradiction here (E5_CONTRA). Duplicating the v2
# bundle's multi-GB xlong files to make v3 "complete" would buy nothing.
#
# After this lands, weka still needs a gantry S3->weka sync or every rung reads MISSING on Beaker
# (the eval-bundle staging trap). This job only does the S3 half.
#SBATCH --job-name=contra-v3-stage
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/contra_v3_stage_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export AWS_PROFILE=S3
SRC=/data/prasann/contra_iid_eval/v3_hub
DST=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3/contra

echo "=== [contra-v3-stage] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$SRC" || { echo "!!! nothing to stage at $SRC"; exit 1; }

for n in 100 190 385 765; do
  f="$SRC/contradiction_eval_pubmed_realistic_n${n}_k3.jsonl"
  [ -f "$f" ] || { echo "!!! missing rung n=$n"; exit 1; }
  lines=$(wc -l < "$f")
  [ "$lines" -ge 500 ] || { echo "!!! rung n=$n has only $lines rows (<500 floor)"; exit 1; }
  echo "  n=$n rows=$lines OK"
done

aws s3 sync "$SRC" "$DST" --only-show-errors || { echo "!!! s3 sync FAILED"; exit 1; }

# provenance travels with the data -- these are what make the numbers defensible
for f in /data/prasann/contra_iid_eval/disjointness_report.json \
         /data/prasann/contra_iid_eval/iid_rung_audit.json; do
  [ -f "$f" ] && aws s3 cp "$f" "$DST/$(basename "$f")" --only-show-errors
done

echo "--- published ---"
aws s3 ls "$DST/"
echo "=== [contra-v3-stage] DONE $(date '+%F %T') ==="
