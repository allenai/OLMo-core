#!/bin/bash
# Step 1/2 of Beaker staging for the 512k/1M/2M ULTRA rungs: cubbins node-local /data -> S3.
# Step 2 (S3 -> weka) is debug/xlong_5task/weka_sync_ultra.sh, a gantry job from the login host.
# An S3 push ALONE is not enough: weka is unreachable from Berkeley, so every rung would log
# MISSING at eval time while the job still exits 0.
#
# The ultra rungs land in the SAME eval/<task>/ directories as the 64k/128k/256k rungs, because
# eval_lc_native.py resolves them by size-labelled glob (*_xlong_{size}.jsonl) under EVAL500_ROOT.
# Keeping them together is what makes --xlong pick up the full 64k..2M ladder in one run.
#
# ~37 GB apparent (ZFS compresses it to ~22 GB on disk, so `du` under-reports what S3 receives).
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=xl5-ultra-s3
#SBATCH --output=/data/prasann/xlong5/logs/ultra_s3_%A.log
set -uo pipefail
export AWS_PROFILE=S3
SRC=/data/prasann/xlong5/eval_ultra
S3=s3://ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/eval

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
echo "--- what we are about to push (apparent bytes, not du) ---"
tot=0
for f in "$SRC"/*/*.jsonl; do
  b=$(stat -c %s "$f"); tot=$((tot+b))
  printf "  %-60s %13d bytes  lines=%s\n" "${f#$SRC/}" "$b" "$(wc -l < "$f")"
done
echo "  TOTAL apparent: $tot bytes (~$((tot/1024/1024/1024)) GiB)"

# Sanity gate: every rung must clear the >=500 eval_size floor before it goes anywhere.
short=0
for f in "$SRC"/*/*.jsonl; do
  n=$(wc -l < "$f")
  [ "$n" -lt 500 ] && { echo "  ⚠ ${f#$SRC/} has only $n examples (<500 floor)"; short=1; }
done
[ "$short" = 1 ] && { echo "ABORT: a rung is below the 500 eval_size floor"; exit 3; }
echo "  all rungs >= 500 examples"

for t in contra nq outlier rerank oolong; do
  [ -d "$SRC/$t" ] || { echo "!! missing $SRC/$t"; continue; }
  echo "=== sync $t ==="
  aws s3 sync "$SRC/$t" "$S3/$t" --only-show-errors
  echo "  rc=$?  remote now: $(aws s3 ls "$S3/$t/" | wc -l) objects"
done
echo "=== done $(date '+%F %T') ==="
