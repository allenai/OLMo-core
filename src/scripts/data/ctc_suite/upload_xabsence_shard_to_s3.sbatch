#!/bin/bash
# Relay step 1/2: push the xabsence 4B train shard from cubbins node-local /data to S3.
# Step 2 (S3 -> weka) is a gantry job run from the login host (see task notes / beaker.md).
# Runs on cubbins where the shard + the S3 AWS profile live. CPU-only.
#
#   sbatch src/scripts/data/ctc_suite/upload_xabsence_shard_to_s3.sbatch
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=ctc-xabs-s3
#SBATCH --output=/data/prasann/joblogs/ctc_xabsence_s3_%A.log

set -uo pipefail
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/ctc_suite
SRC=/data/prasann/ctc_suite/shards_final/xabsence_train
SCRATCH_LOGDIR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$SCRATCH_LOGDIR"
SLURM_LOG=/data/prasann/joblogs/ctc_xabsence_s3_${SLURM_JOB_ID}.log
MIRROR=$SCRATCH_LOGDIR/ctc_xabsence_s3_${SLURM_JOB_ID}.log
( while true; do cp -f "$SLURM_LOG" "$MIRROR" 2>/dev/null; sleep 10; done ) &
MPID=$!
trap 'cp -f "$SLURM_LOG" "$MIRROR" 2>/dev/null; kill $MPID 2>/dev/null' EXIT

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -d "$SRC" ] || { echo "MISSING SHARD DIR: $SRC"; exit 2; }
echo "--- local shard listing ---"; ls -la "$SRC"

echo "--- aws s3 sync xabsence_train ---"
aws s3 sync "$SRC" "$S3/shards/xabsence_train" \
  --exclude "*" --include "token_ids_part_*.npy" --include "labels_mask_*.npy" --include "metadata.json"
echo "sync rc=$?"

echo "=== REMOTE LISTING ==="
aws s3 ls "$S3/shards/xabsence_train/"
echo "=== DONE $(date '+%F %T') ==="
