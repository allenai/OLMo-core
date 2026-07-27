#!/bin/bash
# Push the node-local llama shards from the tokenizing node to shared /scratch so another node can
# stage them. Shards are ~1-2GB; /scratch writes are slow (~5MB/s) but this is a one-time cost.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama-stage-out
#SBATCH --output=/data/prasann/joblogs/llama_stage_out_%j.log
set -uo pipefail
DST=/scratch/users/prasann/ctc_llama/shards
mkdir -p "$DST"
for d in /data/prasann/ctc_llama/shards/*/; do
  n=$(basename "$d")
  echo "copying $n ..."
  mkdir -p "$DST/$n"
  cp -f "$d"/* "$DST/$n"/ && echo "  ok $(du -sh "$DST/$n" | cut -f1)"
done
ls -la "$DST"/*/ | head -40
echo "=== DONE $(date '+%F %T') ==="
touch /scratch/users/prasann/ctc_llama_logs/STAGE_OUT_DONE
