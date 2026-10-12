#!/bin/bash
# One-off: push the Qwen3-4B CPT+fixmark base (the base both methods were developed on) from
# sneetches node-local /data to S3; sync_s3_to_weka.sh then lands it on weka for Beaker.
#SBATCH --job-name=fs-base-s3
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/fs_base_s3_%j.log
set -uo pipefail
export AWS_PROFILE=S3
SRC=/data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark
S3=s3://ai2-llm/checkpoints/prasanns/flop_scaling/bases/q4b-dense-cpt-fixmark
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="; du -sh $SRC
aws s3 sync "$SRC/model_and_optim" "$S3/model_and_optim" && echo "rc=$?"
aws s3 ls "$S3/model_and_optim/" | head -5; aws s3 ls "$S3/model_and_optim/" | wc -l
echo "=== DONE $(date '+%F %T') ==="
