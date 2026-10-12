#!/bin/bash
#SBATCH --job-name=vllm-smoke-v3
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data/prasann/joblogs/vllm_smoke_v3_%j.log
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
/data/prasann/ctc_vllm_venv/bin/python -u \
  $REPO_ROOT/research/landmark/ctc_vllm_validation/smoke_load_v3.py
echo "=== DONE $(date '+%F %T') ==="
