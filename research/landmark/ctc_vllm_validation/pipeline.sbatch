#!/bin/bash
#SBATCH --job-name=ctc-vllm-parity
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/ctc_vllm_parity_%j.log
set -uo pipefail
mkdir -p /data/prasann/joblogs
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
bash $REPO_ROOT/research/landmark/ctc_vllm_validation/pipeline.sh
