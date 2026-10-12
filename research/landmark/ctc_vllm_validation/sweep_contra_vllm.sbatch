#!/bin/bash
#SBATCH --job-name=vllm-contra-4b-sweep
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive_high_sewonm
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/sweep_contra_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
bash $REPO_ROOT/research/landmark/ctc_vllm_validation/sweep_contra_vllm.sh
