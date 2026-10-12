#!/bin/bash
#SBATCH -J vllm-canon-smoke
#SBATCH --output=/data/prasann/joblogs/smoke_canon_%j.log
#SBATCH --time=00:12:00
set -x
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python $REPO_ROOT/research/landmark/ctc_vllm_validation/smoke_canonical.py
