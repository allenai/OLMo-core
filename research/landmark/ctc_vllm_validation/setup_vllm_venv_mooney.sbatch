#!/bin/bash
#SBATCH --job-name=ctc-vllm-venv-mooney
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/data/prasann/joblogs/ctc_vllm_venv_mooney_%j.log
set -uo pipefail
mkdir -p /data/prasann/joblogs
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
bash $REPO_ROOT/research/landmark/ctc_vllm_validation/setup_vllm_venv.sh
