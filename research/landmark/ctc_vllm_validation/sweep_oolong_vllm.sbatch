#!/bin/bash
#SBATCH --job-name=vllm-oolong-4b-sweep
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/sweep_oolong_%j.log
set -uo pipefail
export TASK=oolong
export EVAL_TASK_DIR=oolong
export CKPT=/data/prasann/ctc_suite/ckpts/ctc-4b-oolong-full
export RUNGS="2048 8192 32768 4096 16384"
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
bash $REPO_ROOT/research/landmark/ctc_vllm_validation/sweep_task_vllm.sh
