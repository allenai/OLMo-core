#!/bin/bash
#SBATCH --job-name=vllm-hotpotqa-4b-sweep
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/sweep_hotpotqa_%j.log
set -uo pipefail
export TASK=hotpotqa
export EVAL_TASK_DIR=hotpotqa
export CKPT=/data/prasann/ctc_suite/ckpts/ctc-4b-hotpotqa-full
export RUNGS="2048 8192 4096 16384"
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
bash $REPO_ROOT/research/landmark/ctc_vllm_validation/sweep_task_vllm.sh
