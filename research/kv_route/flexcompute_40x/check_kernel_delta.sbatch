#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --job-name=kerneldelta
#SBATCH --output=/data/prasann/logs/kerneldelta_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
cd "$REPO"
/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -u research/kv_route/flexcompute_40x/check_kernel_delta.py
echo "=== DONE ==="
