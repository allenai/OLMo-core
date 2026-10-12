#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=gdngap
#SBATCH --output=/data/prasann/logs/gdngap_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
cd "$REPO"
/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -u research/kv_route/flexcompute_40x/resolve_gdn_gap.py
echo "=== DONE ==="
