#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=ffnscale
#SBATCH --output=/data/prasann/logs/ffnscale_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
for S in 0.8B 4B 9B 27B; do
  echo "########## scale $S ##########"
  $PY -u research/kv_route/flexcompute_40x/ffn_null_rung_sweep.py --scale $S --tokens 32768 || echo "FAILED $S"
done
echo "=== DONE ==="
