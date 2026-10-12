#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=ffnmoearm
#SBATCH --output=/data/prasann/logs/ffnmoearm_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
echo "=== SMOKE 0.8B/8192 ==="
$PY -u research/kv_route/flexcompute_40x/ffn_moe_arm.py --tokens 8192 --scale 0.8B --ks 8 --tag smoke || exit 1
for S in 0.8B 4B 9B 27B; do
  echo "=== $S / 32768 ==="
  $PY -u research/kv_route/flexcompute_40x/ffn_moe_arm.py --tokens 32768 --scale "$S" --ks 8,2
done
echo "=== DONE ==="
