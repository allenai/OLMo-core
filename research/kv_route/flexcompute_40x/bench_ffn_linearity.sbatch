#!/bin/bash
# Does routed-FFN wall-clock scale linearly with routed-FFN FLOPs? Sweep the cost fraction.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=ffnlinear
#SBATCH --output=/data/prasann/logs/ffnlinear_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
# Sweep tokens too: if the floor is launch/bandwidth bound it shrinks (relatively) with more tokens.
for T in 2048 8192 32768; do
  echo "=== 4B tokens=$T ==="
  $PY -u research/kv_route/flexcompute_40x/bench_ffn_linearity.py --scale 4B --tokens $T
done
echo "=== DONE ==="
