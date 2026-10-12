#!/bin/bash
# Wall-clock anatomy of a training step: ablate FFN / mixer / head and time what changes.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=ffnshare
#SBATCH --output=/data/prasann/logs/ffnshare_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
echo "=== 4B ==="
$PY -u research/kv_route/flexcompute_40x/bench_ffn_share.py --scale 4B --tokens 2048,8192,16384
echo "=== 0.8B (wider sweep: bandwidth-vs-compute hypothesis) ==="
$PY -u research/kv_route/flexcompute_40x/bench_ffn_share.py --scale 0.8B --tokens 2048,8192,32768,65536
echo "=== DONE ==="
