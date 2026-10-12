#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=mixersplit
#SBATCH --output=/data/prasann/logs/mixersplit_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
$PY -u research/kv_route/flexcompute_40x/bench_mixer_split.py --scale 4B  --tokens 8192,16384
$PY -u research/kv_route/flexcompute_40x/bench_mixer_split.py --scale 0.8B --tokens 8192,32768,65536
echo "=== DONE ==="
