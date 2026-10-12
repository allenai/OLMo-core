#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --job-name=steppart_fill
#SBATCH --output=/data/prasann/logs/steppart_fill_%j.log
# The remaining non-AC cells, so the primary (non-AC) basis covers as much of the grid as a single
# H200 can hold: 4B/9B at the training contexts, 0.8B at 16k.
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/data/prasann/tmp
mkdir -p "$TMPDIR"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
B=research/kv_route/flexcompute_40x/bench_step_partition.py
echo "### 0.8B mid noac" && $PY -u $B --scale 0.8B --tokens 16384 --suffix _fill
echo "### 4B mid noac"   && $PY -u $B --scale 4B   --tokens 32768,65536 --iters 5 --suffix _fill
echo "### 9B mid noac"   && $PY -u $B --scale 9B   --tokens 16384,32768 --iters 5 --suffix _fill
echo "### 27B noac"      && $PY -u $B --scale 27B  --tokens 8192 --iters 4 --suffix _fill
echo "=== DONE ==="
