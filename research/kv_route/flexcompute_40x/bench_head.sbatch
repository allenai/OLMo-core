#!/bin/bash
# Measure the LM-head gating win on one H200. Head microbenchmark across context lengths, then a
# full-model step at the sizes that fit without activation checkpointing.
#
#   sbatch --nodelist=cubbins research/kv_route/flexcompute_40x/bench_head.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=headgate-bench
#SBATCH --output=/data/prasann/logs/headgate_%j.log

set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
mkdir -p "${LOG_DIR:-/data/prasann/logs}"
source /data/prasann/node_local_env.sh
# The edits under test live in the NFS checkout, so point at that rather than the node-local
# mirror -- a ~60 s import tax, paid once, in exchange for benchmarking the code we just changed.
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

cd "$REPO"
echo "=== node $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== HEAD MICROBENCHMARK (4B geometry) ==="
$PY -u research/kv_route/flexcompute_40x/bench_head.py --bench head --scale 4B \
    --lengths 2048,8192,32768,65536 \
    --out research/kv_route/flexcompute_40x/bench_head_4B.json

echo "=== FULL MODEL STEP (0.8B) ==="
$PY -u research/kv_route/flexcompute_40x/bench_head.py --bench model --scale 0.8B \
    --lengths 2048,8192,32768 \
    --out research/kv_route/flexcompute_40x/bench_model_0.8B.json

echo "=== FULL MODEL STEP (4B) ==="
$PY -u research/kv_route/flexcompute_40x/bench_head.py --bench model --scale 4B \
    --lengths 2048,8192 \
    --out research/kv_route/flexcompute_40x/bench_model_4B.json

echo "=== DONE $(date) ==="
