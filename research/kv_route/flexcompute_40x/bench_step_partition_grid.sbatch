#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --job-name=steppart_grid
#SBATCH --output=/data/prasann/logs/steppart_grid_%j.log
# Fills in the rest of the (scale x context) grid so the ceiling table is measured rather than
# extrapolated. Rows that do not fit raise OOM, which the script catches and records -- an OOM row
# is a real answer about what a single H200 can hold, not a failure to be worked around.
#
# Both bases are run wherever they fit: non-AC (comparable to the 0.8B/4B numbers already in the
# README) and AC (the only basis 27B and the long-context rows exist on at all).
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

# short contexts: cheap, and the regime where the FLOP model is least trustworthy
for S in 0.8B 4B 9B 27B; do
  echo "### $S short ac"   && $PY -u $B --scale "$S" --ac --tokens 2048,4096 --suffix _short
done
for S in 0.8B 4B 9B; do
  echo "### $S short noac" && $PY -u $B --scale "$S" --tokens 2048,4096 --suffix _short
done

# long contexts: fewer iterations, they are seconds per step
for S in 0.8B 4B 9B 27B; do
  echo "### $S long ac"    && $PY -u $B --scale "$S" --ac --tokens 32768,65536,131072 \
      --iters 4 --suffix _long
done
for S in 0.8B 4B; do
  echo "### $S long noac"  && $PY -u $B --scale "$S" --tokens 131072 --iters 4 --suffix _long
done
echo "=== DONE ==="
