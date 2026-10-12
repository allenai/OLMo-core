#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --job-name=jointsweep
#SBATCH --output=/data/prasann/logs/jointsweep_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
cd "$REPO"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
$PY -u research/kv_route/flexcompute_40x/bench_joint_sweep.py \
  --scale "${SCALE:-4B}" --ctx "${CTX:-16384}" --rounds "${ROUNDS:-3}" \
  --iters "${ITERS:-5}" --phases "${PHASES:-matrix,partition,floor,graph}" \
  --arms "${ARMS:-all}" ${PROFILE:+--profile} ${KERNELS:+--kernels} \
  ${DOCS:+--docs "$DOCS"} ${AC:+--ac} ${EXTRA:-} \
  --out "research/kv_route/flexcompute_40x/joint_sweep_${SCALE:-4B}_${CTX:-16384}${TAG:-}.json"
echo "=== DONE ==="
