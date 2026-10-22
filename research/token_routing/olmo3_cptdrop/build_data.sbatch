#!/bin/bash
# CPU data build for olmo3_cptdrop: longmino pool -> train32k / dev32k / short4k shards on the chosen node's /data.
#   sbatch -w <node> research/token_routing/olmo3_cptdrop/build_data.sbatch   (all three; pick the node with where-to-run)
#   ONLY=dev,short EXTRA="--dev-rows 1024 --dev-name dev32k_1024" sbatch ...   (subset; train ids read back)
#SBATCH --job-name=o3cd-data
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -euo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ROOT=/data/prasann/olmo3_cptdrop
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
mkdir -p "$ROOT/data" "$ROOT/hf_cache" "$ROOT/logs"
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp" TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=32
mkdir -p "$TMPDIR"
# node-local copy of the (stdlib + numpy + transformers) builder: no repo imports, nothing else read from NFS
cp "$REPO/research/token_routing/olmo3_cptdrop/build_pool_data.py" "$ROOT/build_pool_data.py"
OPS_RUN_ID="o3cd-data-${SLURM_JOB_ID}" OPS_EXPERIMENT=olmo3_cptdrop OPS_KIND=data \
OPS_EXPECTED_MINUTES=60 OPS_FIRST_OUTPUT=600 OPS_STALL=1200 \
bash "$REPO/recipes/env/ops_job.sh" \
  "$PY" -u "$ROOT/build_pool_data.py" --out-root "$ROOT/data" --cache "$ROOT/hf_cache" \
  --only "${ONLY:-train,dev,short}" ${EXTRA:-}
