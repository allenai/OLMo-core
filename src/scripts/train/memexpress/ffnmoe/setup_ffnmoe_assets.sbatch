#!/bin/bash
# One-time asset setup for the ffnmoe family on a node: stage the contradiction shard onto the
# node's /data and bake the router keys (+ importance permutation) into a base checkpoint copy.
#
# CPU-only (no --gres): the bake loads a 4B model on CPU, permutes hidden units, creates the
# router/gain keys and re-saves. ~12GB read + ~12GB write of node-local disk.
#
#   sbatch -w sneetches -p jsteinhardt -q preemptive_high --export=ALL setup_ffnmoe_assets.sbatch
#SBATCH --job-name=ffnmoe-setup
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/ffnmoe_setup_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/ffnmoe_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/ffnmoe_setup_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$MIRROR_DIR/setup_${SLURM_JOB_ID}.log" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) & _MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
SRC_ROOT="${SRC_ROOT:-/data/prasann/pooledkv_exp}"
DST_ROOT="${DST_ROOT:-/data/prasann/ffnmoe_exp}"
SRC_BASE="${SRC_BASE:-$SRC_ROOT/q4b-dense-cpt-fixmark/model_and_optim}"
SRC_DATA="${SRC_DATA:-$SRC_ROOT/contra_n100_qboth_train}"
START_LAYER="${START_LAYER:-4}"
DIVISORS="${DIVISORS:-1,4,16,64}"
PERMUTE="${PERMUTE:-weight}"
WIDTH_MULTIPLE="${WIDTH_MULTIPLE:-8}"
SRC_START_LAYER="${SRC_START_LAYER:--1}"   # >=0: source already has trained routers from this layer (two-stage warm start)

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$SRC_BASE/.metadata" ] || { echo "FATAL: source base $SRC_BASE missing .metadata"; exit 2; }
[ -f "$SRC_DATA/metadata.json" ] || { echo "FATAL: source shard $SRC_DATA missing"; exit 3; }

mkdir -p "$DST_ROOT/runs" "$DST_ROOT/eval_results" "$DST_ROOT/dataset-cache" /data/prasann/joblogs
# 19MB shard -- copied so the family is self-contained rather than reading pooledkv's assets.
if [ ! -f "$DST_ROOT/$(basename "$SRC_DATA")/metadata.json" ]; then
  cp -r "$SRC_DATA" "$DST_ROOT/" && echo "staged shard -> $DST_ROOT/$(basename "$SRC_DATA")"
else
  echo "shard already staged"
fi

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

OUT="$DST_ROOT/q4b-dense-cpt-fixmark-ffnmoe${OUT_SUFFIX:-}"   # OUT_SUFFIX=-L0 for a start-layer-0 bake
if [ "${REBUILD:-0}" = "1" ] && [ -d "$OUT" ]; then echo "REBUILD=1: removing $OUT"; rm -rf "$OUT"; fi
if [ -f "$OUT/model_and_optim/.metadata" ]; then
  echo "baked base already exists at $OUT -- delete it to rebuild"
else
  echo "=== baking router keys (start_layer=$START_LAYER divisors=$DIVISORS permute=$PERMUTE) ==="
  cd "$REPO"
  python src/scripts/train/memexpress/ffnmoe/bake_ffn_moe_into_base.py \
    --base "$SRC_BASE" --out "$OUT" \
    --start-layer "$START_LAYER" --divisors "$DIVISORS" --permute "$PERMUTE" --src-start-layer "$SRC_START_LAYER" --width-multiple "$WIDTH_MULTIPLE"
  RC=$?
  [ "$RC" -eq 0 ] || { echo "FATAL: bake failed rc=$RC"; exit "$RC"; }
fi

echo "=== ASSETS ==="
ls -la "$OUT/model_and_optim" | head -5
du -sh "$OUT" "$DST_ROOT/$(basename "$SRC_DATA")"
echo "=== DONE $(date '+%F %T') ==="
