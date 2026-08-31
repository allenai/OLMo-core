#!/bin/bash
# Build the oracle log-mass slot cache (see build_oracle_slot_cache.py) on 2 GPUs, one builder
# rank per GPU over disjoint instance stripes. NO full-context forwards -- rows are ~150 tokens.
#
#   sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
#     --export=ALL,DATA=/data/prasann/pooledkv_exp/contra32k_train,OUT=/data/prasann/pooledkv_exp/oracle_cache_contra32k \
#     build_oracle_cache.sbatch
#
# Optional: EXTRA="--limit-instances 4" for a smoke run.
#SBATCH --job-name=oracle-cache
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/oracle_cache_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/pooledkv_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/oracle_cache_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$MIRROR_DIR/oracle_cache_${SLURM_JOB_ID}.log" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) & _MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
DATA="${DATA:-/data/prasann/pooledkv_exp/contra32k_train}"
OUT="${OUT:-/data/prasann/pooledkv_exp/oracle_cache_contra32k}"
EXTRA="${EXTRA:-}"
mkdir -p "$OUT"

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
echo "=== HOST=$(hostname) DATA=$DATA OUT=$OUT EXTRA='$EXTRA' GPUS=${CLEAN[*]:0:2} START=$(date '+%F %T') ==="

cd "$REPO"
CUDA_VISIBLE_DEVICES="${CLEAN[0]}" python src/scripts/train/memexpress/pooledkv/build_oracle_slot_cache.py \
  --data "$DATA" --out "$OUT" --rank 0 --world 2 $EXTRA > "$OUT/build_rank0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES="${CLEAN[1]}" python src/scripts/train/memexpress/pooledkv/build_oracle_slot_cache.py \
  --data "$DATA" --out "$OUT" --rank 1 --world 2 $EXTRA > "$OUT/build_rank1.log" 2>&1 &
P1=$!
( while true; do sleep 60; tail -2 "$OUT/build_rank0.log" 2>/dev/null; done ) & _TAIL_PID=$!
wait $P0; RC0=$?
wait $P1; RC1=$?
kill $_TAIL_PID 2>/dev/null
tail -5 "$OUT/build_rank0.log" "$OUT/build_rank1.log"
echo "=== DONE rc0=$RC0 rc1=$RC1 $(date '+%F %T') ==="
