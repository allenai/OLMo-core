#!/bin/bash
# FLOP-scaling study data build (records/flop-scaling-ffn-kv-plan.md §3), one task per job:
#   1. per-rung TRAIN pools via the ctc-data CLI (fixed doc count per rung, --pool auto seed pools;
#      the HF seed pools were prefetched on the login node into the NFS HF cache -> offline here),
#   2. short-heavy nested arms at the token budgets (compose_shortheavy_arms.py),
#   3. tokenize each arm for Qwen3 (marker-wrapped dense shards, query AFTER, no-cot) with the
#      gold sidecar the soft-token arm needs,
#   4. upload the shards to S3 (the weka sync is a separate gantry job: sync_s3_to_weka.sh).
#
# CPU only. Runs on mooney's node-local /data (NFS-safe: logs + work on /data).
#   TASK=outlier sbatch -w mooney debug/flop_scaling/build_shortheavy_data.sbatch
#SBATCH --job-name=fs-data
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/fs_data_%x_%j.log
set -uo pipefail
TASK="${TASK:?set TASK=outlier|contradiction|nq|oolong}"
BUDGETS="${BUDGETS:-8M,16M,32M,64M,128M}"
# Pool sizes per rung: what the LARGEST budget needs (+10% margin). 128M short-heavy =
# 45%/2048 = 28.1k at 2k, 8.4k at 4k, 2.5k at 8k, 625 at 16k, 156 at 32k.
POOL_2K="${POOL_2K:-31000}"; POOL_4K="${POOL_4K:-9300}"; POOL_8K="${POOL_8K:-2800}"
POOL_16K="${POOL_16K:-700}"; POOL_32K="${POOL_32K:-180}"
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
CTC_SRC="${CTC_SRC:-/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core/ctc/src}"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
[ -x "$PY" ] || PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WORK=/data/prasann/flop_scaling
S3="${S3:-s3://ai2-llm/checkpoints/prasanns/flop_scaling/shards}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/data/prasann/tmp PYTHONWARNINGS=ignore
mkdir -p "$WORK/pools" "$WORK/arms" "$WORK/shards" "$WORK/seed_pools" /data/prasann/tmp
echo "=== HOST=$(hostname) TASK=$TASK BUDGETS=$BUDGETS START=$(date '+%F %T') ==="
# Seed pool: compute nodes have no Hub access and the login-node HF cache is node-local /tmp, so
# the pools were copied once to NFS (/scratch/users/prasann/ctc_seed_pools/); stage the one we
# need to node-local /data (read once from NFS) and pass it explicitly.
SEED_NFS="${SEED_NFS:-/scratch/users/prasann/ctc_seed_pools}"
POOL_FILE="$WORK/seed_pools/${TASK}.seed.jsonl.gz"
if [ ! -s "$POOL_FILE" ]; then
  echo "--- staging seed pool $TASK from $SEED_NFS $(date '+%T') ---"
  cp "$SEED_NFS/${TASK}.seed.jsonl.gz" "$POOL_FILE" || { echo "!!! seed pool missing at $SEED_NFS"; exit 1; }
fi
ls -la "$POOL_FILE"

# ---------- 1. per-rung pools ----------
# oolong's unified task name for the converter is oolong; nq's is retrieval (BUILD_MATRIX §1).
case "$TASK" in
  nq) CONV_TASK=retrieval; CHUNK_BY=document ;;
  oolong) CONV_TASK=oolong; CHUNK_BY=line ;;
  *) CONV_TASK=$TASK; CHUNK_BY=document ;;
esac
i=0
for R in 2k 4k 8k 16k 32k; do
  case $R in 2k) N=$POOL_2K;; 4k) N=$POOL_4K;; 8k) N=$POOL_8K;; 16k) N=$POOL_16K;; 32k) N=$POOL_32K;; esac
  OUT="$WORK/pools/${TASK}_${R}"
  if [ -s "$OUT/$TASK/train.jsonl" ] && [ "$(wc -l < "$OUT/$TASK/train.jsonl")" -ge "$N" ]; then
    echo "[skip] pool $TASK $R exists ($(wc -l < "$OUT/$TASK/train.jsonl") rows)"; continue
  fi
  echo "--- pool $TASK $R: $N examples $(date '+%T') ---"
  i=$((i+1))
  PYTHONPATH="$CTC_SRC" $PY -m ctc.data.cli build --task "$TASK" --out "$OUT" --split train \
    --rungs "$R" --train "$N" --seed $((42 + i)) --pool "$POOL_FILE" --force \
    || { echo "!!! pool build FAILED: $TASK $R"; exit 1; }
  echo "    -> $(wc -l < "$OUT/$TASK/train.jsonl") rows"
done

# ---------- 2. compose nested short-heavy arms ----------
$PY "$REPO/debug/flop_scaling/compose_shortheavy_arms.py" --task "$TASK" \
  --pools-dir "$WORK/pools" --out-dir "$WORK/arms" --budgets "$BUDGETS" || exit 1

# ---------- 3. tokenize each arm (Qwen3, marker-wrapped dense, query after, no-cot) ----------
for f in "$WORK"/arms/${TASK}_sh*.jsonl; do
  ARM=$(basename "$f" .jsonl); OUT="$WORK/shards/$ARM"
  if [ -s "$OUT/metadata.json" ] && [ -s "$OUT/gold_fingerprints.json" -o "$TASK" = oolong ]; then
    echo "[skip] shard $ARM"; continue
  fi
  echo "--- tokenizing $ARM $(date '+%T') ---"
  mkdir -p "$OUT"
  PYTHONPATH="$REPO/src" $PY "$REPO/src/scripts/data/convert_unified_to_document_landmark.py" \
    --input-jsonl "$f" --task "$CONV_TASK" --out-dir "$OUT" --emit dense --marker-set qwen3 \
    --tokenizer "$TOKENIZER" --seq-len 40960 --query-position after --cot-mode none \
    --chunk-by "$CHUNK_BY" --emit-gold-sidecar --num-proc 16 \
    || { echo "!!! tokenize FAILED: $ARM"; exit 1; }
  $PY - "$OUT" <<'PYEOF'
import json, sys
m = json.load(open(f"{sys.argv[1]}/metadata.json"))
keys = {k: m[k] for k in ("num_instances", "total_tokens", "task", "marker_set", "query_position", "cot_mode") if k in m}
print("    metadata:", keys)
PYEOF
done

# ---------- 4. upload ----------
export AWS_PROFILE=S3
for d in "$WORK"/shards/${TASK}_sh*/; do
  ARM=$(basename "$d")
  echo "--- s3 sync $ARM ---"
  aws s3 sync "$d" "$S3/$ARM" --exclude "*" --include "token_ids_part_*.npy" --include "labels_mask_*.npy" \
    --include "metadata.json" --include "gold_fingerprints.json" --include "gold_pairs.json" || echo "!!! s3 sync failed: $ARM"
done
aws s3 ls "$S3/" | grep "$TASK"
echo "=== DONE $TASK $(date '+%F %T') ==="
