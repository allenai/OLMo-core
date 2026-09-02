#!/bin/bash
# Qwen3.5 KV-arm data: re-tokenize the prior campaigns' JSONL arms WITH document markers
# (marker set qwen3_5) + gold sidecars, so the soft-token arm trains on exactly the dense arms'
# examples. Local (mooney) for the arms whose JSONL lives here (outlier mix_s*, nq nmix_s*);
# contradiction/oolong (JSONL on weka only) run in build_kv35_shards_beaker.sh.
#   ARMS="mix_s16M mix_s32M" sbatch -w mooney debug/flop_scaling/build_kv35_shards.sbatch
#SBATCH --job-name=fs35-kvdata
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/fs35_kvdata_%j.log
set -uo pipefail
ARMS="${ARMS:-mix_s16M mix_s32M mix_s64M mix_s160M mix_s320M}"
TASK="${TASK:-outlier}"          # outlier | nq (unified task name: outlier | retrieval)
ARMDIR="${ARMDIR:-/data/prasann/outlier_lengthmix/arms}"
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python; [ -x "$PY" ] || PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OUTROOT=/data/prasann/flop_scaling35/shards; mkdir -p $OUTROOT
S3=s3://ai2-llm/checkpoints/prasanns/flop_scaling35/shards
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore TMPDIR=/data/prasann/tmp
case "$TASK" in nq) CONV=retrieval;; *) CONV=$TASK;; esac
for A in $ARMS; do
  SRC=$ARMDIR/$A.jsonl; [ -s "$SRC" ] || { echo "!!! missing $SRC"; continue; }
  B=$(echo "$A" | grep -oE "s[0-9]+M"); OUT=$OUTROOT/${TASK}_${B}_mk
  [ -s $OUT/metadata.json ] && { echo "[skip] $OUT"; continue; }
  echo "--- $TASK $A -> $OUT $(date +%T) ---"; mkdir -p $OUT
  PYTHONPATH=$REPO/src $PY $REPO/src/scripts/data/convert_unified_to_document_landmark.py \
    --input-jsonl "$SRC" --task "$CONV" --out-dir "$OUT" --emit dense --marker-set qwen3_5 \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --seq-len 65536 --query-position after --cot-mode none \
    --chunk-by document --emit-gold-sidecar --num-proc 16 || { echo "!!! FAILED $A"; continue; }
  $PY -c "import json;m=json.load(open('$OUT/metadata.json'));print('   ', {k:m.get(k) for k in ('num_instances','task','marker_set','query_position','max_example_len')})"
  ls $OUT | grep -c gold_fingerprints
done
export AWS_PROFILE=S3
for d in $OUTROOT/${TASK}_*_mk/; do aws s3 sync "$d" "$S3/$(basename $d)" --exclude "*" --include "token_ids_part_*.npy" --include "labels_mask_*.npy" --include "metadata.json" --include "gold_fingerprints.json" --only-show-errors && echo "synced $(basename $d)"; done
echo "=== DONE $(date '+%F %T') ==="
