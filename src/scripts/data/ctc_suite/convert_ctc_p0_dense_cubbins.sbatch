#!/bin/bash
# CTC suite Stage-1 P0 dense document-chunked tokenization, Qwen3.5 marker set, run NODE-LOCAL on
# cubbins against the ready P0 train pools under /data/prasann/ctc_suite_data/ (hotpotqa + reorder
# generated directly on cubbins; contradiction/oolong/outlier/rerank/grouping scp'd in from mooney's
# single_task_ladders_20k / ctc_suite_data pools -- see records/ctc-suite-scaling-plan.md and
# src/scripts/data/ctc_suite/BUILD_MATRIX.md). nq is SKIPPED (regen in flight elsewhere).
#
# Wraps src/scripts/data/convert_unified_to_document_landmark.py (--emit dense --marker-set qwen3_5
# --tokenizer Qwen/Qwen3.5-0.8B-Base --seq-len 40960 --cot-mode none). Multi-rung pools (hotpotqa,
# reorder, grouping) pass all their per-rung files at once via --input-jsonl (nargs="+"); the
# already-combined 20k pools (contradiction/oolong/outlier/rerank) pass a single file.
#
# Output -> /data/prasann/ctc_suite/shards/<task>_train/.
#
#   sbatch --array=0-6 src/scripts/data/ctc_suite/convert_ctc_p0_dense_cubbins.sbatch
#   # single task (e.g. re-run one that failed):
#   sbatch --array=2 src/scripts/data/ctc_suite/convert_ctc_p0_dense_cubbins.sbatch
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctc-p0-conv
#SBATCH --output=/data/prasann/joblogs/ctc_p0_conv_%a_%A.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
DATA=/data/prasann/ctc_suite_data
OUT=/data/prasann/ctc_suite/shards
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
mkdir -p "$OUT" /data/prasann/joblogs

# index -> "name task chunk_by extra_args files..."
TASKS=(hotpotqa reorder contradiction oolong outlier rerank grouping)
NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"

case "$NAME" in
  hotpotqa)
    TASK=retrieval; CHUNK=document; EXTRA=""
    FILES=(
      "$DATA/hotpotqa/hotpotqa_train_k11_bridge_hn1_4000.jsonl"
      "$DATA/hotpotqa/hotpotqa_train_k24_bridge_hn2_4000.jsonl"
      "$DATA/hotpotqa/hotpotqa_train_k50_bridge_hn5_4000.jsonl"
      "$DATA/hotpotqa/hotpotqa_train_k100_bridge_hn10_4000.jsonl"
      "$DATA/hotpotqa/hotpotqa_train_k205_bridge_hn20_4000.jsonl"
    )
    ;;
  reorder)
    TASK=reorder; CHUNK=document; EXTRA=""
    FILES=(
      "$DATA/reorder/reorder_gutenberg100w_n12_train_4000.jsonl"
      "$DATA/reorder/reorder_gutenberg100w_n27_train_4000.jsonl"
      "$DATA/reorder/reorder_gutenberg100w_n57_train_4000.jsonl"
      "$DATA/reorder/reorder_gutenberg100w_n116_train_4000.jsonl"
      "$DATA/reorder/reorder_gutenberg100w_n234_train_4000.jsonl"
    )
    ;;
  contradiction)
    TASK=contradiction; CHUNK=document; EXTRA=""
    FILES=("$DATA/contradiction_pool/contradiction_train_pubmed_realistic_n50-950_k3.jsonl")
    ;;
  oolong)
    TASK=oolong; CHUNK=line; EXTRA="--item-regex ||"
    FILES=("$DATA/oolong_pool/oolong_ladder_train_combined.jsonl")
    ;;
  outlier)
    TASK=outlier; CHUNK=document; EXTRA=""
    FILES=("$DATA/outlier_pool/outlier_wiki100w_contin_n14-220_k3_20000.jsonl")
    ;;
  rerank)
    TASK=rerank; CHUNK=document; EXTRA=""
    FILES=("$DATA/rerank_pool/msmarco_trainhn_train_k20-315_20000.jsonl")
    ;;
  grouping)
    TASK=grouping; CHUNK=document; EXTRA=""
    FILES=(
      "$DATA/grouping_pool/openalex_grouping_n10_levels_train_4000.jsonl"
      "$DATA/grouping_pool/openalex_grouping_n21_levels_train_4000.jsonl"
      "$DATA/grouping_pool/openalex_grouping_n43_levels_train_4000.jsonl"
      "$DATA/grouping_pool/openalex_grouping_n88_levels_train_4000.jsonl"
      "$DATA/grouping_pool/openalex_grouping_n176_levels_train_4000.jsonl"
    )
    ;;
  *)
    echo "unknown task index $SLURM_ARRAY_TASK_ID -> $NAME"; exit 1 ;;
esac

echo "=== HOST=$(hostname) NAME=$NAME TASK=$TASK CHUNK=$CHUNK FILES=${#FILES[@]} START=$(date '+%F %T') ==="
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "  MISSING INPUT: $f"; exit 1; }
done

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

OUTDIR="$OUT/${NAME}_train"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

python "$CONV" --emit dense --task "$TASK" --chunk-by "$CHUNK" $EXTRA --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 40960 --shard-tokens 50000000 \
  --input-jsonl "${FILES[@]}" --out-dir "$OUTDIR"
rc=$?
echo "  convert rc=$rc $(date '+%T')"

echo "=== VERIFY box markers ($NAME) $(date '+%T') ==="
python3 - "$OUTDIR" <<'PY'
import sys, glob, json, numpy as np
d = sys.argv[1]
meta_path = d + "/metadata.json"
try:
    meta = json.load(open(meta_path))
    print("metadata:", {k: meta.get(k) for k in ("marker_set", "doc_start_id", "doc_end_id", "eos_token_id")})
except Exception as e:
    print("NO/BAD metadata.json:", e)
fs = sorted(glob.glob(d + "/token_ids_part_*.npy"))
if not fs:
    print("NO SHARDS"); sys.exit(1)
DS, DE, EOS = 248049, 248050, 248044
a = np.concatenate([np.fromfile(f, dtype=np.uint32) for f in fs])
ds = int((a == DS).sum()); de = int((a == DE).sum()); eos = int((a == EOS).sum())
ok = ds > 0 and ds == de
print(f"toks={len(a)} examples(eos)={eos} box_start={ds} box_end={de} OK={ok}")
print("ALL_OK" if ok else "FAILED")
PY
echo "=== DONE $NAME $(date '+%T') ==="
