#!/bin/bash
# Reconvert reorder (task 24) ONLY, after the document-wrapping fix (2026-07-19): the reorder
# prompt renderer collapses each passage's internal "\n\n" to "\n"
# (corpus_reasoning/lib/data_format.py _format_documents, task=="reorder"), but
# olmo_core/data/document_chunk_landmark.py::_wrap_documents did a verbatim substring search
# against the RAW (un-collapsed) document text, so every passage with an internal paragraph break
# failed to match and stayed FREE (unisolated). Fixed by mirroring the renderer's normalization in
# _wrap_documents via a new task-keyed _RENDERER_TEXT_NORMALIZERS table (segment_prompt_to_chunks
# now threads `task` through to _wrap_documents). See BUILD_MATRIX.md Stage-2 build log / FIX 1.
#
# Same flags as the original array job (convert_ctc_p0_dense_cubbins.sbatch, index 1 = reorder);
# this is a standalone single-task rerun so it doesn't require re-running the other 6 tasks.
#
#   sbatch src/scripts/data/ctc_suite/reconvert_reorder_cubbins.sbatch
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctc-reorder-reconv
#SBATCH --output=/data/prasann/joblogs/ctc_reorder_reconv_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
DATA=/data/prasann/ctc_suite_data
OUT=/data/prasann/ctc_suite/shards
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
mkdir -p "$OUT" /data/prasann/joblogs

TASK=reorder; CHUNK=document
FILES=(
  "$DATA/reorder/reorder_gutenberg100w_n12_train_4000.jsonl"
  "$DATA/reorder/reorder_gutenberg100w_n27_train_4000.jsonl"
  "$DATA/reorder/reorder_gutenberg100w_n57_train_4000.jsonl"
  "$DATA/reorder/reorder_gutenberg100w_n116_train_4000.jsonl"
  "$DATA/reorder/reorder_gutenberg100w_n234_train_4000.jsonl"
)

echo "=== HOST=$(hostname) TASK=$TASK CHUNK=$CHUNK FILES=${#FILES[@]} START=$(date '+%F %T') ==="
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "  MISSING INPUT: $f"; exit 1; }
done

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

OUTDIR="$OUT/reorder_train"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

python "$CONV" --emit dense --task "$TASK" --chunk-by "$CHUNK" --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 40960 --shard-tokens 50000000 \
  --input-jsonl "${FILES[@]}" --out-dir "$OUTDIR"
rc=$?
echo "  convert rc=$rc $(date '+%T')"

echo "=== VERIFY box markers + docs/ex coverage (reorder) $(date '+%T') ==="
python3 - "$OUTDIR" "${FILES[@]}" <<'PY'
import sys, glob, json, numpy as np
d = sys.argv[1]
in_files = sys.argv[2:]
meta_path = d + "/metadata.json"
try:
    meta = json.load(open(meta_path))
    print("metadata:", {k: meta.get(k) for k in ("marker_set", "doc_start_id", "doc_end_id", "eos_token_id", "num_instances", "num_dropped")})
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

expected_docs = 0
for p in in_files:
    with open(p) as f:
        for line in f:
            if line.strip():
                expected_docs += len(json.loads(line)["documents"])
avg_docs_per_ex = ds / eos if eos else float("nan")
print(f"expected total docs (raw input) = {expected_docs}; box_start count = {ds} "
      f"coverage = {ds/expected_docs:.1%}" if expected_docs else "n/a")
print(f"avg docs/ex wrapped (box_start / #examples) = {avg_docs_per_ex:.1f}")
assert ds / expected_docs >= 0.90, (
    f"REGRESSION: only {ds/expected_docs:.1%} of reorder documents were wrapped "
    "(expected >=90%) -- the verbatim-match fix may have broken again."
)
print("ALL_OK" if ok else "FAILED")
PY
echo "=== DONE reorder $(date '+%T') ==="
