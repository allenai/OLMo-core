#!/bin/bash
# Rebuild the CTC-suite oolong training shard WITHOUT the `--item-regex` leak, so oolong can finally
# get a like-for-like chunked-vs-dense row.
#
# THE BUG. Four converters passed a bare `--item-regex '||'`. As a regex that is an alternation of
# two empty branches, so it matches EVERY line. In `--chunk-by line` mode (oolong only) that wrapped
# the instruction / question / data-header lines as their own chunks and left the blank lines
# between them FREE, bridging chunks the document-chunked mask is supposed to isolate. oolong was
# the single anomaly in CHUNK_LEAK_AUDIT.md (~5 inter-chunk FREE tokens/example; all 25 other shards
# measured 0), and it degrades exactly one arm of one task -- which reads as a modeling result
# ("oolong chunked is just low") rather than a data bug.
#
# THE FIX IS ALREADY IN THE REPO -- `convert_ctc_p0_dense_cubbins.sbatch` now passes EXTRA="" and the
# converter hard-rejects any `--item-regex` matching the empty string
# (src/test/data/document_chunk_item_regex_test.py). Only the artifact on disk is stale: the live
# `/scratch/.../shards/oolong_train` was built 2026-07-19, before the fix.
#
# WHY NOT REUSE THE KNOWN-CLEAN SHARD. `horton:/data/prasann/xlong5_qafter/shards_chunked/oolong_train`
# is clean, but it belongs to the xlong5 2k-256k mix: `max_example_len` 249,226 and
# `query_position=after`. Training it needs a ~250k sequence (context parallelism), and its prompt
# layout does not match the CTC-suite oolong eval rungs. It is not a drop-in for this ladder.
#
# So: re-tokenize the SAME source pool the July shard came from, with the fixed invocation. No
# examples are regenerated. Settings mirror the old shard exactly (cot_mode=none -- note the
# converter's DEFAULT is 'plan'; chunk_by=line; seq_len 40960; query_position=both, the CTC-suite
# convention the eval rungs are rendered in) so the only difference is the removed bad flag.
#
#   sbatch debug/ctc_modelscale/rebuild_oolong_shard.sbatch
#SBATCH --job-name=oolong-rebuild
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/oolong_rebuild_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
SCAN=$REPO/debug/ctc_modelscale/scan_oolong_chunks.py
SRC=/data/prasann/ctc_suite_data/oolong_pool/oolong_ladder_train_combined.jsonl
OLD=/scratch/users/prasann/ctc_suite_staged/shards/oolong_train
OUTDIR=/data/prasann/ctc_suite/shards_final/oolong_train_fixed

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$SRC" ] || { echo "MISSING INPUT: $SRC"; exit 2; }

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "=== BEFORE: scan the stale 2026-07-19 shard (expect LEAK) ==="
python "$SCAN" --shard "$OLD" --max-examples 400

echo "=== REBUILD (no --item-regex; converter default is the escaped r'\\|\\|') ==="
echo "  input lines=$(wc -l < "$SRC")"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
python "$CONV" --emit dense --task oolong --chunk-by line --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --query-position both --seq-len 40960 --shard-tokens 60000000 \
  --input-jsonl "$SRC" --out-dir "$OUTDIR"
rc=$?
echo "  convert rc=$rc $(date '+%T')"
[ $rc -ne 0 ] && exit $rc

echo "=== AFTER: scan the rebuilt shard (expect CLEAN) ==="
python "$SCAN" --shard "$OUTDIR" --max-examples 400

echo "=== part/byte consistency (the qdmatch_nq truncation class) ==="
python - <<PY
import glob, json, os
d = "$OUTDIR"
tok = sorted(glob.glob(os.path.join(d, "token_ids_part_*.npy")))
lab = sorted(glob.glob(os.path.join(d, "labels_mask_part_*.npy")))
m = json.load(open(os.path.join(d, "metadata.json")))
tb = sum(os.path.getsize(p) for p in tok); lb = sum(os.path.getsize(p) for p in lab)
print(f"token parts={len(tok)} bytes={tb}")
print(f"label parts={len(lab)} bytes={lb}")
print(f"num_instances={m['num_instances']} max_example_len={m['max_example_len']} "
      f"item_regex={m.get('item_regex')!r} query_position={m.get('query_position')!r}")
ok = len(tok) == len(lab) and tb == m["num_tokens"] * 4 and lb == m["num_tokens"]
print("VERDICT:", "PASS" if ok else "FAIL -- truncated")
PY
echo "=== DONE $(date '+%F %T') ==="
