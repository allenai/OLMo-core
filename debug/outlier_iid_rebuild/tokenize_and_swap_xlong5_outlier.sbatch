#!/bin/bash
# Tokenize the scale-K outlier pool and SWAP it into the xlong5 qafter root.
#
# Replaces shards_full/outlier_train in
#   /weka/.../prasanns/xlong5_2k256k_qwen35_qafter/
# Settings copied verbatim from that shard's own metadata.json so the replacement is drop-in:
#   emit=dense, cot_mode=none, wrap_docs=false, doc_markers=false, marker_set=qwen3_5,
#   tokenizer=Qwen3.5-4B-Base, eos=248044, query_position=AFTER, seq-len 262144
#
# THE ORIGINAL IS PRESERVED, NOT DELETED. It is moved to outlier_train_prescalek_YYYYMMDD/ on weka
# before the new data lands, so the swap is reversible and anyone mid-run can point back at it.
# (outlier is query-position agnostic per that root's README -- its qafter and both arms are the
# same data -- so this pool is valid for the `both` root too.)
#
# WHAT CHANGED: K now scales with n (K ~ n/9.5) instead of being pinned, and every majority topic
# exceeds the 3-doc outlier by >=2. Verified per band: K = 3/5/10/19/37/73/152.5 across
# n = 14-28 ... 908-1816, docs/topic 9.0-9.8, min_gap 2 everywhere.
#SBATCH --job-name=outlier-x5-tok-swap
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_x5_tok_swap_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
export AWS_PROFILE=S3
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

REPO=/data/prasann/repo/OLMo-core
# NODE-LOCAL interpreter. The same env also lives on NFS at /scratch/users/prasann/conda/envs/,
# and pointing at that copy cost ~18 min per attempt: the process parks in D state
# (rpc_wait_bit_killable, ~1% CPU) paging site-packages over the ~5 MB/s link before it executes a
# single line. Twice that read as a hung job. Node-local is the only correct choice here.
# (The -eval env is not an option either way -- it lacks cached_path, which olmo_core imports.)
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
SRC=/data/prasann/outlier_xlong5_train/outlier_scalek_2k-256k_k3_20000.jsonl
OUT=/data/prasann/outlier_xlong5_train/tokenized_qafter
TOK=/data/prasann/hf_models/Qwen3.5-4B-Base

echo "=== [outlier-x5-tok-swap] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$SRC" || { echo "!!! pool missing"; exit 1; }
[ -d "$TOK" ] || TOK=Qwen/Qwen3.5-4B-Base
echo "tokenizer: $TOK"

# Wipe the output dir, don't just mkdir into it. The converter writes token_ids_part_NNNNNN.npy and
# overwrites index-by-index, so a rerun that emits FEWER shards than its predecessor leaves the old
# high-numbered parts behind -- and every consumer globs part_*.npy, so those orphans get read as
# training data. A previous attempt here wrote 18 parts; the corrected --shard-tokens 1e8 writes ~4.
rm -rf "$OUT"
mkdir -p "$OUT"

# PREFLIGHT. Both previous failures surfaced ~18 minutes in -- one a missing module, one pure NFS
# import latency -- so the run looked alive the whole time it was doomed. Import the exact modules
# the converter needs first and time it: a node-local env clears this in seconds, and an NFS one
# announces itself immediately instead of after a third of an hour.
echo "--- preflight: interpreter + imports ---"
echo "python: $PY"
case "$PY" in /data/*) ;; *) echo "!!! interpreter is NOT node-local -- refusing (see header)"; exit 1;; esac
t0=$(date +%s)
PYTHONPATH="$REPO/src" $PY -c "
import sys, time
t=time.time()
import cached_path, transformers, datasets
from olmo_core.data.document_chunk_landmark import *
print(f'  imports OK in {time.time()-t:.1f}s  (py {sys.version.split()[0]})', flush=True)
" || { echo "!!! preflight FAILED -- aborting before the expensive work"; exit 1; }
echo "--- preflight done in $(( $(date +%s) - t0 ))s ---"

# Flags are NOT hand-picked -- they are the shipped shard's own build line from
# debug/xlong5_qafter/tokenize_all.sbatch (VARIANT=full, spec "outlier:outlier:document").
# The first attempt omitted three of them and the converter's defaults silently disagreed with the
# shard being replaced: chunk_by line-vs-document, doc markers ON in a `shards_full` (no-marker)
# root, and seq-len 262144 vs 250000 (which changes WHICH examples get dropped). A dense arm
# carrying document markers is exactly the kind of difference that reads later as a modeling result.
PYTHONPATH="$REPO/src" $PY "$REPO/src/scripts/data/convert_unified_to_document_landmark.py" \
  --input-jsonl "$SRC" \
  --task outlier \
  --out-dir "$OUT" \
  --emit dense \
  --chunk-by document \
  --no-doc-markers \
  --seq-len 250000 \
  --shard-tokens 100000000 \
  --query-position after \
  --cot-mode none \
  --tokenizer "$TOK" \
  --marker-set qwen3_5 \
  --num-proc 32 || { echo "!!! tokenization FAILED"; exit 1; }

echo ""
echo "=== GATE: new shard metadata must match the shard it replaces ==="
# Fetch the real metadata.json off S3 and diff it field by field. Reading two JSONs by eye is how
# the chunk_by / doc_markers / seq_len mismatch survived the first attempt -- this makes a drop-in
# replacement a checked property, and a mismatch stops the job before anything is uploaded.
aws s3 cp s3://ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35_qafter/shards_full/outlier_train/metadata.json \
  "$TMPDIR/orig_metadata.json" --only-show-errors || { echo "!!! could not fetch original metadata"; exit 1; }

$PY - "$TMPDIR/orig_metadata.json" "$OUT/metadata.json" <<'PYEOF' || { echo "!!! GATE FAILED -- not uploading"; exit 1; }
import json, sys
orig, new = (json.load(open(p)) for p in sys.argv[1:3])
# Structural fields: a drop-in replacement MUST agree on every one of these.
MUST = ["task","emit","cot_mode","query_position","chunk_by","wrap_docs","doc_markers",
        "use_titles","free_pad_repeat","summary_every_k","marker_set","eos_token_id",
        "doc_start_id","doc_end_id","landmark_token_id","pad_token_id","dtype","mask_dtype"]
# Content fields: these SHOULD differ -- it is different data. Printed, never enforced.
INFO = ["num_instances","num_dropped","num_tokens","num_loss_tokens","max_example_len","min_example_len"]
bad = [(k, orig.get(k), new.get(k)) for k in MUST if orig.get(k) != new.get(k)]
print(f"{'field':<22} {'original':>14} {'new':>14}")
for k in MUST:
    flag = "  <-- MISMATCH" if orig.get(k) != new.get(k) else ""
    print(f"{k:<22} {str(orig.get(k)):>14} {str(new.get(k)):>14}{flag}")
print("--- content (expected to differ) ---")
for k in INFO:
    print(f"{k:<22} {str(orig.get(k)):>14} {str(new.get(k)):>14}")
if bad:
    print(f"\n{len(bad)} structural mismatch(es); this is NOT a drop-in replacement.")
    sys.exit(1)
print("\nAll structural fields match -- safe to swap.")
PYEOF

echo ""
echo "=== upload replacement to S3 ==="
# --delete is load-bearing, not tidiness: a plain sync leaves orphaned high-numbered parts from any
# previous upload (an 18-part run once landed here), and consumers glob part_*.npy.
aws s3 sync "$OUT" s3://ai2-llm/checkpoints/prasanns/_transfer/outlier_train_scalek_qafter \
  --delete --only-show-errors || { echo "!!! s3 upload FAILED"; exit 1; }
aws s3 ls s3://ai2-llm/checkpoints/prasanns/_transfer/outlier_train_scalek_qafter/
echo "=== [outlier-x5-tok-swap] DONE $(date '+%F %T') ==="
