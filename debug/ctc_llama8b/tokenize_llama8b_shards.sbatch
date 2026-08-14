#!/bin/bash
# Build the Llama-tokenized contradiction + qdmatch_fiqa TRAIN shards for the Llama-3.1-8B family
# control, at settings matched to the Qwen3.5-4B suite shards so the cross-family numbers are
# comparable.
#
#   sbatch debug/ctc_llama8b/tokenize_llama8b_shards.sbatch
#
# ── WHY NOT REUSE THE EXISTING LLAMA CONTRADICTION SHARD ──────────────────────────────────────
# /scratch/users/prasann/ctc_llama/shards/contradiction20k_train_llama exists, but it was built at
# --seq-len 18432, and the converter DROPS over-length examples: 8,735 instances survived out of a
# ~20k pool, against 19,366 / max_example_len 40,957 for the Qwen shard. That is a different (and
# much easier) training set on a shorter ladder, which is exactly why the earlier Llama runs
# stopped at 16k. Comparing an 8.7k-instance Llama run against a 19.4k-instance Qwen run and
# calling the difference "family" would be wrong. Rebuilt here at 40960.
#
# ── qdmatch_fiqa: PIN THE q61 LADDER, DO NOT ls -t ────────────────────────────────────────────
# The generator emitted TWO query-count ladders into the same directory (q5/13/29/61/124 and
# q9/21/42/84/168). The Qwen build selected per-rung with `ls -t | head -1`, i.e. by mtime, which
# happened to land on the q61 family -- and that is the family the shipped eval is scored on
# ("scored on the q61 rung family, matching the trained shard"). Reproducing a *mtime-dependent*
# selection on a different day is not reproducible, so the q61 files are named explicitly here.
# If these two ever disagree, the shard and the eval ladder are measuring different tasks.
#
# ── TOKENIZER ─────────────────────────────────────────────────────────────────────────────────
# Uses the Llama-3.2-3B marker tokenizer. Llama 3.1 and 3.2 share a tokenizer (128256 vocab, same
# reserved slots), and the marker ids the shard is built against come from RESERVED_IDS['llama'],
# which is a fixed id set -- so this is tokenizer-compatible with the 8B base. VERIFY_STEP below
# re-checks that against the 8B's own marker tokenizer once the Beaker convert job has produced it
# (debug/ctc_llama8b/convert_base_gantry.sh); a mismatch means these shards must be rebuilt.
#
#SBATCH --job-name=llama8b-shards
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=normal
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/llama8b_shards_%j.log
#
# NO --gres: tokenization is pure CPU. Asking for a GPU would queue this behind the training
# fan-out for no reason (and, at preemptive_high, would count against the 8-GPU per-user cap).
# qos=normal for the same reason -- nothing here is worth preempting anyone over.
#
# ⚠ --output MUST be under /data. A slurm --output on /accounts or /scratch deadlocks the job in
# nfs_wait_bit_killable before it prints anything (see local_cluster.md).
# ⚠ --nodelist=cubbins is load-bearing: BOTH raw pools live on cubbins' node-local /data. Reading
# them over /net from another node is the ~5 MB/s NFS path on a 1.6 GB file.

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann
PY="$ENV/bin/python"

TOK=/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok
SEQ=40960
STAGE=/data/prasann/ctc_llama8b/shards
mkdir -p "$STAGE" /data/prasann/joblogs
echo "=== HOST=$(hostname) START=$(date '+%F %T') seq_len=$SEQ tok=$TOK ==="
[ -d "$TOK" ] || { echo "FATAL: marker tokenizer $TOK missing"; exit 2; }
cd "$REPO"

# ---- 1/2 contradiction ------------------------------------------------------------------------
IN_CONTRA=/data/prasann/ctc_suite_data/contradiction_pool/contradiction_train_pubmed_realistic_n50-950_k3.jsonl
OUT_CONTRA="$STAGE/contradiction_train_llama8b"
[ -f "$IN_CONTRA" ] || { echo "FATAL: contradiction pool $IN_CONTRA missing on $(hostname)"; exit 3; }
echo "--- 1/2 contradiction -> $OUT_CONTRA"
rm -rf "$OUT_CONTRA"; mkdir -p "$OUT_CONTRA"
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none \
  --marker-set llama --tokenizer "$TOK" \
  --seq-len "$SEQ" --limit 0 --shard-tokens 60000000 \
  --input-jsonl "$IN_CONTRA" --out-dir "$OUT_CONTRA"
echo "contradiction rc=$?"; cat "$OUT_CONTRA/metadata.json"; echo

# ---- 2/2 qdmatch_fiqa -------------------------------------------------------------------------
# q61 ladder, named explicitly (see the header note on ls -t).
SRC=/data/prasann/qdmatch_fiqa
OUT_QD="$STAGE/qdmatch_fiqa_train_llama8b"
INPUTS=(
  "$SRC/qdmatch_train_fiqa_r2048_q5_n5_k3_separate.jsonl"
  "$SRC/qdmatch_train_fiqa_r4096_q13_n13_k3_separate.jsonl"
  "$SRC/qdmatch_train_fiqa_r8192_q29_n29_k3_separate.jsonl"
  "$SRC/qdmatch_train_fiqa_r16384_q61_n61_k3_separate.jsonl"
  "$SRC/qdmatch_train_fiqa_r32768_q124_n124_k3_separate.jsonl"
)
for f in "${INPUTS[@]}"; do
  [ -f "$f" ] || { echo "FATAL: missing q61-ladder input $f"; exit 4; }
  echo "  input: $(basename "$f")"
done
echo "--- 2/2 qdmatch_fiqa -> $OUT_QD"
rm -rf "$OUT_QD"; mkdir -p "$OUT_QD"
# --query-position both MIRRORS THE QWEN SHARD. The project default for new runs is `after`, but
# the qwen3_5 qdmatch_fiqa shard this is being compared against was built with `both`, and the
# eval renders to match the shard. Changing it here would make the Llama arm non-comparable in a
# way that looks like a family effect.
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task qdmatch --chunk-by document --cot-mode none \
  --marker-set llama --tokenizer "$TOK" \
  --query-position both --seq-len "$SEQ" --shard-tokens 60000000 \
  --input-jsonl "${INPUTS[@]}" --out-dir "$OUT_QD"
echo "qdmatch_fiqa rc=$?"; cat "$OUT_QD/metadata.json"; echo

# ---- comparability report ---------------------------------------------------------------------
# Print the Llama instance counts next to the Qwen ones. A large shortfall means the converter is
# still dropping the long tail and the ladder is not matched, which is the whole reason for this
# rebuild -- better to see it here than to infer it from a training curve.
echo "=== COMPARABILITY (llama vs the qwen3_5 suite shards) ==="
$PY - <<'PY'
import json, os
want = {"contradiction_train_llama8b": ("contradiction", 19366, 40957),
        "qdmatch_fiqa_train_llama8b": ("qdmatch_fiqa", 19997, 40399)}
for d, (name, qn, qmax) in want.items():
    p = f"/data/prasann/ctc_llama8b/shards/{d}/metadata.json"
    if not os.path.exists(p):
        print(f"  {name:14} MISSING"); continue
    m = json.load(open(p))
    n = m.get("num_instances") or m.get("instances")
    mx = m.get("max_example_len")
    frac = (n / qn) if (n and qn) else 0
    flag = "" if frac >= 0.9 else f"   <<< only {frac:.0%} of the qwen shard -- LONG TAIL STILL DROPPED"
    print(f"  {name:14} llama n={n} max_len={mx}   qwen n={qn} max_len={qmax}{flag}")
PY

echo "=== staging to /scratch for the S3 relay ==="
DST=/scratch/users/prasann/ctc_llama8b/shards
mkdir -p "$DST"
cp -r "$STAGE"/contradiction_train_llama8b "$STAGE"/qdmatch_fiqa_train_llama8b "$DST"/ 2>/dev/null
echo "staged -> $DST"
echo "=== DONE $(date '+%F %T') ==="
