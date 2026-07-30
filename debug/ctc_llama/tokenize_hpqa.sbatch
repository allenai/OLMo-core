#!/bin/bash
# Tokenize the hotpotqa RETRIEVAL pool into Llama document-chunked dense SFT shards.
# hotpotqa routes through --task retrieval (metric gold_id_f1); this is the O(N) half of the CTC
# hypothesis, the counterpart to the O(N*M) qdmatch row.
#
# The BUILD_MATRIX-§2 pool (n in 11/24/50/100/205, bridge, hard-negs n/10, 4000 each) already has
# its BM25 hard negatives mined, from the original Qwen build. Nothing about hotpotqa needs
# regeneration -- only re-TOKENIZATION with the Llama tokenizer + marker set. That also keeps
# pyserini's JVM out of the loop entirely; it is documented to hang on jsteinhardt compute nodes.
#
# Subsample 2000 per n with seed 42 -> 10k, discrete-uniform over n, exactly as the recipe says.
#
# NOTE: no `#SBATCH --gres` line at all. Requesting `gpu:0` explicitly makes slurm report
# ReqNodeNotAvail even on a node that is MIXED with hundreds of free CPUs; omitting it schedules
# instantly.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=llama-tok-hpqa
#SBATCH --output=/scratch/users/prasann/ctc_llama_logs/tok_hpqa_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
WORK=/data/prasann/ctc_llama/hpqa_subsample
OUTDIR=/data/prasann/ctc_llama/shards/hotpotqa_train_llama
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann
TOK=/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok
SEQ_LEN=${SEQ_LEN:-28672}
mkdir -p "$LOGDIR" "$WORK"
echo "=== HOST=$(hostname) START=$(date '+%F %T') SEQ_LEN=$SEQ_LEN ==="
# Prefer the ALREADY-MATERIALISED 10k subsample on shared storage. A cross-family comparison on
# different training data is not a comparison, and "same seed" is NOT sufficient to guarantee the
# same draw: the sibling used a single Random(42) with rng.sample() across the 5 bucket files,
# while a fresh Random(42) per bucket (or shuffle-then-head) selects a DIFFERENT subset from the
# identical pool at the identical seed. Consuming the sibling's emitted file removes the question
# entirely -- both families read byte-identical examples. subsample_hpqa_pool.py remains only as
# the fallback for regenerating the ladder from scratch if that file is ever lost.
SHARED10K=${SHARED10K:-/scratch/users/prasann/ctc_suite_staged/hpqa_tmp/hotpotqa_10k.jsonl}
rm -f "$WORK"/*.jsonl
if [ -s "$SHARED10K" ]; then
  echo "SOURCE: shared 10k subsample $SHARED10K ($(wc -l < "$SHARED10K") lines, md5 $(md5sum "$SHARED10K" | cut -d' ' -f1))"
  ln -sf "$SHARED10K" "$WORK/hotpotqa_10k.jsonl"
else
  echo "SOURCE: FALLBACK -- regenerating the subsample from the pool ($SHARED10K missing)"
  python "$REPO/debug/ctc_llama/subsample_hpqa_pool.py" \
    --pool /scratch/users/prasann/ctc_suite_staged/hpqa_pool \
    --out "$WORK" --per-bucket 2000 --seed 42 || exit 2
fi
ls -la "$WORK"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
cd "$REPO"
python src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task retrieval --chunk-by document --cot-mode none \
  --marker-set llama --tokenizer "$TOK" \
  --seq-len "$SEQ_LEN" --num-proc 8 --shard-tokens 60000000 \
  --input-jsonl "$WORK"/*.jsonl --out-dir "$OUTDIR"
rc=$?
echo "convert rc=$rc"
[ $rc -eq 0 ] || exit $rc
cat "$OUTDIR/metadata.json"
cp -f "$OUTDIR/metadata.json" "$LOGDIR/shard_meta_hotpotqa.json"
DST=/scratch/users/prasann/ctc_llama/shards/hotpotqa_train_llama
mkdir -p "$DST"; cp -f "$OUTDIR"/* "$DST"/ && echo "staged -> $DST"
touch "$LOGDIR/HPQA_SHARD_READY"
echo "=== DONE $(date '+%F %T') ==="
