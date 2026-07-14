#!/bin/bash
# Detached end-to-end builder for the NQ length-ladder (8k/16k/32k/64k) ADDITIVE SFT data.
# Produces aligned per-rung jsonl, a combined train jsonl, and tokenizes the combined train
# into /scratch/users/prasann/cpt_data/nq_ladder64k (Qwen3, task=retrieval, eos 151643, cap 65536).
# Self-contained: regenerates the 64k rung via pyserini, aligns, subsamples, combines, tokenizes.
set -uo pipefail

CR=/scratch/users/prasann/corpus-reasoning
PY_GEN=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python   # has pyserini
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python        # has transformers/numpy
HEAD=/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/242e737e-b9ef-4a5f-9d27-880b1817b6c0/scratchpad/head_jsonl.py
MEASURE=/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/242e737e-b9ef-4a5f-9d27-880b1817b6c0/scratchpad/measure2.py
LADDER=$CR/data/ladder64k
RAW=$CR/data/ladder64k_raw
OUT=/scratch/users/prasann/cpt_data/nq_ladder64k
export PYTHONPATH=$CR
export TOKENIZERS_PARALLELISM=false
cd $CR
mkdir -p $LADDER

echo "===== build_nq_ladder64k START $(date) ====="

# ---------------------------------------------------------------------------
# 1. Generate the 64k rung (k420) fresh via pyserini BM25 (idempotent: skip if present)
# ---------------------------------------------------------------------------
# kill any stale generator from the interactive session
pkill -f "generate_nq_training_data.py" 2>/dev/null
sleep 3
rm -rf $RAW
echo "----- generating 64k rung (k420, 250 train / 130 eval) $(date) -----"
$PY_GEN scripts/data/generate_nq_training_data.py \
    --num-train 250 --num-eval 130 --num-docs 420 --num-hard-negatives 419 \
    --output-dir $RAW
echo "generation exit=$? $(date)"
ls -la $RAW

RAW_TRAIN=$(ls $RAW/nq_train_k420_hn419_*.jsonl 2>/dev/null | head -1)
RAW_EVAL=$(ls $RAW/nq_validation_k420_hn419_*.jsonl 2>/dev/null | head -1)
echo "raw train=$RAW_TRAIN  raw eval=$RAW_EVAL"

# ---------------------------------------------------------------------------
# 2. (align SKIPPED) generate_nq_training_data.py now sources gold/hard/random all from
#    wikipedia-dpr-100w, so align_hn_doc_lengths.py is a no-op (see its docstring). Use raw
#    directly; keep the *_aligned filenames so downstream steps are unchanged.
# ---------------------------------------------------------------------------
echo "----- (align skipped: unified wikipedia-dpr-100w sourcing) copying raw 64k rung $(date) -----"
cp "$RAW_TRAIN" $LADDER/nq_train_k420_aligned.jsonl
cp "$RAW_EVAL"  $LADDER/nq_validation_k420_aligned.jsonl

# ---------------------------------------------------------------------------
# 3. Subsample rung files (8k/16k/32k already produced interactively; recreate to be safe)
# ---------------------------------------------------------------------------
echo "----- subsampling rungs $(date) -----"
$PY $HEAD $LADDER/nq_train_k50_aligned.jsonl   $LADDER/rung_8k_train_k50.jsonl    400
$PY $HEAD $LADDER/nq_train_k100_aligned.jsonl  $LADDER/rung_16k_train_k100.jsonl  300
$PY $HEAD $LADDER/nq_train_k200_aligned.jsonl  $LADDER/rung_32k_train_k200.jsonl  200
$PY $HEAD $LADDER/nq_train_k420_aligned.jsonl  $LADDER/rung_64k_train_k420.jsonl  250
$PY $HEAD $LADDER/nq_validation_k50_aligned.jsonl   $LADDER/rung_8k_eval_k50.jsonl    100
$PY $HEAD $LADDER/nq_validation_k100_aligned.jsonl  $LADDER/rung_16k_eval_k100.jsonl  100
$PY $HEAD $LADDER/nq_validation_k200_aligned.jsonl  $LADDER/rung_32k_eval_k200.jsonl  100
$PY $HEAD $LADDER/nq_validation_k420_aligned.jsonl  $LADDER/rung_64k_eval_k420.jsonl  100

# ---------------------------------------------------------------------------
# 4. Build the COMBINED train jsonl (additive: existing 2500 k20 + new rungs)
# ---------------------------------------------------------------------------
echo "----- building combined train jsonl $(date) -----"
COMBINED=$LADDER/combined_train.jsonl
cat $CR/data/nq_train_k20_hn19_2500_aligned.jsonl \
    $LADDER/rung_8k_train_k50.jsonl \
    $LADDER/rung_16k_train_k100.jsonl \
    $LADDER/rung_32k_train_k200.jsonl \
    $LADDER/rung_64k_train_k420.jsonl > $COMBINED
echo "combined train lines: $(wc -l < $COMBINED)"

# ---------------------------------------------------------------------------
# 5. VERIFY token-length distribution per rung (Qwen3) + existing baseline
# ---------------------------------------------------------------------------
echo "----- per-rung token-length distribution (aligned, Qwen3, incl EOS) $(date) -----"
$PY $MEASURE \
    "$CR/data/nq_train_k20_hn19_2500_aligned.jsonl:aligned" \
    "$LADDER/rung_8k_train_k50.jsonl:aligned" \
    "$LADDER/rung_16k_train_k100.jsonl:aligned" \
    "$LADDER/rung_32k_train_k200.jsonl:aligned" \
    "$LADDER/rung_64k_train_k420.jsonl:aligned" \
    "$LADDER/rung_8k_eval_k50.jsonl:aligned" \
    "$LADDER/rung_16k_eval_k100.jsonl:aligned" \
    "$LADDER/rung_32k_eval_k200.jsonl:aligned" \
    "$LADDER/rung_64k_eval_k420.jsonl:aligned"

# ---------------------------------------------------------------------------
# 6. Tokenize the COMBINED train into the new olmo-core SFT dir (cap 65536)
# ---------------------------------------------------------------------------
echo "----- tokenizing combined train -> $OUT $(date) -----"
$PY scripts/data/convert_nq_to_sft.py \
    --input $COMBINED \
    --out-dir $OUT \
    --max-seq-len 65536

echo "----- final metadata -----"
cat $OUT/metadata.json
echo
echo "===== build_nq_ladder64k DONE $(date) ====="
