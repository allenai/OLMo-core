#!/bin/bash
# Held-out contradiction eval for the gold-grad arms, run DIRECTLY on the current session's GPUs.
#
# Gold-grad only changes the training BACKWARD pass -- the saved checkpoints are ordinary
# document-chunked models -- so this is the existing docchunk-contra eval, unmodified and unaware of
# gold. Every arm is scored identically, which is what makes the comparison mean anything.
#
# eval_size defaults to 488 = the WHOLE contradiction eval file. Do not lower it: a sub-500 eval
# inflates noise into apparent findings (binomial SE at f1~0.7 is +/-0.021 at 488 but +/-0.046 at 100).
#
#   bash src/scripts/train/sft/eval_q06b_goldgrad_local.sh n100
#   RUNS="q06b-goldgrad-fa-n20-full ..." bash src/scripts/train/sft/eval_q06b_goldgrad_local.sh n20
set -uo pipefail

RUNG="${1:-n100}"
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
CR_SRC=/scratch/users/prasann/corpus-reasoning
TAG="${TAG:-famark}"
CKPT_ROOT="${CKPT_ROOT:-/data/prasann/olmo_ckpts}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
EVAL_SIZE="${EVAL_SIZE:-488}"
# The gold answer is ~27 tokens ('<think></think>[[a, b], [c, d], [e, f]]<|im_end|>'). The eval decodes
# one example at a time (no batching), so max_new_tokens is a direct multiplier on wall-clock for any
# example that fails to emit EOS. 256 was ~9x the real answer length; 64 is still >2x headroom.
MAXNEW="${MAXNEW:-64}"
LOGDIR=/data/prasann/goldgrad_local
mkdir -p "$LOGDIR"

# VARIANT must match how the checkpoint was TRAINED:
#   dense -> DocumentChunkedAttention (n20 rung)
#   full  -> plain causal, NO docchunk module (n100 rung, trained with --plain-attention). It builds the
#            SAME box-marker prefill as dense -- the markers are just tokens that full attention ignores
#            -- so the prompt format still matches training exactly.
case "$RUNG" in
  n20)  NDOCS=20;  MAXLEN=4096; VARIANT="${VARIANT:-dense}" ;;
  n100) NDOCS=100; MAXLEN=8192; VARIANT="${VARIANT:-full}"  ;;
  *) echo "usage: $0 {n20|n100}"; exit 2 ;;
esac
CONTRA_FILE="contradiction_eval_pubmed_both_n${NDOCS}_k3.jsonl"
RUNS="${RUNS:-q06b-goldgrad-${TAG}-${RUNG}-full q06b-goldgrad-${TAG}-${RUNG}-gpr2 q06b-goldgrad-${TAG}-${RUNG}-rand2 q06b-goldgrad-${TAG}-${RUNG}-gsub1_15}"

CR=/data/prasann/goldgrad/cr_eval
mkdir -p "$CR/data" "$CR/outputs/eval_results"
rsync -a --exclude '__pycache__' "$CR_SRC/scripts" "$CR/" 2>/dev/null
cp -u "$CR_SRC/data/$CONTRA_FILE" "$CR/data/" 2>/dev/null
[ -f "$CR/data/$CONTRA_FILE" ] || { echo "FATAL: eval set $CONTRA_FILE missing"; exit 3; }

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$CR"
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$CR"

echo "=== HOST=$(hostname) RUNG=$RUNG eval_size=$EVAL_SIZE EVAL=$CONTRA_FILE START=$(date '+%F %T') ==="
for RUN in $RUNS; do
  MODEL_PATH="$CKPT_ROOT/$RUN"
  if [ ! -d "$MODEL_PATH/model_and_optim" ]; then echo "=== SKIP $RUN (no checkpoint) ==="; continue; fi
  OUT="outputs/eval_results/goldgrad-${RUN}_native.json"
  echo "=== [$RUN] eval start $(date '+%F %T') ==="
  torchrun --nproc_per_node="$NGPU" --master_port=$((29900 + RANDOM % 90)) \
    scripts/eval/eval_lc_native_docchunk_contra.py \
    --variant "$VARIANT" --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer Qwen/Qwen3-0.6B --contra-data "data/$CONTRA_FILE" \
    --cot-mode none --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --max-test-samples "$EVAL_SIZE" 2>&1 | tee "$LOGDIR/eval-${RUN}.log" | grep -E "f1=|Error|Traceback"
  echo "=== [$RUN] rc=$? $(date '+%F %T') ==="
done
echo "=== ALL EVAL DONE $(date '+%F %T') ==="
