#!/bin/bash
# ON-BEAKER runner for the PREFILL-WIDE top-k landmark sweep. Trimmed copy of
# src/scripts/train/memexpress/singletask_ladder/run_beaker_multirung_eval.sh: same weka wiring
# (eval code in-repo, ladder data from the v2 CLEAN bundle, checkpoint auto-globbed), but it calls
# debug/prefill_topk/eval_lc_native_prefill_topk.py once per prefill-top-k config instead of the
# production eval once.
#
# The production landmark eval applies hard top-k block retrieval only at DECODE; the prefill still
# soft-gates every prompt token over all past blocks. Each config below re-runs the SAME ladder with
# top-k additionally applied to every prefill query, so the deltas are the cost of being honestly
# sparse.
#
# Env in (set by launch_beaker_prefill_topk_eval.py):
#   RUN        run name under $WEKA_LLM/checkpoints/prasanns/<RUN>   (checkpoint source + label)
#   WEKA_LLM   weka ai2-llm root
#   TASK       ladder task key (default contradiction)
#   RUNGS      comma rung list (default 2k,8k,16k,32k)
#   CONFIGS    ';'-separated "<tag>|<extra eval flags>" list; default = the sweep below
#   STEP/CKPT  optional step pin / absolute step dir
#   EVAL_OUT_DIR, MAX_TEST, MAX_LENGTH, NGPU, TOKENIZER, PROMPT_FORMAT
set -uo pipefail
RUN="${RUN:?set RUN=<run name>}"
WEKA_LLM="${WEKA_LLM:?set WEKA_LLM=<weka ai2-llm root>}"
TASK="${TASK:-contradiction}"
RUNGS="${RUNGS:-2k,8k,16k,32k}"
STEP="${STEP:-}"
MAX_TEST="${MAX_TEST:-600}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
NGPU="${NGPU:-2}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"
# Landmark/compressive generation is bs=1 (blocks are tied to absolute position).
BATCH_SIZE=1

PRASANNS="$WEKA_LLM/checkpoints/prasanns"
BUNDLE="${BUNDLE:-$PRASANNS/_eval_bundle}"
EVAL500="${EVAL500:-$PRASANNS/_eval_bundle_eval500_v2_clean}"
RUN_DIR="$PRASANNS/$RUN"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$RUN_DIR/eval_prefill_topk}"
REPO="${REPO:-$PWD}"

export PYTHONPATH="$REPO/src/scripts:$REPO/src:${PYTHONPATH:-}"
export EVAL500_ROOT="$EVAL500"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$EVAL_OUT_DIR"

echo "=== BEAKER prefill-topk eval | host=$(hostname) RUN=$RUN TASK=$TASK RUNGS=$RUNGS NGPU=$NGPU START=$(date -u '+%F %T')Z ==="
echo "    BUNDLE=$BUNDLE"
echo "    EVAL500=$EVAL500"
echo "    OUT=$EVAL_OUT_DIR"
nvidia-smi -L 2>/dev/null | head -8 || true

# ---- resolve the checkpoint step dir (CKPT override > STEP pin > latest complete step) ----
if [ -n "${CKPT:-}" ]; then
  :
elif [ -n "$STEP" ]; then
  CKPT="$RUN_DIR/$STEP"
else
  CKPT=""
  for d in $(ls -d "$RUN_DIR"/step*/ 2>/dev/null | sed 's#/$##' | sort -V); do
    [ -f "$d/model_and_optim/.metadata" ] && CKPT="$d"
  done
fi
CKPT="${CKPT%/}"
if [ -z "$CKPT" ] || [ ! -f "$CKPT/model_and_optim/.metadata" ]; then
  echo "ERROR: no complete step dir under $RUN_DIR (CKPT='$CKPT')"; ls -la "$RUN_DIR" 2>/dev/null | head -20
  exit 2
fi
echo "    CKPT=$CKPT"

python -c "import scipy, sklearn" 2>/dev/null || pip install --quiet scipy scikit-learn || true
cd "$REPO"

# tag|extra flags. Baseline first so a partial job still yields the reference point.
#   baseline            = production behaviour (dense prefill, top-k at decode only)
#   prefill_topkNNpct   = the same top-k rule applied to EVERY prefill query as well
#   alpha defaults to each compressive layer's nonselected_landmark_mass (0.1), matching decode;
#   the *_alpha0 configs hard-drop non-selected blocks instead.
DEFAULT_CONFIGS="baseline_decode_only|;prefill_topk10pct|--prefill-topk-fraction 0.1;prefill_topk25pct|--prefill-topk-fraction 0.25;prefill_topk50pct|--prefill-topk-fraction 0.5;prefill_topk10pct_alpha0|--prefill-topk-fraction 0.1 --prefill-nonselected-mass 0"
CONFIGS="${CONFIGS:-$DEFAULT_CONFIGS}"

rc=0
IFS=';' read -r -a CFG_ARR <<< "$CONFIGS"
for entry in "${CFG_ARR[@]}"; do
  TAG="${entry%%|*}"
  EXTRA="${entry#*|}"
  OUT="$EVAL_OUT_DIR/${TASK}_${TAG}.json"
  if [ -f "$OUT" ]; then echo "SKIP $TAG (exists: $OUT)"; continue; fi
  PORT=$(( 20000 + RANDOM % 20000 ))
  echo "=== [$TAG] flags='$EXTRA' -> $OUT ($(date -u '+%T')Z) ==="
  torchrun --nproc_per_node="$NGPU" --master_port="$PORT" \
    debug/prefill_topk/eval_lc_native_prefill_topk.py \
    --model-path "$CKPT" --out "$OUT" --tokenizer "$TOKENIZER" \
    --prompt-format "$PROMPT_FORMAT" --max-length "$MAX_LENGTH" \
    --root "$BUNDLE" --max-test-samples "$MAX_TEST" --batch-size "$BATCH_SIZE" \
    --skip-ruler --skip-gen --ladder --ladder-version v2 \
    --ladder-tasks "$TASK" --ladder-rungs "$RUNGS" --contra-max-new-tokens 512 \
    $EXTRA
  crc=$?; [ $crc -ne 0 ] && rc=$crc
  echo "=== [$TAG] rc=$crc $(date -u '+%T')Z ==="
  [ -f "$OUT" ] && cat "$OUT"
done

echo "=== DONE rc=$rc results in $EVAL_OUT_DIR $(date -u '+%F %T')Z ==="
exit $rc
