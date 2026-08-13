#!/bin/bash
# ON-BEAKER multi-rung long-context eval runner for **HuggingFace-format** models.
#
# The hf-backend twin of ../singletask_ladder/run_beaker_multirung_eval.sh. Same eval data, same
# ladder table (both source ladder_rungs.sh), same scorer -- the ONLY difference is that the model
# is built by transformers instead of olmo_core, so third-party checkpoints we cannot express in
# olmo_core land in the same rows as our own runs.
#
# Built for HiLS-Attention-7B (chunk-wise sparse attention, out-of-tree modeling code) and its
# control, the Olmo-3-1025-7B base it was continued-pretrained from. Works for any HF causal LM.
#
# Env in (set by the launcher):
#   MODEL        ABSOLUTE path to a weka-staged HF checkpoint dir (config.json + safetensors)
#   MODEL_NAME   short label for result files / the flat $RESULTS mirror
#   TASK         contra|nq|rerank|outlier|oolong|fiqa|scifact|outlier_review|contra_fever
#   WEKA_LLM     weka ai2-llm root (e.g. /weka/oe-training-default/ai2-llm)
#   PROMPT_FORMAT   raw (default; these are BASE models) | chat
#   CHAT_TEMPLATE   jinja file for PROMPT_FORMAT=chat; default = the repo's olmo3_chatml.jinja
#   TOKENIZER    default: $MODEL itself (an HF dir carries its own tokenizer -- unlike a distcp
#                step dir, which is why the olmo_core runner has to infer a family and this
#                one must not)
#   ATTN_IMPL    dense-layer attention impl; empty -> the harness probes fa3 -> fa2 -> sdpa
#   MAX_TEST 600 | MAX_LENGTH 40960 | BATCH_SIZE 8 | NGPU 8 | LADDER_VERSION v2
#   LADDER_XLONG 0|1 , XLONG_ONLY 0|1 , XLONG_RUNGS "64k,128k" , EVAL_TAG , EVAL_OUT_DIR , RESULTS
set -uo pipefail
TASK="${TASK:?set TASK}"
MODEL="${MODEL:?set MODEL=<abs path to HF checkpoint dir>}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=<results label>}"
WEKA_LLM="${WEKA_LLM:?set WEKA_LLM=<weka ai2-llm root>}"
MAX_TEST="${MAX_TEST:-600}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NGPU="${NGPU:-8}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
QUERY_POSITION="${QUERY_POSITION:-both}"
ATTN_IMPL="${ATTN_IMPL:-}"
EVAL_TAG="${EVAL_TAG:-}"
SUF="${EVAL_TAG:+_$EVAL_TAG}"

REPO="${REPO:-$PWD}"
SLDIR="$REPO/src/scripts/train/memexpress/singletask_ladder"
HFDIR="$REPO/src/scripts/train/memexpress/hils_eval"

if [ ! -f "$MODEL/config.json" ]; then
  echo "ERROR: $MODEL has no config.json -- not an HF checkpoint dir."; ls -la "$MODEL" 2>&1 | head -20; exit 2
fi
TOKENIZER="${TOKENIZER:-$MODEL}"

# ---- HiLS models need their out-of-tree runtime (tilelang/veomni + the HiLS repo) --------------
IS_HILS=$(python - "$MODEL/config.json" <<'PYEOF'
import json, sys
print("1" if "hils" in str(json.load(open(sys.argv[1])).get("model_type", "")) else "0")
PYEOF
)
if [ "$IS_HILS" = "1" ]; then
  echo "=== HiLS checkpoint detected -- installing the HiLS runtime ==="
  # shellcheck disable=SC1091
  source "$HFDIR/hils_env_setup.sh" || { echo "ERROR: HiLS env setup failed"; exit 2; }
  # HiLS ties its chunk grid and sliding window to ABSOLUTE position, exactly like our landmark and
  # compressive variants: left-padding a batch shifts every chunk boundary, so batched generation
  # does not merely slow down, it changes the mask. bs=1, unconditionally.
  echo "    [hils] forcing BATCH_SIZE=1 (chunk grid is absolute-position-tied)"
  BATCH_SIZE=1
fi

# ---- position-budget guard --------------------------------------------------------------------
# A checkpoint whose max_position_embeddings is BELOW the prompts we intend to feed it does not
# error -- it reads garbage positions past the ceiling and the rung looks like a long-context
# collapse that is really a config mismatch (the trap the Qwen3.5 256k sweep fell into). Olmo-3
# base ships 65536 while HiLS-7B ships 131072, so the control and the treatment hit this at
# DIFFERENT rungs; say so loudly rather than let it be discovered in the numbers.
MAXPOS=$(python - "$MODEL/config.json" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1])).get("max_position_embeddings", 0))
PYEOF
)
echo "    MODEL=$MODEL (max_position_embeddings=$MAXPOS)"

PRASANNS="$WEKA_LLM/checkpoints/prasanns"
BUNDLE="${BUNDLE:-$PRASANNS/_eval_bundle}"
LADDER_VERSION="${LADDER_VERSION:-v2}"
if [ "$LADDER_VERSION" != "v2" ] && [ "$LADDER_VERSION" != "v3" ] && [ "$LADDER_VERSION" != "fast" ]; then
  echo "ERROR: LADDER_VERSION=$LADDER_VERSION is not supported -- v2, v3 and fast are the ladders." >&2
  exit 2
fi
# Bundle selection mirrors run_beaker_multirung_eval.sh EXACTLY, including the v2 else-branch that
# once went missing and killed every v2 job at startup under `set -u`.
if [ "$LADDER_VERSION" = "fast" ]; then
  EVAL500="${EVAL500:-$PRASANNS/_eval_bundle_eval500_v2_fast}"
elif [ "$LADDER_VERSION" = "v3" ]; then
  EVAL500="${EVAL500:-$PRASANNS/_eval_bundle_eval500_v3}"
else
  EVAL500="${EVAL500:-$PRASANNS/_eval_bundle_eval500_v2_clean}"
fi
VFLAG="--ladder-version $LADDER_VERSION"
LADDER_XLONG="${LADDER_XLONG:-0}"
XLONG_RUNGS="${XLONG_RUNGS:-64k,128k}"
XLFLAG=""
[ "$LADDER_XLONG" = "1" ] && XLFLAG="--xlong"
RESULTS="${RESULTS:-$PRASANNS/_eval_results}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:?set EVAL_OUT_DIR=<abs weka dir for result JSONs>}"
RUN_TAG="${MODEL_NAME}${SUF}"
[ -n "$SUF" ] && EVAL_OUT_DIR="${EVAL_OUT_DIR%/}${SUF}"

export PYTHONPATH="$REPO/src/scripts:$REPO/src:${PYTHONPATH:-}"
export EVAL500_ROOT="$EVAL500"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$EVAL_OUT_DIR" "$RESULTS"

echo "=== BEAKER hf-backend eval | host=$(hostname) MODEL_NAME=$MODEL_NAME TASK=$TASK NGPU=$NGPU START=$(date -u '+%F %T')Z ==="
echo "    BUNDLE=$BUNDLE"
echo "    EVAL500=$EVAL500"
echo "    PROMPT_FORMAT=$PROMPT_FORMAT TOKENIZER=$TOKENIZER"
nvidia-smi -L 2>/dev/null | head -8 || true

# ---- chat template: BASE models ship none, so "chat" needs an explicit one ---------------------
CT_FLAG=""
if [ "$PROMPT_FORMAT" = "chat" ]; then
  CHAT_TEMPLATE="${CHAT_TEMPLATE:-$REPO/src/scripts/ctc_eval/lib/chat_templates/olmo3_chatml.jinja}"
  [ -f "$CHAT_TEMPLATE" ] || { echo "ERROR: CHAT_TEMPLATE=$CHAT_TEMPLATE not found"; exit 2; }
  CT_FLAG="--chat-template $CHAT_TEMPLATE"
  echo "    CHAT_TEMPLATE=$CHAT_TEMPLATE"
fi
AT_FLAG=""
[ -n "$ATTN_IMPL" ] && AT_FLAG="--attn-impl $ATTN_IMPL"

python -c "import scipy, sklearn" 2>/dev/null || pip install --quiet scipy scikit-learn || true

cd "$REPO"
PORT=$(( 20000 + RANDOM % 20000 ))

# ---- rungs: the SHARED table (see ladder_rungs.sh) ---------------------------------------------
# shellcheck disable=SC1091
. "$SLDIR/ladder_rungs.sh"

if [ "$LADDER_XLONG" = "1" ]; then
  case "$TASK" in
    contra|nq|outlier|rerank|oolong)
      if [ "${XLONG_ONLY:-0}" = "1" ]; then RUNGS="$XLONG_RUNGS"; else RUNGS="$RUNGS,$XLONG_RUNGS"; fi
      BATCH_SIZE=1
      # Same caps as the olmo_core runner: rung label + ~10% (built prompts run 0.4-3.3% OVER the
      # label, and truncating the TAIL removes the question -> f1 0.000 at parse_rate 1.0).
      # eval_lc_native.py re-derives this itself and raises an undersized value.
      case ",$XLONG_RUNGS," in
        *,512k,*) MAX_LENGTH=578765;  PREFILL_CHUNK_SIZE=32768 ;;
        *,256k,*) MAX_LENGTH=290406;  PREFILL_CHUNK_SIZE=32768 ;;
        *,128k,*) MAX_LENGTH=146227 ;;
        *)        MAX_LENGTH=68608  ;;
      esac
      # NOTE: PREFILL_CHUNK_SIZE is an olmo_core-backend knob (its generation module implements
      # chunked prefill). The hf backend has no equivalent, so a >=256k rung here is a one-shot
      # prefill and will very likely OOM on one 80GB card. Kept in the env for symmetry, but the
      # launcher should not send this runner past 128k without a memory plan.
      [ -n "${PREFILL_CHUNK_SIZE:-}" ] && export PREFILL_CHUNK_SIZE
      echo "    [xlong] RUNGS=$RUNGS MAX_LENGTH=$MAX_LENGTH BATCH_SIZE=$BATCH_SIZE" ;;
    *) echo "    [xlong] no xlong rungs for TASK=$TASK; base ladder unchanged." ;;
  esac
fi

# The model's own ceiling vs what we are about to feed it. Compared AFTER the xlong block, because
# that is where MAX_LENGTH actually gets its final value.
if [ "$MAXPOS" != "0" ] && [ "$MAX_LENGTH" -gt "$MAXPOS" ]; then
  echo "    ⚠ MAX_LENGTH=$MAX_LENGTH EXCEEDS the checkpoint's max_position_embeddings=$MAXPOS."
  echo "      This measures position EXTRAPOLATION, not in-ceiling capability. Label every number"
  echo "      from this job accordingly -- and note the HiLS/Olmo-3 pair does NOT share a ceiling"
  echo "      (131072 vs 65536), so the same rung is in-ceiling for one and not the other."
fi

case "$LADDER_VERSION" in
  v2) OUT="$EVAL_OUT_DIR/${TASK}_multirung.json" ;;
  *)  OUT="$EVAL_OUT_DIR/${TASK}_multirung_${LADDER_VERSION}.json" ;;
esac
echo "=== EVAL $TASK rungs=$RUNGS ladder=$LADDER_VERSION backend=hf -> $OUT ($(date -u '+%T')Z) ==="
torchrun --nproc_per_node="$NGPU" --master_port="$PORT" src/scripts/ctc_eval/eval/eval_lc_native.py \
  --backend hf --model-path "$MODEL" --out "$OUT" --tokenizer "$TOKENIZER" \
  --prompt-format "$PROMPT_FORMAT" --query-position "$QUERY_POSITION" $CT_FLAG $AT_FLAG \
  --max-length "$MAX_LENGTH" --root "$BUNDLE" --max-test-samples "$MAX_TEST" \
  --batch-size "$BATCH_SIZE" --skip-ruler --skip-gen \
  --ladder $VFLAG $XLFLAG --ladder-tasks "$LTASK" --ladder-rungs "$RUNGS" $EXTRA
rc=$?

case "$LADDER_VERSION" in
  v2) RES_BASE="$RESULTS/${RUN_TAG}_${TASK}_multirung" ;;
  *)  RES_BASE="$RESULTS/${RUN_TAG}_${TASK}_multirung_${LADDER_VERSION}" ;;
esac
if [ -f "$OUT" ]; then
  cp "$OUT" "${RES_BASE}.json" 2>/dev/null || true
  GEN="${OUT%.json}.generations.jsonl"
  [ -f "$GEN" ] && cp "$GEN" "${RES_BASE}.generations.jsonl" 2>/dev/null || true
  echo "--- $OUT ---"; cat "$OUT"
  # Reading generations is not optional for a BASE model: an unaligned model that ignores the
  # answer format scores ~0 for a reason that has nothing to do with long context, and the only
  # way to tell that apart from a real capability result is to look at the text.
  [ -f "$GEN" ] && python src/scripts/ctc_eval/eval/print_gen_sample.py "$GEN" "${GEN_SAMPLE_N:-6}" || true
fi
echo "=== DONE TASK=$TASK rc=$rc result=${RES_BASE}.json $(date -u '+%F %T')Z ==="
exit $rc
