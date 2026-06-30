#!/bin/bash
# ON-BEAKER multi-rung NATIVE long-context eval runner (8-GPU data-parallel torchrun).
#
# This is the on-node half of the Beaker eval flow. It is uploaded to the weka eval bundle by
# `upload_lc_eval_bundle.sh` and invoked inside a gantry job by `run_q4b_beaker_multirung_eval.py`.
# It mirrors the LOCAL driver `run_q4b_stl_multirung_eval.sbatch`, but reads EVERYTHING from weka
# (eval code + eval data + the checkpoint) so nothing has to be synced to the Beaker node:
#
#   * eval CODE   : corpus-reasoning `scripts/` tree under  $BUNDLE                (PYTHONPATH + cwd)
#   * `data/...`  : relative ladder/base files under         $BUNDLE/data          (--root=$BUNDLE)
#   * `_600/_500` : the goal-rung ladder files under          $EVAL500             (EVAL500_ROOT env)
#   * checkpoint  : the just-trained distcp step dir under     $RUN_DIR/step*       (auto-globbed)
#
# Native olmo_core generate (NO HF / NO vLLM) so it works for dense / landmark / compressive /
# docchunk, and runs 8-way DP via `torchrun --nproc_per_node=8`.
#
# Env in (set by the launcher):
#   RUN         run name (checkpoints live at $WEKA_LLM/checkpoints/prasanns/$RUN/step*)
#   TASK        contra | nq | rerank | outlier | oolong
#   VARIANT     dense | landmark | compressive | docchunk   (docchunk -> OOLONG only)
#   WEKA_LLM    weka ai2-llm root (e.g. /weka/oe-training-default/ai2-llm)
#   STEP        optional: pin a specific step dir (e.g. step580); empty -> latest complete step
#   MAX_TEST    default 600 ; MAX_LENGTH default 40960 ; BATCH_SIZE default 8 ; NGPU default 8
set -uo pipefail
TASK="${TASK:?set TASK=contra|nq|rerank|outlier|oolong}"
VARIANT="${VARIANT:?set VARIANT=dense|landmark|compressive|docchunk}"
RUN="${RUN:?set RUN=<run name>}"
WEKA_LLM="${WEKA_LLM:?set WEKA_LLM=<weka ai2-llm root>}"
STEP="${STEP:-}"
MAX_TEST="${MAX_TEST:-600}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NGPU="${NGPU:-8}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"   # chat=SFT (apply_chat_template) | raw=BASE/CPT | alpaca=legacy

PRASANNS="$WEKA_LLM/checkpoints/prasanns"
BUNDLE="${BUNDLE:-$PRASANNS/_eval_bundle}"
EVAL500="${EVAL500:-$PRASANNS/_eval_bundle_eval500}"
RESULTS="${RESULTS:-$PRASANNS/_eval_results}"
RUN_DIR="$PRASANNS/$RUN"
# Where the per-task result JSONs land. Default = the run's own eval/ dir under prasanns/<RUN>.
# Override via the launcher's --results-dir (forwarded as EVAL_OUT_DIR) to write anywhere on weka.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$RUN_DIR/eval}"

REPO="${REPO:-$PWD}"                          # cloned OLMo-core repo (gantry cwd); eval CODE = in-repo ctc_eval
export PYTHONPATH="$REPO/src/scripts:$REPO/src:${PYTHONPATH:-}"   # so `import ctc_eval...` resolves (olmo_core also pip -e)
export EVAL500_ROOT="$EVAL500"                # eval_lc_native.py reads the _600/_500 rungs from here (weka data)
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM at long ctx on smaller GPUs
mkdir -p "$EVAL_OUT_DIR" "$RESULTS"

echo "=== BEAKER multirung eval | host=$(hostname) RUN=$RUN TASK=$TASK VARIANT=$VARIANT NGPU=$NGPU START=$(date -u '+%F %T')Z ==="
echo "    BUNDLE=$BUNDLE"
echo "    EVAL500=$EVAL500"
nvidia-smi -L 2>/dev/null | head -8 || true

# ---- resolve the checkpoint step dir (CKPT override > STEP pin > latest complete step) ----
if [ -n "${CKPT:-}" ]; then
  :  # explicit absolute step dir (e.g. for a one-off validation against any weka checkpoint)
elif [ -n "$STEP" ]; then
  CKPT="$RUN_DIR/$STEP"
else
  CKPT=""
  for d in $(ls -d "$RUN_DIR"/step*/ 2>/dev/null | sed 's#/$##' | sort -V); do
    [ -f "$d/model_and_optim/.metadata" ] && CKPT="$d"   # keep last (highest step) that is complete
  done
fi
CKPT="${CKPT%/}"
if [ -z "$CKPT" ] || [ ! -f "$CKPT/model_and_optim/.metadata" ]; then
  echo "ERROR: no complete step dir (config.json + model_and_optim/.metadata) under $RUN_DIR (CKPT='$CKPT')"
  echo "       contents:"; ls -la "$RUN_DIR" 2>/dev/null | head -20
  exit 2
fi
echo "    CKPT=$CKPT"

# ---- make sure the rerank/outlier metric deps are importable (lazy scipy/sklearn) ----
python -c "import scipy, sklearn" 2>/dev/null || pip install --quiet scipy scikit-learn || true

cd "$REPO"   # CODE is in-repo (ctc_eval); DATA comes from weka via --root "$BUNDLE" + EVAL500_ROOT
PORT=$(( 20000 + RANDOM % 20000 ))
TR="torchrun --nproc_per_node=$NGPU --master_port=$PORT src/scripts/ctc_eval/eval/eval_lc_native.py --prompt-format $PROMPT_FORMAT"

# ---- docchunk: box-marker prefill eval; only OOLONG is wired (eval_lc_native_docchunk.py) ----
if [ "$VARIANT" = "docchunk" ]; then
  if [ "$TASK" != "oolong" ]; then
    echo "NOTE: docchunk native eval supports OOLONG only; skipping TASK=$TASK."; exit 0
  fi
  rc=0
  for rung in 1024 2048 4096 8192 16384 32768; do
    DF="$BUNDLE/data/oolong_test_synth_ctx${rung}_spliteval.jsonl"
    [ -f "$DF" ] || { echo "[docchunk oolong ctx$rung] MISSING $DF, skipping"; continue; }
    O="$EVAL_OUT_DIR/oolong_docchunk_${rung}.json"
    echo "=== docchunk oolong ctx$rung -> $O ==="
    torchrun --nproc_per_node="$NGPU" --master_port="$PORT" src/scripts/ctc_eval/eval/eval_lc_native_docchunk.py \
      --variant dense --model-path "$CKPT" --out "$O" --tokenizer "$TOKENIZER" \
      --oolong-data "$DF" --max-test-samples "$MAX_TEST" --max-length "$MAX_LENGTH" --mem-freq 63 || rc=$?
    [ -f "$O" ] && cp "$O" "$RESULTS/${RUN}_oolong_docchunk_${rung}.json" 2>/dev/null || true
  done
  echo "=== DONE docchunk rc=$rc $(date -u '+%F %T')Z ==="; exit $rc
fi

# ---- rerank: CE-graded eval files (k20~3k, k50~8k, k100~16k); eval each at the base ladder rung ----
if [ "$TASK" = "rerank" ]; then
  rc=0
  for pair in "3k:data/msmarco_trainhn_eval_k20_500.jsonl" \
              "8k:data/msmarco_trainhn_eval_k50_500.jsonl" \
              "16k:data/msmarco_trainhn_eval_k100_500.jsonl"; do
    r="${pair%%:*}"; CEF="${pair#*:}"
    [ -f "$BUNDLE/$CEF" ] || { echo "[rerank CE @${r}] MISSING $CEF, skipping"; continue; }
    O="$EVAL_OUT_DIR/rerank_ce_${r}.json"
    echo "=== rerank CE @${r} ($CEF) -> $O ==="
    $TR --model-path "$CKPT" --out "$O" --tokenizer "$TOKENIZER" --max-length "$MAX_LENGTH" \
        --root "$BUNDLE" --max-test-samples "$MAX_TEST" --batch-size "$BATCH_SIZE" --skip-ruler --skip-gen \
        --ladder --ladder-tasks rerank --ladder-rungs 2k --rerank-data "$CEF" || rc=$?
    [ -f "$O" ] && cp "$O" "$RESULTS/${RUN}_rerank_ce_${r}.json" 2>/dev/null || true
  done
  echo "=== DONE rerank-CE rc=$rc $(date -u '+%F %T')Z ==="; exit $rc
fi

# ---- dense / landmark / compressive: standard multi-rung ladder (NDCG/F1/score per rung) ----
case "$TASK" in
  contra)  RUNGS="2k,8k,16k,32k"; LTASK=contradiction; EXTRA="--contra-data data/contradiction_eval_pubmed_both_n100_k3.jsonl --contra-max-new-tokens 512" ;;
  nq)      RUNGS="3k,8k,16k,32k"; LTASK=nq;            EXTRA="--nq-data data/n2ified_eval_nq_q50.jsonl" ;;
  outlier) RUNGS="3k,8k,16k,32k"; LTASK=outlier;       EXTRA="--outlier-data data/outlier_wiki100w_n55_k3_eval_600.jsonl" ;;
  oolong)  RUNGS="8k,16k,32k";    LTASK=oolong;        EXTRA="" ;;
  *) echo "ERROR unknown TASK=$TASK"; exit 2 ;;
esac
OUT="$EVAL_OUT_DIR/${TASK}_multirung.json"
echo "=== EVAL $TASK rungs=$RUNGS -> $OUT ($(date -u '+%T')Z) ==="
$TR --model-path "$CKPT" --out "$OUT" --tokenizer "$TOKENIZER" --max-length "$MAX_LENGTH" \
    --root "$BUNDLE" --max-test-samples "$MAX_TEST" --batch-size "$BATCH_SIZE" --skip-ruler --skip-gen \
    --ladder --ladder-tasks "$LTASK" --ladder-rungs "$RUNGS" $EXTRA
rc=$?
if [ -f "$OUT" ]; then
  cp "$OUT" "$RESULTS/${RUN}_${TASK}_multirung.json" 2>/dev/null || true
  echo "--- $OUT ---"; cat "$OUT"
fi
echo "=== DONE TASK=$TASK rc=$rc result=$RESULTS/${RUN}_${TASK}_multirung.json $(date -u '+%F %T')Z ==="
exit $rc
