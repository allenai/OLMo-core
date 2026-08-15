#!/usr/bin/env bash
# Re-evaluate the Llama-3.2-3B contradiction arms on the contradiction_iid ladder, on Berkeley.
#
#   bash debug/ctc_crossfamily/sweep_llama_iid.sh [node]
#
# ── WHY ───────────────────────────────────────────────────────────────────────────────────────
# The published Llama contradiction numbers are on `eval_rungs/contradiction`; Qwen's are on
# `contradiction_clean`. Those are different corpora, so the cross-family figure was comparing
# models on different inputs. contradiction_iid is the ladder matching the contradiction_train
# shard all arms trained on, and where the Qwen3.5-4B reference lives.
#
# No transfer is needed: both arms already sit on cubbins' node-local disk.
#
# ⚠ MAXLEN=32768 + ALLOW_SHORT ON BOTH ARMS, DELIBERATELY.
# Under the Llama tokenizer the 16k contradiction rung has p50 ~29k and max ~101k tokens, so no
# affordable budget fits every prompt (fitting the max would mean a ~100k-token KV cache per
# example). run_rung_eval hard-fails rather than silently truncating, which is right. The matched
# choice is to score BOTH arms at the SAME budget so both lose the SAME examples, and 32768 is
# what the OLMo ladder used -- so this is also comparable across families. The skipped count is
# recorded in each result JSON; read it before quoting the 16k rung.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NODE="${1:-cubbins}"
GPUS="${GPUS:-2}"
CKPT_ROOT="${CKPT_ROOT:-/data/prasann/ctc_suite/ckpts}"
RUNGDIR=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction_iid
OUTROOT="$REPO/results/ctc_suite_llama_iid"
MAXLEN=32768
mkdir -p "$OUTROOT"

PORT=29400
for ARM in full chunked-mix; do
  case "$ARM" in
    full)        VARIANT=dense;   CK="$CKPT_ROOT/llama32-3b-contra-full" ;;
    chunked-mix) VARIANT=chunked; CK="$CKPT_ROOT/llama32-3b-contra-chunked-mix" ;;
  esac
  for RUNG in 2560 4096 8192 16384; do
    PORT=$((PORT + 13))
    sbatch --nodelist="$NODE" --partition=jsteinhardt --qos=preemptive_high --account=site \
      --gres=gpu:H200:$GPUS --job-name="llid-${ARM}-${RUNG}" \
      --export=ALL,CKPT="$CK",TASK=contradiction,VARIANT="$VARIANT",ARM="$ARM",RUNG="$RUNG",\
EVAL_JSONL="$RUNGDIR/rung_${RUNG}.jsonl",NGPU="$GPUS",MAXLEN="$MAXLEN",ALLOW_SHORT=1,\
OUTROOT="$OUTROOT",MASTER_PORT="$PORT" \
      "$REPO/debug/ctc_llama/eval_llama_native.sbatch"
  done
done
squeue -u prasann -o '%.11i %.24j %.8T %.9M %R'
