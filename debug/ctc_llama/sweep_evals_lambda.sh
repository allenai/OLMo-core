#!/bin/bash
# Fan out the 2k/4k/8k/16k eval ladder for one Llama CTC checkpoint on the LAMBDA cluster.
# Run this ON the lambda head node:
#   bash $LROOT/ctc_llama/sweep_evals_lambda.sh <task> <arm> <ckpt-dir> [gpus-per-rung]
set -o pipefail
TASK="${1:?task}"; ARM="${2:?arm}"; CKPT="${3:?ckpt dir}"; GPUS="${4:-2}"
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
case "$ARM" in
  full)        VARIANT=dense ;;
  chunked-mix) VARIANT=chunked ;;
  *) echo "arm must be full|chunked-mix"; exit 1 ;;
esac
case "$TASK" in
  contradiction) RUNGDIR=contradiction ;;
  qdmatch)       RUNGDIR=qdmatch_hpqa ;;
  *) echo "task must be contradiction|qdmatch"; exit 1 ;;
esac
for RUNG in 2048 4096 8192 16384; do
  EVAL_JSONL=$LROOT/eval_rungs/$RUNGDIR/rung_${RUNG}.jsonl
  # +25% slack over the rung label: it was calibrated on the Qwen tokenizer and the same text is
  # longer under Llama's BPE. Too small a max_length silently skips prompts and scores them empty.
  MAXLEN=$((RUNG + RUNG / 4 + 3072))
  sbatch --partition=lambda --account=site --qos=preemptive_high -w lambda-hyperplane05 \
    --gres=gpu:A100:$GPUS --nodes=1 --job-name="ev-${TASK}-${ARM}-${RUNG}" \
    --export=ALL,CKPT="$CKPT",TASK="$TASK",VARIANT="$VARIANT",ARM="$ARM",RUNG="$RUNG",EVAL_JSONL="$EVAL_JSONL",NGPU="$GPUS",MAXLEN="$MAXLEN" \
    $LROOT/ctc_llama/eval_llama_lambda.sbatch
done
squeue -u prasann -o '%.8i %.28j %.8T %.8M %N'
