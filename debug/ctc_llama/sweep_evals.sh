#!/bin/bash
# Fan out the 2k/4k/8k/16k eval ladder for one Llama CTC checkpoint (one task x one arm) on the
# Berkeley cluster. Each rung gets its own job with a UNIQUE --master-port (concurrent evals
# otherwise collide on torchrun's default 29500 and all but one die in ~3s) and its own OUTROOT
# subtree, and is node-pinned to the host that holds the checkpoint on /data.
#
#   bash debug/ctc_llama/sweep_evals.sh <node> <task> <arm> <ckpt-dir> [gpus-per-rung]
#     task: contradiction | qdmatch     arm: full | chunked-mix
#
# Arm -> driver flags: full = --variant dense --arm full ; chunked-mix = --variant chunked
# --arm chunked-mix (the driver's "dense" means full attention; see run_rung_eval.py).
set -uo pipefail
NODE="${1:?node}"; TASK="${2:?task}"; ARM="${3:?arm}"; CKPT="${4:?ckpt dir}"; GPUS="${5:-2}"
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
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
PORT=29200
for RUNG in 2048 4096 8192 16384; do
  PORT=$((PORT + 17))
  EVAL_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/$RUNGDIR/rung_${RUNG}.jsonl
  # MAXLEN >= rung + task max_new + 512 (the driver enforces this too), PLUS 25% slack: the rung
  # labels were calibrated on the Qwen tokenizer, and the same text tokenizes longer under Llama's
  # 128k BPE. Without the slack the 16k rung's prompts get skipped as too long and are scored as
  # empty -- metric 0.000 at parse_rate 1.0, which reads as a model failure (maxlen-truncation trap).
  MAXLEN=$((RUNG + RUNG / 4 + 3072))
  sbatch --nodelist="$NODE" --gres=gpu:H200:$GPUS --job-name="ev-${TASK}-${ARM}-${RUNG}" \
    --export=ALL,CKPT="$CKPT",TASK="$TASK",VARIANT="$VARIANT",ARM="$ARM",RUNG="$RUNG",\
EVAL_JSONL="$EVAL_JSONL",NGPU="$GPUS",MAXLEN="$MAXLEN",MASTER_PORT="$PORT" \
    "$REPO/debug/ctc_llama/eval_llama_native.sbatch"
done
squeue -u prasann -o '%.11i %.26j %.8T %.10M %R'
