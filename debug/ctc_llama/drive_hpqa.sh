#!/bin/bash
# Autonomous driver for the Llama hotpotqa (RETRIEVAL) row: wait for the raw pool -> tokenize ->
# train both arms -> chain the 2k/4k/8k/16k eval ladder behind each arm.
#
# hotpotqa is the O(N) half of the CTC hypothesis (expect dense ~= chunked), the counterpart to the
# O(N*M) qdmatch row (expect dense >> chunked). It routes through --task retrieval (gold_id_f1).
#
# Run detached from the login node:
#   nohup bash debug/ctc_llama/drive_hpqa.sh > /scratch/users/prasann/ctc_llama_logs/drive_hpqa.log 2>&1 &
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
RAW=$REPO/debug/ctc_multifamily/hpqa_raw
say() { echo "[drive_hpqa $(date '+%F %T')] $*"; }

# ---- 1. wait for the raw pool. The generator is owned by the coordinator and may run EITHER as a
#         login-node process OR as a slurm job (job name gen-hpqa-*), so wait on both: a bare
#         file-exists check would fire on a partially written pool (the loop appends one n-bucket
#         at a time) and silently train on a fraction of the ladder. ----
say "waiting for raw hotpotqa pool in $RAW"
while true; do
  n_files=$(ls "$RAW"/*.jsonl 2>/dev/null | wc -l)
  running_local=$(pgrep -u "$USER" -f generate_hotpotqa_data.py | wc -l)
  running_slurm=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -c 'gen-hpqa')
  if [ "$n_files" -gt 0 ] && [ "$running_local" -eq 0 ] && [ "$running_slurm" -eq 0 ]; then break; fi
  sleep 60
done
say "raw pool ready: $(ls "$RAW"/*.jsonl | wc -l) files, $(cat "$RAW"/*.jsonl | wc -l) lines"

# ---- 2. tokenize with the Llama marker set ----
rm -f "$LOGDIR/HPQA_SHARD_READY"
jid=$(sbatch --parsable "$REPO/debug/ctc_llama/tokenize_hpqa.sbatch")
say "tokenization job $jid"
while [ ! -f "$LOGDIR/HPQA_SHARD_READY" ]; do
  squeue -j "$jid" -h -o %T 2>/dev/null | grep -q . || { sleep 30; break; }
  sleep 60
done
[ -f "$LOGDIR/shard_meta_hotpotqa.json" ] || { say "FATAL: tokenization produced no metadata"; exit 3; }
MAXLEN_EX=$(python3 -c "import json;print(json.load(open('$LOGDIR/shard_meta_hotpotqa.json'))['max_example_len'])")
NINST=$(python3 -c "import json;print(json.load(open('$LOGDIR/shard_meta_hotpotqa.json'))['num_instances'])")
# Train at the smallest multiple of 512 that still covers the longest example: PadToLength pads
# EVERY instance to seq_len, so any slack above max_example_len is pure wasted compute.
SEQ_LEN=$(( ((MAXLEN_EX + 511) / 512) * 512 ))
say "shard: num_instances=$NINST max_example_len=$MAXLEN_EX -> SEQ_LEN=$SEQ_LEN"

# ---- 3. train both arms (4 GPUs each = the 8-GPU preemptive_high cap) ----
declare -A TRAIN
for V in full chunked-mix; do
  RUN=llama32-3b-hpqa-$V
  j=$(sbatch --parsable --partition=jsteinhardt --qos=preemptive_high --account=site -w cubbins \
    --gres=gpu:H200:4 --time=10:00:00 --job-name="$RUN" \
    --export=ALL,TASK=retrieval,DATA_SRC=/data/prasann/ctc_llama/shards/hotpotqa_train_llama,VARIANT=$V,SCALE=3b,MODEL_FAMILY=llama,BASE_SRC=/data/prasann/ctc_llama/bases/llama32-3b-base-fixmark,SEQ_LEN=$SEQ_LEN,EPOCHS=1,GLOBAL_BATCH=8,MICRO_BATCH=1,LR=5e-05,ACT_CKPT=full,SHARD_DEGREE=4,NGPU=4,RUN=$RUN,WANDB_GROUP=ctc-suite-llama-hotpotqa,NOWANDB=1 \
    "$REPO/src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch")
  TRAIN[$V]=$j
  say "train $V -> job $j (seq_len=$SEQ_LEN)"
done

# ---- 4. chain the eval ladder behind each arm ----
# --max-length is set explicitly and generously: the rung label bounds the DOCUMENT COUNT, not the
# prompt length, so the driver default (rung+2048) silently skips long prompts and scores them 0 at
# parse_rate 1.0. retrieval's decode budget is only 64 tokens, so a large max_length is cheap.
PORT=29700
for V in full chunked-mix; do
  case "$V" in full) VARIANT=dense;; chunked-mix) VARIANT=chunked;; esac
  for RUNG in 2048 4096 8192 16384; do
    PORT=$((PORT + 31))
    MAXLEN=32768; [ "$RUNG" -ge 16384 ] && MAXLEN=40960
    sbatch --nodelist=cubbins --gres=gpu:H200:2 --job-name="ev-hpqa-${V}-${RUNG}" \
      --dependency=afterany:${TRAIN[$V]} \
      --export=ALL,CKPT=/data/prasann/ctc_suite/ckpts/llama32-3b-hpqa-$V,TASK=hotpotqa,VARIANT=$VARIANT,ARM=$V,RUNG=$RUNG,EVAL_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/hotpotqa/rung_${RUNG}.jsonl,NGPU=2,MAXLEN=$MAXLEN,MASTER_PORT=$PORT \
      "$REPO/debug/ctc_llama/eval_llama_native.sbatch"
  done
done
say "all hotpotqa jobs submitted"
squeue -u "$USER" -o '%.11i %.28j %.8T %.8M %R'
