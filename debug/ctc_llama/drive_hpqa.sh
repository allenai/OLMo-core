#!/bin/bash
# Autonomous driver for the Llama hotpotqa (RETRIEVAL) row: wait for the tokenized shard -> train
# both arms -> chain the 2k/4k/8k/16k eval ladder behind each arm.
#
# hotpotqa is the O(N) half of the CTC hypothesis (expect dense ~= chunked), the counterpart to the
# O(N*M) qdmatch row (expect dense >> chunked). It routes through --task retrieval (gold_id_f1).
#
# Run detached from the login node:
#   setsid nohup bash debug/ctc_llama/drive_hpqa.sh > /scratch/users/prasann/ctc_llama_logs/drive_hpqa.log 2>&1 < /dev/null &
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
say() { echo "[drive_hpqa $(date '+%F %T')] $*"; }

# ---- 1. wait for the tokenization job (submitted separately) to publish the shard ----
say "waiting for $LOGDIR/HPQA_SHARD_READY"
while [ ! -f "$LOGDIR/HPQA_SHARD_READY" ]; do
  if ! squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -q 'llama-tok-hpqa'; then
    [ -f "$LOGDIR/HPQA_SHARD_READY" ] || { say "FATAL: tokenization job gone and no shard"; exit 3; }
  fi
  sleep 60
done
MAXLEN_EX=$(python3 -c "import json;print(json.load(open('$LOGDIR/shard_meta_hotpotqa.json'))['max_example_len'])")
NINST=$(python3 -c "import json;print(json.load(open('$LOGDIR/shard_meta_hotpotqa.json'))['num_instances'])")
NDROP=$(python3 -c "import json;print(json.load(open('$LOGDIR/shard_meta_hotpotqa.json'))['num_dropped'])")
# Train at the smallest multiple of 512 that still covers the longest example: PadToLength pads
# EVERY instance up to seq_len, so any slack above max_example_len is pure wasted compute and
# padding is the dominant cost driver here.
SEQ_LEN=$(( ((MAXLEN_EX + 511) / 512) * 512 ))
say "shard: num_instances=$NINST dropped=$NDROP max_example_len=$MAXLEN_EX -> SEQ_LEN=$SEQ_LEN"

# ---- 2. train both arms (4 GPUs each = the 8-GPU preemptive_high cap) ----
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

# ---- 3. chain the eval ladder behind each arm ----
# MAXLEN is deliberately NOT passed: run_rung_eval now auto-sizes --max-length from the MEASURED
# prompt distribution of the rung file and hard-fails rather than silently scoring skipped examples
# 0 at parse_rate 1.0. retrieval's decode budget is only 64 tokens, so a generous auto-sized
# max_length is cheap here (unlike contradiction, whose 101k-token tail forced a matched-budget
# compromise).
PORT=29900
for V in full chunked-mix; do
  case "$V" in full) VARIANT=dense;; chunked-mix) VARIANT=chunked;; esac
  for RUNG in 2048 4096 8192 16384; do
    PORT=$((PORT + 31))
    sbatch --nodelist=cubbins --gres=gpu:H200:2 --job-name="ev-hpqa-${V}-${RUNG}" \
      --dependency=afterany:${TRAIN[$V]} \
      --export=ALL,CKPT=/data/prasann/ctc_suite/ckpts/llama32-3b-hpqa-$V,TASK=hotpotqa,VARIANT=$VARIANT,ARM=$V,RUNG=$RUNG,EVAL_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/hotpotqa/rung_${RUNG}.jsonl,NGPU=2,MASTER_PORT=$PORT \
      "$REPO/debug/ctc_llama/eval_llama_native.sbatch"
  done
done
say "all hotpotqa jobs submitted"
squeue -u "$USER" -o '%.11i %.28j %.8T %.8M %R'
