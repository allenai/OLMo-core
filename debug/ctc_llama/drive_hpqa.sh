#!/bin/bash
# Autonomous driver for the Llama hotpotqa (RETRIEVAL) row on sneetches: wait for node prep ->
# train both arms (4 GPUs each, IDENTICAL hardware and world size) -> chain the eval ladder.
#
# sneetches was chosen over cubbins because cubbins had 3 GPUs held at equal QOS (preemptive_high),
# leaving only 5 of 8 -- not enough for a symmetric 4+4 split. Running the two arms on different
# GPU counts would have made world_size differ between them, which changes data sharding and the
# mask-mix anneal granularity; keeping them identical isolates the mask as the only difference.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
say() { echo "[drive_hpqa $(date '+%F %T')] $*"; }

say "waiting for $LOGDIR/SNEETCHES_READY"
while [ ! -f "$LOGDIR/SNEETCHES_READY" ]; do
  squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -q 'llama-prep-sneetches' || {
    [ -f "$LOGDIR/SNEETCHES_READY" ] || { say "FATAL: prep job gone and node not ready"; exit 3; }
  }
  sleep 45
done
MAXLEN_EX=$(python3 -c "import json;print(json.load(open('$LOGDIR/shard_meta_hotpotqa.json'))['max_example_len'])")
NINST=$(python3 -c "import json;print(json.load(open('$LOGDIR/shard_meta_hotpotqa.json'))['num_instances'])")
# Smallest multiple of 512 covering the longest example: PadToLength pads EVERY instance to
# seq_len, so slack above max_example_len is pure wasted compute.
SEQ_LEN=$(( ((MAXLEN_EX + 511) / 512) * 512 ))
say "shard: num_instances=$NINST max_example_len=$MAXLEN_EX -> SEQ_LEN=$SEQ_LEN"

declare -A TRAIN
for V in full chunked-mix; do
  RUN=llama32-3b-hpqa-$V
  j=$(sbatch --parsable --partition=jsteinhardt --qos=preemptive_high --account=site -w sneetches \
    --gres=gpu:H200:4 --time=10:00:00 --job-name="$RUN" \
    --export=ALL,TASK=retrieval,DATA_SRC=/data/prasann/ctc_llama/shards/hotpotqa_train_llama,VARIANT=$V,SCALE=3b,MODEL_FAMILY=llama,BASE_SRC=/data/prasann/ctc_llama/bases/llama32-3b-base-fixmark,SEQ_LEN=$SEQ_LEN,EPOCHS=1,GLOBAL_BATCH=8,MICRO_BATCH=1,LR=5e-05,ACT_CKPT=full,SHARD_DEGREE=4,NGPU=4,RUN=$RUN,WANDB_GROUP=ctc-suite-llama-hotpotqa,NOWANDB=1 \
    "$REPO/src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch")
  TRAIN[$V]=$j
  say "train $V -> job $j (seq_len=$SEQ_LEN, 4 GPUs, sneetches)"
done

# Evals: node-pinned to sneetches (checkpoints are node-local /data). MAXLEN is deliberately NOT
# passed -- run_rung_eval auto-sizes it from the MEASURED prompt distribution and hard-fails on
# skips, and retrieval's 64-token decode budget makes a generous budget cheap.
PORT=30100
for V in full chunked-mix; do
  case "$V" in full) VARIANT=dense;; chunked-mix) VARIANT=chunked;; esac
  for RUNG in 2048 4096 8192 16384; do
    PORT=$((PORT + 31))
    sbatch --nodelist=sneetches --gres=gpu:H200:2 --job-name="ev-hpqa-${V}-${RUNG}" \
      --dependency=afterany:${TRAIN[$V]} \
      --export=ALL,CKPT=/data/prasann/ctc_suite/ckpts/llama32-3b-hpqa-$V,TASK=hotpotqa,VARIANT=$VARIANT,ARM=$V,RUNG=$RUNG,EVAL_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/hotpotqa/rung_${RUNG}.jsonl,NGPU=2,MASTER_PORT=$PORT \
      "$REPO/debug/ctc_llama/eval_llama_native.sbatch"
  done
done
say "all hotpotqa jobs submitted"
squeue -u "$USER" -o '%.11i %.28j %.8T %.8M %R'
