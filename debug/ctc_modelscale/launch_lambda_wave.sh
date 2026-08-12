#!/bin/bash
# Model-scale study (paper fig:modelscale extension): submit one wave of CTC-suite training runs
# to the lambda A100 cluster. 4 tasks x {full, chunked-mix} x {0.8b, 2b}, reusing the EXISTING
# 20k train shards already staged under $LROOT/data (no data rebuild -- explicit user requirement).
#
#   ./launch_lambda_wave.sh 0.8b      # wave 1
#   ./launch_lambda_wave.sh 2b        # wave 2 (only after the 2B base finishes staging)
#
# FAIR SHARE (lambda_cluster.md, user directive 2026-07-24): the *preempting* footprint is capped
# at two nodes -- at most 8 GPUs on preemptive_high and 8 on preemptive at any instant. Each run
# takes a whole 8-GPU node, so that means at most ONE running job per preempting QOS. This is
# enforced structurally with `--dependency=singleton`: all jobs sharing a (job-name, user) pair
# serialize, so lane A never has two jobs on preemptive_high simultaneously, likewise lane B on
# preemptive. Lane C runs at `normal`, which is exempt (it cannot preempt and is itself
# preemptible, so it only soaks idle GPUs) -- but it is serialized too, since there are only 3 live
# nodes total.
#
# Checkpoints go to node-local /tmp (lambda's /data is root-owned and NFS is a 1.30T per-user
# quota that 16 distcp checkpoints would blow through). /tmp persists across jobs on a node, so the
# harvest step must node-pin to the HOST printed in each job's log banner -- the ledger records it.
set -o pipefail
SCALE="${1:?usage: launch_lambda_wave.sh <0.8b|2b>}"
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
LAUNCHER="$LROOT/OLMo-core/src/scripts/train/memexpress/ctc_suite/run_ctc_lambda.sbatch"
LEDGER="$(dirname "$0")/LAUNCH_LEDGER.tsv"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

case "$SCALE" in
  0.8b) BASE=q35-08b-base-modelonly; TAG=08b ;;
  2b)   BASE=q35-2b-base-modelonly;  TAG=2b  ;;
  *) echo "FATAL: scale must be 0.8b or 2b"; exit 2 ;;
esac

# task  seq_len  data_src   -- seq_len must be >= the shard's max_example_len (PadToLength silently
# drops longer examples otherwise); these match the 4B fan-out exactly so the scales are comparable.
# Ordered by priority: contradiction is the paper's reference task, hotpotqa is the low-CTC anchor.
TASKS=(
  "contradiction 40960 contradiction_train"
  "hotpotqa      26112 hotpotqa_train"
  "reorder       40960 reorder_train"
  "qdmatch_nq    33792 qdmatch_nq_train"
)
ARMS=(full chunked-mix)
LANES=(A B C)
declare -A LANE_QOS=( [A]=preemptive_high [B]=preemptive [C]=normal )

i=0
for spec in "${TASKS[@]}"; do
  set -- $spec
  TASK=$1; SEQ=$2; DATA=$3
  for ARM in "${ARMS[@]}"; do
    LANE=${LANES[$(( i % 3 ))]}
    QOS=${LANE_QOS[$LANE]}
    ARMTAG=$([ "$ARM" = "full" ] && echo full || echo cmix)
    RUN="ctcms-${TASK}-${ARMTAG}-${TAG}"
    OUT=$(ssh lambda "cd $LROOT && sbatch --parsable \
        --qos=$QOS --time=$TIME_LIMIT \
        --job-name=ctcms-lane$LANE --dependency=singleton \
        --export=ALL,TASK=$TASK,DATA_SRC=$DATA,VARIANT=$ARM,SCALE=$SCALE,EPOCHS=1,\
SEQ_LEN=$SEQ,RUN=$RUN,ACT_CKPT=full,SHARD_DEGREE=8,NGPU=8,\
BASE_SRC=$LROOT/bases/$BASE,SAVE_ROOT=/tmp/prasann/ctcms/ckpts,WORK_ROOT=/tmp/prasann/ctcms,\
WANDB_GROUP=ctc-modelscale-$TASK \
        $LAUNCHER" 2>&1 | tail -1)
    echo -e "$(date -Is)\t$RUN\t$TASK\t$ARM\t$SCALE\t$SEQ\tlane$LANE\t$QOS\t$OUT" | tee -a "$LEDGER"
    i=$(( i + 1 ))
  done
done
echo "--- submitted wave $SCALE; queue: ---"
ssh lambda 'squeue -u prasann -o "%.8i %.14j %.9q %.8T %.10M %R"'
