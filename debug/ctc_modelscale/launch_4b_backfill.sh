#!/bin/bash
# Model-scale study: backfill the three 4B reference runs that the July fan-out never produced,
# so every task in the study has a 4B point on the scale plot.
#
#   qdmatch_nq full + chunked-mix -- launched in July but never landed in
#                                    results/ctc_suite/dense_vs_chunked_table.md
#   hotpotqa    full              -- the -full checkpoint was corrupt on S3 (missing distcp shard),
#                                    which is why the table's hotpotqa row is CHUNKED-ONLY
#
# Runs on jsteinhardt mooney (H200 141GB, has both shards AND the audited q35-4b base node-local
# under /data/prasann/ctc_suite). The jsteinhardt per-user cap is 8 GPUs = one 8-GPU node, so the
# three runs are serialized with --dependency=singleton rather than submitted in parallel.
#
# seq_len and epochs are copied from the original 4B fan-out so these are drop-in comparable with
# the rest of the 4B table: qdmatch_nq 33792, hotpotqa 26112, 1 epoch, 20k examples.
set -o pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LAUNCHER=$REPO/src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch
BASE=/data/prasann/ctc_suite/bases/q35-4b-base-modelonly
LEDGER="$(dirname "$0")/LAUNCH_LEDGER.tsv"

# run_name : task : seq_len : data_src : variant
#
# ⚠ qdmatch_nq points at `qdmatch_nq_train_20k` on /scratch, NOT mooney's node-local
# `/data/.../qdmatch_nq_train` -- that one is a leftover 2500-example PILOT STUB
# (max_example_len 2664). The launcher stages with `cp -n`, which would NOT overwrite the stub,
# so a same-named copy would have silently trained on 2.5k examples at the wrong length. The
# `_20k` dir is a hardlink farm of the real 20k shard (same filesystem, no extra space) whose
# distinct basename forces a clean node-local staging dir.
RUNS=(
  "ctcms-qdmatch_nq-full-4b:qdmatch_nq:33792:/scratch/users/prasann/ctc_suite_staged/shards/qdmatch_nq_train_20k:full"
  "ctcms-qdmatch_nq-cmix-4b:qdmatch_nq:33792:/scratch/users/prasann/ctc_suite_staged/shards/qdmatch_nq_train_20k:chunked-mix"
  "ctcms-hotpotqa-full-4b:hotpotqa:26112:/data/prasann/ctc_suite/data/hotpotqa_train:full"
)

for spec in "${RUNS[@]}"; do
  IFS=: read -r RUN TASK SEQ DSRC ARM <<< "$spec"
  OUT=$(PARTITION=jsteinhardt QOS=preemptive_high ACCOUNT=site NODE=mooney TIME=12:00:00 \
    TASK=$TASK DATA_SRC=$DSRC VARIANT=$ARM SCALE=4b EPOCHS=1 SEQ_LEN=$SEQ \
    RUN=$RUN BASE_SRC=$BASE NGPU=8 ACT_CKPT=full SHARD_DEGREE=8 \
    WANDB_GROUP=ctc-modelscale-$TASK \
    sbatch --parsable --partition=jsteinhardt --qos=preemptive_high --account=site \
           -w mooney --time=12:00:00 --job-name=ctcms-4b --dependency=singleton \
           --export=ALL "$LAUNCHER" 2>&1 | tail -1)
  echo -e "$(date -Is)\t$RUN\t$TASK\t$ARM\t4b\t$SEQ\tmooney\tjsteinhardt\t$OUT" | tee -a "$LEDGER"
done
echo "--- berkeley queue ---"
squeue -u prasann -o "%.8i %.16j %.12P %.8T %.10M %R"
