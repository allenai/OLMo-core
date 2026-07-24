#!/bin/bash
# Launch the 2k-context contradiction SFT sanity runs: Qwen3-4B and Qwen3.5-4B, full attention.
# Both go through the (now family-aware) run_ctc_local.sbatch -> train_ctc_suite.py; family is
# auto-detected from each shard's marker_set. One model per berkeleynlp node (parallel).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LAUNCH=$REPO/src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch
DATA=/scratch/users/prasann/ctc_qwen_compare
Q3_BASE=/scratch/users/prasann/ctc_qwen_compare/bases/qwen3-4b-base-trainedmark
Q35_BASE=/scratch/users/prasann/ctc_suite_lambda_stage/q35-4b-base-modelonly

# --- guard: prerequisites present ---
for p in "$DATA/contra_2k_qwen3_n77_5k/metadata.json" "$DATA/contra_2k_qwen35_n77_5k/metadata.json" \
         "$Q3_BASE/model_and_optim/.metadata" "$Q35_BASE/model_and_optim/.metadata"; do
  [ -f "$p" ] || { echo "MISSING prerequisite: $p"; exit 1; }
done
echo "all prerequisites present."

COMMON="VARIANT=full SCALE=4b SEQ_LEN=5632 EPOCHS=3 GLOBAL_BATCH=8 MICRO_BATCH=1 NGPU=8 \
TASK=contradiction PARTITION=berkeleynlp TIME=05:00:00 WANDB_GROUP=q3-vs-q35-contra-2k-sanity"

echo "=== launch Qwen3.5-4B on horton ==="
env $COMMON NODE=horton \
  DATA_SRC=$DATA/contra_2k_qwen35_n77_5k BASE_SRC=$Q35_BASE \
  RUN=q35-4b-contra-2k-full-sanity \
  bash "$LAUNCH"

echo "=== launch Qwen3-4B on lorax ==="
env $COMMON NODE=lorax \
  DATA_SRC=$DATA/contra_2k_qwen3_n77_5k BASE_SRC=$Q3_BASE \
  RUN=q3-4b-contra-2k-full-sanity \
  bash "$LAUNCH"

echo "=== submitted; queue ==="
squeue -u prasann -o "%.10i %.28j %.11P %.2t %.9M %R" | grep -E "contra-2k|JOBID"
