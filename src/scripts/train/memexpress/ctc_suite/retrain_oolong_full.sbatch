#!/bin/bash
# Retrain the MISSING ctc-4b-oolong-FULL (dense) arm. No ctc-4b-oolong-full exists on S3 (only
# -cmix), so the grid's oolong row is chunked-only. This trains the dense counterpart on the SAME
# oolong_train shard (cot_mode=none) with the SAME CTC-4B dense recipe as every other -full task
# (VARIANT=full, 4b, epochs=1, gb=8) from the VERIFIED q35-4b HYBRID base (vocab 248320, marker
# audit PASS) -- NOT the wrong non-hybrid q4b-base-modelonly.
# seq_len=40960 (oolong max_example_len=39971); 8 GPUs for throughput (21000 ex ~= 2625 steps).
# NOWANDB=1 (offline env) -> loss curve captured from the node log.
#SBATCH --job-name=ctc-4b-oolong-full-retrain
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/ctc_suite_%x_%j.log
set -uo pipefail
export TASK=oolong
export DATA_SRC=/scratch/users/prasann/ctc_suite_staged/shards/oolong_train
export VARIANT=full
export SCALE=4b
export SEQ_LEN=40960
export EPOCHS=1
export GLOBAL_BATCH=8
export NGPU=4
export ACT_CKPT=full
export SEED=0
export BASE_SRC=/scratch/users/prasann/ctc_suite_lambda_stage/q35-4b-base-modelonly
export RUN=ctc-4b-oolong-full-retrain
export WANDB_GROUP=ctc-suite-oolong
export NOWANDB=1
exec bash /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch
