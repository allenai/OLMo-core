#!/bin/bash
# Launch the multirung native eval for one finished length-mix run.
#   bash eval_run.sh <run_name> <dense|landmark>
# Grades on the v3 bundle outlier ladder (3k/8k/16k/32k = n22/55/110/220; our 8k/16k/32k pools
# n57/111/220 match it; bottom rung is a mild n14-vs-n22 mismatch, identical across runs).
set -uo pipefail
RUN=$1; VARIANT=$2
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"; export PYTHONPATH="$REPO/src"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
LOGD=debug/outlier_lengthmix_scaling/launches; mkdir -p "$LOGD"
timeout 240 $PY -u src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py \
  "$RUN" ai2/jupiter-cirrascale-2 --task outlier --variant "$VARIANT" \
  --tokenizer Qwen/Qwen3.5-0.8B --ladder-version v3 --query-position after \
  --ngpu 8 > "$LOGD/eval-${RUN}.log" 2>&1
tail -3 "$LOGD/eval-${RUN}.log"
