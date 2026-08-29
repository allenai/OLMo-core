#!/bin/bash
# Resubmit every cev/evh eval killed by the --master_port=0 bug. Fixed port scheme inside wrap.
set -uo pipefail
submit () { # RUN RUNGS E5ROOT OUTTAG
  local RUN=$1 RUNGS=$2 E5=$3 TAG=$4
  sbatch --job-name="rev-$RUN" -w mooney --qos=preemptive_high --account=site \
    --gres=gpu:H200:2 --cpus-per-task=16 --mem=200G --time=05:00:00 \
    --output=/data/prasann/joblogs/rev_${RUN}_%j.log \
    --partition=jsteinhardt --wrap='
set -e
export HOME=/data/prasann/home TMPDIR=/data/prasann/tmp AWS_PROFILE=S3
export TRITON_CACHE_DIR=/data/prasann/triton_cache
RUN='"$RUN"'
DST=/data/prasann/ctc_suite/ckpts/$RUN
mkdir -p "$DST"
aws s3 sync "s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/ckpt_relay/$RUN" "$DST" --only-show-errors || [ $? -eq 2 ]
ls "$DST/config.json" "$DST/model_and_optim/.metadata" || { echo "ckpt incomplete"; exit 1; }
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd /scratch/users/prasann/corpus-reasoning
export PYTHONPATH=$REPO/src:$REPO/src/scripts:/scratch/users/prasann/corpus-reasoning
export EVAL500_ROOT='"$E5"'
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OLMO_LANDMARK_SPARSE_DECODE=1 $PY -m torch.distributed.run --nproc_per_node=2 --master_port=$((20000 + SLURM_JOB_ID % 10000)) $REPO/src/scripts/ctc_eval/eval/eval_lc_native.py \
  --model-path "$DST" --out "$DST/eval_'"$TAG"'_multirung.json" --tokenizer Qwen/Qwen3.5-0.8B \
  --max-length 40960 --max-test-samples 600 --batch-size 1 --skip-ruler --skip-gen \
  --landmark-mem-id 248200 --landmark-pad-id 248203 --eos-token-id 248044 \
  --prompt-format chat --query-position after \
  --ladder --ladder-tasks '"$TAG"' --ladder-rungs '"$RUNGS"'
cp "$DST/eval_'"$TAG"'_multirung.json" /scratch/users/prasann/stl_eval_results/${RUN}-chat_'"$TAG"'_multirung.json
rm -rf "$DST"
echo EVAL-OK'
  sleep 1
}
E5O=/data/prasann/outlier_lengthmix/eval500_local/outlier
E5Q=/data/prasann/qdmatch_lengthmix/eval_rungs
# sparse lropt ladder (outlier)
for R in lmx-slm-p8k250-lropt-4b lmx-slm-p8k1000-lropt-4b lmx-slm-p8k2000-lropt-4b \
         lmx-slm-p8k4000-lropt-4b lmx-slm-p8k8000-lropt-4b lmx-slm-p8k16000-lropt-4b; do
  submit "$R" 3k,8k "$E5O" outlier
done
# check#2 sparse (wave-2 names)
for R in lmx-slm-m8kmix-4b lmx-slm-p8k5000-4b lmx-slm-p8k8000-4b lmx-slm-p8k4000-4b; do
  submit "$R" 3k,8k "$E5O" outlier
done
# 32k sparse
submit lmx-slm-p32k2000-4b 3k,8k,16k,32k "$E5O" outlier
# qdmatch smoke (all four)
for R in lmx-full-q2k5000-qd-4b lmx-slm-q2k5000-qd-4b lmx-full-q8k4000-qd-4b lmx-slm-q8k4000-qd-4b; do
  submit "$R" 3k,8k "$E5Q" qdmatch_nq
done
echo RESUBMIT-DONE
