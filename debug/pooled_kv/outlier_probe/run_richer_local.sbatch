#!/bin/bash
# Richer-slot eval-side probe on the local (Berkeley) replicate checkpoint + corpus.
# Env knobs: SHARD, ROWS, GENROWS, SLOTS, OUT.
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --job-name=richer
#SBATCH --output=/data/prasann/outlier_probe/richer_%j.out
set -x
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp
mkdir -p $TRITON_CACHE_DIR $TMPDIR /data/prasann/outlier_probe
cd /scratch/users/prasann/outlier_probe/wt
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PATH=/data/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
CK=/data/prasann/dense_ckpts/lmx-full-mixs160M-4b-20260828T170816-0700/model_and_optim
date
PYTHONPATH=src $PY debug/pooled_kv/outlier_probe/outlier_richer_slot_probe.py \
  --rows "${ROWS:-200}" --gen-rows "${GENROWS:-48}" --gen-max-new 64 \
  --slots "${SLOTS:-mean,cmean100,cent_cmean,enc4}" \
  --ckpt $CK --ckpt-name lmx-full-mixs160M-4b --tokenizer /data/prasann/hf_tok \
  --shard "${SHARD:-/data/prasann/outlier_probe/work_n55/outlier_n55}" \
  --out "${OUT:-/data/prasann/outlier_probe/richer_local_n55.json}" --weka-out none
date; echo LOCAL_DONE
