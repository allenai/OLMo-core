#!/bin/bash
# Oracle slot-separability readout on the local (Berkeley) replicate checkpoint + corpus.
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --job-name=slotsep
#SBATCH --output=/data/prasann/outlier_probe/slotsep_%j.out
set -x
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp
mkdir -p $TRITON_CACHE_DIR $TMPDIR /data/prasann/outlier_probe
cd /scratch/users/prasann/outlier_probe/wt
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PATH=/data/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
CK=/data/prasann/dense_ckpts/lmx-full-mixs160M-4b-20260828T170816-0700/model_and_optim
date
PYTHONPATH=src $PY debug/pooled_kv/outlier_probe/slot_separability.py \
  --rows "${ROWS:-200}" --ckpt $CK --ckpt-name lmx-full-mixs160M-4b \
  --tokenizer /data/prasann/hf_tok --shard "${SHARD:-/data/prasann/outlier_probe/work2/outlier_2k}" \
  --out "${OUT:-/data/prasann/outlier_probe/slotsep_2k.json}"
date; echo SEP_DONE
