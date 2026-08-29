#!/bin/bash
# 2-GPU repro driver for the CP+AC RoPE-dtype CheckpointError.
#   sbatch --partition=jsteinhardt --qos=preemptive_high --account=site -w mooney \
#     --export=ALL,SEQ_LEN=4096,CP=2,AC=full,COMPILE=1 debug/cp_ac_rope_dtype/run_repro.sbatch
#SBATCH --job-name=cprepro
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=/data/prasann/joblogs/cprepro_%x_%j.log
set -x
mkdir -p /data/prasann/joblogs
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH=$REPO/src
export PYTHONWARNINGS=ignore
export TMPDIR=/data/prasann/tmp
export TRITON_CACHE_DIR=/data/prasann/tmp/triton
mkdir -p $TMPDIR $TRITON_CACHE_DIR
export OMP_NUM_THREADS=8
export NGPU=${NGPU:-2}
cd $REPO
$PY -m torch.distributed.run --nproc_per_node=$NGPU \
  --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
  debug/cp_ac_rope_dtype/repro_cp_ac_rope.py
