#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --job-name=lsp-drop
#SBATCH --output=/data/prasann/layer_probe/drop_%j.out
set -x
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/hf
mkdir -p $TRITON_CACHE_DIR $TMPDIR /data/prasann/layer_probe
REPO_ROOT="${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CK_C=/data/prasann/dense_ckpts/tsl-full-contradiction-s56M-4b-20260831T235942-0700/model_and_optim
CK_O=/data/prasann/dense_ckpts/tsl-full-oolong-s80M-4b-20260901T085004-0700/model_and_optim
W=/data/prasann/layer_probe/work
O=/data/prasann/layer_probe
S="none,droppooled,list:0,every8,every4,all"
run () { local name=$1; shift; date; PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name exit=$?"; tail -18 $O/$name.log; }
run Z_contra8k_k12_droppooled --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$S" --slot-source embed_mean --stale-keys mask
run Z2_contra32k_k12_droppooled --task contradiction --rung 32k --rows 64 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_32768_r64 --patterns "$S" --slot-source embed_mean --stale-keys mask
run Z3_oolong8k_k3_droppooled --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --patterns "$S" --slot-source embed_mean --stale-keys mask
date; echo DROP_DONE
