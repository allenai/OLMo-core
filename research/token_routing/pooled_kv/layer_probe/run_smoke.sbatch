#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --job-name=lsp-smoke
#SBATCH --output=/data/prasann/layer_probe/smoke_%j.out
set -x
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/hf
mkdir -p $TRITON_CACHE_DIR $TMPDIR /data/prasann/layer_probe
REPO_ROOT="${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CK=/data/prasann/dense_ckpts/tsl-full-contradiction-s56M-4b-20260831T235942-0700/model_and_optim
O=/data/prasann/layer_probe
date
PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py \
  --task contradiction --rung 8k --rows 2 --keep 0.0833 --selftest \
  --ckpt $CK --shard /scratch/users/prasann/slot_probe/contra_iid_8192 \
  --patterns none,all,top50,bottom50,gdn,attn \
  --out $O/smoke_contra_8k.json > $O/smoke_contra_8k.log 2>&1
echo "exit=$?"
date; echo SMOKE_DONE
