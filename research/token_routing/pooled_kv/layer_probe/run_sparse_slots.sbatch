#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --job-name=lsp-sparse
#SBATCH --output=/data/prasann/layer_probe/sparse_%j.out
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
# How FEW layers need to carry the pooled documents' slot at all? (stale-keys mask: in every other
# layer the pooled docs and their slots are simply absent.)
S="none,all,alt,every3,every4,every6,every8,every16,list:0,list:0+16,attn,alt1,gdn"
S=$(echo $S | sed 's/list:0+16/list:0,16/')   # keep the comma inside the list spec
S="none,all,alt,every3,every4,every6,every8,every16,attn,alt1,gdn,list:0"
run () { local name=$1; shift; date; PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name exit=$?"; tail -22 $O/$name.log; }
run W_contra8k_k12_sparseslots --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$S" --slot-source embed_mean --stale-keys mask
run W2_contra8k_k12_sparseslots_layermean --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$S" --slot-source layer_input_mean --stale-keys mask
run X_contra32k_k12_sparseslots --task contradiction --rung 32k --rows 64 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_32768_r64 --patterns "$S" --slot-source embed_mean --stale-keys mask
run Y_oolong8k_k3_sparseslots --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --patterns "$S" --slot-source embed_mean --stale-keys mask
date; echo SPARSE_DONE
