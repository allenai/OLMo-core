#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=lsp-hdr
#SBATCH --output=/data/prasann/layer_probe/hdr_%j.out
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
# decisive conditions + the fromL scan that tests whether the mid-stack cut catastrophe was an
# id-readout artefact of the header-less slot
P="none,droppooled,list:0,every8,every4,top25,all,from4,from6,from8,from10,from12,from16"
UU="none,top25,top50,all"
run () { local name=$1; shift; date; PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name exit=$?"; tail -24 $O/$name.log; }

# smoke: header-real must not break the two reductions
run HS_smoke --task contradiction --rung 8k --rows 2 --keep 0.0833 --selftest --prefix-real stop1 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "none,all,top25" --stale-keys mask
# HA/HB: contradiction @8k, header-real stop1, keep 1/12 and 1/36
run HA_contra8k_k12_hdr --task contradiction --rung 8k --rows 200 --keep 0.0833 --prefix-real stop1 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$P" --stale-keys mask
run HB_contra8k_k36_hdr --task contradiction --rung 8k --rows 200 --keep 0.0278 --prefix-real stop1 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$P" --stale-keys mask
# HC: contradiction @32k, header-real
run HC_contra32k_k12_hdr --task contradiction --rung 32k --rows 64 --keep 0.0833 --prefix-real stop1 \
  --ckpt $CK_C --shard $W/contra_iid_32768_r64 --patterns "$P" --stale-keys mask
# HD: oolong @8k, header-real stop3
run HD_oolong8k_k3_hdr --task oolong --rung 8k --rows 200 --keep 0.3333 --prefix-real stop3 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --patterns "$P" --stale-keys mask
# HE/HF: two-level with headers real in the ultra row too
run HE_contra8k_k12_2lvl_hdr --task contradiction --rung 8k --rows 200 --keep 0.0833 --prefix-real stop1 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "$UU" --slot-source layer_input_mean
run HF_oolong8k_k3_2lvl_hdr --task oolong --rung 8k --rows 200 --keep 0.3333 --prefix-real stop3 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --ultra-patterns "$UU" --slot-source layer_input_mean
date; echo HDR_DONE
