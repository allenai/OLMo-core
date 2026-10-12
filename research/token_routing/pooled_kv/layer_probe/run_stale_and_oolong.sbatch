#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --job-name=lsp-stale
#SBATCH --output=/data/prasann/layer_probe/stale_%j.out
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
P="none,all,top25,top50,top75,bottom25,bottom50,bottom75,alt,alt1,every3,every4,gdn,attn,gdn_top50,attn_top50"
PC="none,from1,from2,from3,from4,from5,from6,from7,from8,from10,from12,from16,from20,from24,all"
run () { local name=$1; shift; date; PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name exit=$?"; tail -20 $O/$name.log; }

# I: re-run of C with the REALIZABLE FLOP accounting for --stale-keys mask (pooled tokens are
#    dropped from every layer at or after the first soft layer, since nothing reads them again).
run I_contra8k_k12_stalemask --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$P" --slot-source embed_mean --stale-keys mask
# J: stale mask + slot rebuilt at each soft run (both fixes together)
run J_contra8k_k12_stalemask_layermean --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$P" --slot-source layer_input_mean --stale-keys mask
# K: does the read-then-compress cut generalise to oolong (gold-blind, line-chunked)?
run K_oolong8k_k3_cutscan_layermean --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --patterns "$PC" --slot-source layer_input_mean
run L_oolong8k_k3_cutscan_embed --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --patterns "$PC" --slot-source embed_mean
# M: does the cut generalise to 32k on contradiction?
run M_contra32k_k12_cutscan_layermean --task contradiction --rung 32k --rows 64 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_32768_r64 --patterns "$PC" --slot-source layer_input_mean
date; echo STALE_DONE
