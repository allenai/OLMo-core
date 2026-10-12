#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --job-name=lsp-2gold
#SBATCH --output=/data/prasann/layer_probe/twogold_%j.out
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
U="none,all,top25,top50,top75,bottom50,alt,every4,attn,gdn"
run () { local name=$1; shift; date; PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name exit=$?"; tail -22 $O/$name.log; }
# T/U: ORACLE variant -- the gold docs stay real even in ultra layers, so the ultra level only
# removes the non-gold kept documents. Isolates "is the collapse just the gold doc?"
run T_contra8k_k12_2lvl_goldreal --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "$U" --slot-source layer_input_mean --ultra-keep-gold
run U_contra8k_k3_2lvl_goldreal --task contradiction --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "$U" --slot-source layer_input_mean --ultra-keep-gold
# V: oolong is gold-blind (no pinpoint evidence) -- does IT tolerate ultra layers?
run V_oolong8k_k3_2lvl_layermean --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --ultra-patterns "$U" --slot-source layer_input_mean
run V2_oolong8k_k3_2lvl_embed --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --ultra-patterns "$U" --slot-source embed_mean
date; echo TWOGOLD_DONE
