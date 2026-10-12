#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --job-name=lsp-sweep
#SBATCH --output=/data/prasann/layer_probe/sweep_%j.out
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
PF="none,all,from4,from8,from12,from16,from20,from24,from28"
run () {  # run <name> <args...>
  local name=$1; shift
  date
  PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" \
    --out $O/$name.json > $O/$name.log 2>&1
  echo "$name exit=$?"
  tail -20 $O/$name.log
}

# A: main layer-pattern sweep, contradiction @8k, keep 1/12
run A_contra8k_k12_embed --task contradiction --rung 8k --rows 200 --keep 0.0833 --selftest \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$P" --slot-source embed_mean
# B: read-then-compress cut (slot = mean of the doc's hidden states at the cut layer)
run B_contra8k_k12_layermean --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$PF" --slot-source layer_input_mean
# B2: the same cuts with the ordinary embed-mean slot, so B is comparable
run B2_contra8k_k12_embed_from --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$PF" --slot-source embed_mean
# C: does masking the pooled docs' stale tokens out of later DENSE layers rescue bottom-half soft?
run C_contra8k_k12_stalemask --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$P" --slot-source embed_mean --stale-keys mask
# D: keep dependence, contradiction @8k keep 1/3
run D_contra8k_k3_embed --task contradiction --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$P" --slot-source embed_mean
# E: oolong (gold-blind) @8k keep 1/3
run E_oolong8k_k3_embed --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --patterns "$P" --slot-source embed_mean
# F: length dependence, contradiction @32k keep 1/12
run F_contra32k_k12_embed --task contradiction --rung 32k --rows 64 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_32768_r64 --patterns "$P" --slot-source embed_mean
date; echo SWEEP_DONE
