#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=lsp-2lvl
#SBATCH --output=/data/prasann/layer_probe/twolevel_%j.out
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
PC="none,from1,from2,from3,from4,from5,from6,from7,from8,from10,from12,from16,from20,from24,all"
run () { local name=$1; shift; date; PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name exit=$?"; tail -22 $O/$name.log; }

# smoke first: the two-level reductions must hold before anything else runs
run S2_smoke --task contradiction --rung 8k --rows 2 --keep 0.0833 --selftest \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "none,all,top50" --slot-source layer_input_mean
# N/O: two-level ("soft layers") on contradiction @8k, base keep 1/12, both kept-doc slot sources
run N_contra8k_k12_2lvl_layermean --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "$U" --slot-source layer_input_mean
run O_contra8k_k12_2lvl_embed --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "$U" --slot-source embed_mean
# P: fatter base keep set (1/3) + ultra layers, for the matched-FLOP comparison against keep 1/12
run P_contra8k_k3_2lvl_layermean --task contradiction --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "$U" --slot-source layer_input_mean
# Q: stale-mask variant of the two-level mode
run Q_contra8k_k12_2lvl_stalemask --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --ultra-patterns "$U" --slot-source layer_input_mean --stale-keys mask
# R: two-level at 32k
run R_contra32k_k12_2lvl_layermean --task contradiction --rung 32k --rows 64 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_32768_r64 --ultra-patterns "$U" --slot-source layer_input_mean
# re-runs of the jobs the mid-flight edit killed
run I_contra8k_k12_stalemask --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 \
  --patterns "none,all,top25,top50,top75,bottom25,bottom50,bottom75,alt,alt1,every3,every4,gdn,attn,gdn_top50,attn_top50" \
  --slot-source embed_mean --stale-keys mask
run K_oolong8k_k3_cutscan_layermean --task oolong --rung 8k --rows 200 --keep 0.3333 \
  --ckpt $CK_O --shard $W/oolong_8192_r200 --patterns "$PC" --slot-source layer_input_mean
run M_contra32k_k12_cutscan_layermean --task contradiction --rung 32k --rows 64 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_32768_r64 --patterns "$PC" --slot-source layer_input_mean
date; echo TWOLEVEL_DONE
