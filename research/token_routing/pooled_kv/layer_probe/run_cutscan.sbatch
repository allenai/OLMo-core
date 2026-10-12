#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --job-name=lsp-cut
#SBATCH --output=/data/prasann/layer_probe/cut_%j.out
set -x
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/hf
mkdir -p $TRITON_CACHE_DIR $TMPDIR /data/prasann/layer_probe
REPO_ROOT="${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CK_C=/data/prasann/dense_ckpts/tsl-full-contradiction-s56M-4b-20260831T235942-0700/model_and_optim
W=/data/prasann/layer_probe/work
O=/data/prasann/layer_probe
# Fine scan of the read-then-compress cut depth: the A/B sweep found a sharp damage PEAK around
# a cut at layer 8 that is NOT monotone in the number of soft layers.
PC="none,from1,from2,from3,from4,from5,from6,from7,from8,from9,from10,from11,from12,from14,all"
run () { local name=$1; shift; date; PYTHONPATH=src $PY research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name exit=$?"; tail -20 $O/$name.log; }
run G_contra8k_k12_cutscan_embed --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$PC" --slot-source embed_mean
run H_contra8k_k12_cutscan_layermean --task contradiction --rung 8k --rows 200 --keep 0.0833 \
  --ckpt $CK_C --shard $W/contra_iid_8192_r200 --patterns "$PC" --slot-source layer_input_mean
date; echo CUTSCAN_DONE
