#!/bin/bash
# First real run through the migrated `sft.py` (task #21).
#
# Deliberately uses `mathmatch`: a pure-synthetic task with no corpus and no network, whose train
# and eval halves come from the SAME generator with the SAME knobs -- so the train/eval in-domain
# question cannot confound the result. This tests MECHANICS (loader, chunked mask, fingerprint
# callback, optimizer, checkpoint write), not model quality.
#
# Everything node-local per local_cluster.md: interpreter, shards, caches, logs, checkpoints.
#SBATCH --job-name=ctc-smoke-sft
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/data/prasann/joblogs/ctc-smoke-sft-%j.log

set -euo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_smoke
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

# Node-local caches. HOME on NFS wedges the flashinfer/triton JIT; TMPDIR on / fills the disk.
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export TRITON_CACHE_DIR=$WORK/triton
export HF_HOME=/data/prasann/hf-cache
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$WORK/shards" /data/prasann/joblogs

echo "=== node $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Shards were built on the login node (synthetic task, CPU only); stage them node-local.
rsync -a "$REPO/debug/train_smoke/shards/mathmatch/" "$WORK/shards/mathmatch/"
echo "=== shard dir ==="; ls -la "$WORK/shards/mathmatch/"

cd "$REPO"
export PYTHONPATH=src:ctc/src

# `$PY -m torch.distributed.run`, never bare `torchrun`: the bare name resolves off PATH to
# /usr/local/linux/miniforge-3.13, whose interpreter lives on NFS. That run parked in D state on
# `rpc_wait_bit_killabl` at ~2% CPU and looked exactly like a slow model load. The interpreter has
# to be the node-local one, and this is the only spelling that guarantees it.
echo "=== interpreter ==="; $PY -c "import sys, torch; print(sys.executable, torch.__version__)"

echo "=== SFT: 20 steps, qwen3_0_6B from scratch, chunked mask ==="
$PY -m torch.distributed.run --nproc-per-node=1 --master_port=29571 src/scripts/ctc/train/sft.py ctc-smoke-sft \
    --data "$WORK/shards/mathmatch" \
    --from-scratch \
    --arch chunked \
    --model qwen3_0_6B \
    --tokenizer qwen3_5 \
    --seq-len 2048 \
    --max-steps 20 \
    --gpus-per-node 1 \
    --lr 1e-5 \
    --save-folder "$WORK/ckpt_sft" \
    --save-interval 20 \
    --ephemeral-save-interval 10 \
    --wandb-project ""

echo "=== SFT EXIT=$? ==="
echo "=== checkpoint tree ==="
find "$WORK/ckpt_sft" -maxdepth 3 | head -40
echo "=== fingerprint written into the checkpoint? ==="
find "$WORK/ckpt_sft" -name "*fingerprint*" -exec echo {} \; -exec cat {} \;
