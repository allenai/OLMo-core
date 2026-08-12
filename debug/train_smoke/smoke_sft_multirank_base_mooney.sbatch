#!/bin/bash
# Task #25, mooney variant. Identical experiment to smoke_sft_multirank_base.sbatch; it differs only
# in where it runs and in the staging step that move requires.
#
#   world_size 2  -> FSDP actually wraps, and `reduce_metrics` actually reduces across ranks.
#                    Both were no-ops at world_size 1.
#   --base        -> a real, marker-repaired Qwen3-0.6B checkpoint instead of --from-scratch, which
#                    exercises the base-loading path and the marker embedding rows.
#
# WHY NOT horton. The prerequisites were built there, but all eight of its cards are held by
# `preemptive` and `preemptive_high_*` jobs, and on `berkeleynlp` we hold only `preemptive`, which
# preempts `normal` and nothing else. That request does not queue for minutes, it starves: single-GPU
# jobs backfill into every card that frees, and a 2-GPU request never sees two at once. `jsteinhardt`
# is our own partition, where `preemptive_high` preempts `normal,preemptive`. mooney is a jsteinhardt
# H200 node that also carries the node-local conda env -- only horton, cubbins and mooney do.
#
# Model/tokenizer are pinned to the base: /data/prasann/dcmix_base is a Qwen3-0.6B at Qwen3's
# 151936-row vocabulary (its embedding distcp shards are 2 x 311 MB = 151936 x 1024 x fp32), so
# `--model qwen3_0_6B --tokenizer qwen3`, and the shards must be Qwen3-tokenized -- the ones from
# the 2026-08-11 smoke are qwen3_5 (vocab 248320) and would not load into this embedding matrix.
#
# Everything node-local per local_cluster.md: interpreter, shards, caches, logs, checkpoints.
#SBATCH --job-name=ctc-smoke-mr
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --output=/data/prasann/joblogs/ctc-smoke-mr-%j.log

set -euo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_smoke
SRC=/net/horton/data/prasann/ctc_smoke
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

# Node-local caches. HOME on NFS wedges the flashinfer/triton JIT; TMPDIR on / fills the disk.
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export TRITON_CACHE_DIR=$WORK/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" /data/prasann/joblogs /data/prasann/ctc_work

echo "=== node $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# STAGING. The repaired base and the qwen3 shards were built on horton's /data. `/net/horton/...` is
# the NFS layer -- reading a 2 GB checkpoint or a data shard through it during training is the
# `nfs_wait_bit_killabl` deadlock, not merely slow. So copy once, here, and then never touch /net
# again: every path the trainer sees below is on this node's own disk.
echo "=== staging base + shards from horton ($(date +%T)) ==="
# rsync creates only the FINAL component of the destination, so a nested `shards/<task>/` fails with
# `mkdir ... No such file or directory` unless the parent exists first.
mkdir -p "$WORK/base_fixmark" "$WORK/shards/mathmatch_qwen3"
rsync -a "$SRC/base_fixmark/" "$WORK/base_fixmark/"
rsync -a "$SRC/shards/mathmatch_qwen3/" "$WORK/shards/mathmatch_qwen3/"
echo "=== staged ($(date +%T)) ==="
du -sh "$WORK/base_fixmark" "$WORK/shards/mathmatch_qwen3"

cd "$REPO"
export PYTHONPATH=src:ctc/src

# Re-audit the copy the run will ACTUALLY read. The point of the marker audit is that the repair
# survives every round trip between the process that made it and the process that trains on it; a
# copy is one more such trip, so it gets checked rather than assumed. --check-only needs no HF
# tokenizer, so this stays offline.
echo "=== marker audit on the STAGED copy ==="
$PY src/scripts/ctc/fix_marker_embeddings.py --base "$WORK/base_fixmark/model_and_optim" \
    --check-only --model qwen3_0_6B --tokenizer qwen3

SAVE=$WORK/ckpt_sft_mr
# A save folder that already holds a checkpoint is loaded FIRST and silently wins over --base
# ("Ignoring load path ... since checkpoint was found in save folder"), which would make the whole
# point of this run untestable. Start clean.
rm -rf "$SAVE"

echo "=== shards ==="; ls -la "$WORK/shards/mathmatch_qwen3"

# `$PY -m torch.distributed.run`, never bare `torchrun`: the bare name resolves off PATH to
# /usr/local/linux/miniforge-3.13, whose interpreter lives on NFS. That run parked in D state on
# `rpc_wait_bit_killabl` at ~2% CPU and looked exactly like a slow model load.
#
# --base points at base_fixmark/model_and_optim, NOT its parent. model_and_optim/ is what
# save_model_and_optim_state writes; olmo-core loads a checkpoint dir or a dir of stepN/
# checkpoints, and the parent is neither -- pointing --base there fails inside trainer.fit()
# AFTER both ranks have built the model and FSDP has wrapped it.
echo "=== interpreter ==="; $PY -c "import sys, torch; print(sys.executable, torch.__version__)"

echo "=== SFT: 30 steps, world_size 2, chunked mask, REAL marker-repaired base ==="
$PY -m torch.distributed.run --nproc-per-node=2 --master_port=29581 \
    src/scripts/ctc/train/sft.py ctc-smoke-mr \
    --data "$WORK/shards/mathmatch_qwen3" \
    --base "$WORK/base_fixmark/model_and_optim" \
    --arch chunked \
    --model qwen3_0_6B \
    --tokenizer qwen3 \
    --seq-len 2048 \
    --max-steps 30 \
    --gpus-per-node 2 \
    --lr 1e-5 \
    --save-folder "$SAVE" \
    --save-interval 30 \
    --ephemeral-save-interval 10 \
    --wandb-project ""

echo "=== SFT EXIT=$? ==="
# NOT `find ... | head`: head closes the pipe, find dies on SIGPIPE (141), and `set -o pipefail`
# turns that into a FAILED job for a run that exited 0 -- which is exactly the "sbatch reports
# FAILED on a run that succeeded" trap, self-inflicted. Bound the walk instead of the output.
echo "=== checkpoint tree ==="
find "$SAVE" -maxdepth 2
echo "=== fingerprint written into the checkpoint? ==="
find "$SAVE" -name "*fingerprint*" -exec echo {} \; -exec cat {} \;
