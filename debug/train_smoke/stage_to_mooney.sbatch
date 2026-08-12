#!/bin/bash
# CPU-only pre-staging for smoke_sft_multirank_base_mooney.sbatch.
#
# The GPU launcher stages the base + shards itself, but that rsync pulls 2.0 GB across /net (the
# ~5 MB/s NFS layer) -- ~7 minutes of two H200s sitting idle. Doing it here first makes the GPU
# job's rsync a no-op, and lets the marker audit on mooney's OWN copy be read BEFORE the GPU job
# is submitted rather than after it has already started training.
#SBATCH --job-name=ctc-smoke-stage
#SBATCH --partition=jsteinhardt
#SBATCH --qos=normal
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/data/prasann/joblogs/ctc-smoke-stage-%j.log

set -euo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_smoke
SRC=/net/horton/data/prasann/ctc_smoke
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

export HOME=$WORK/home
export TMPDIR=$WORK/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

echo "=== node $(hostname) ==="
echo "=== staging ($(date +%T)) ==="
# rsync creates only the FINAL component of the destination, so a nested `shards/<task>/` fails with
# `mkdir ... No such file or directory` unless the parent exists first.
mkdir -p "$WORK/base_fixmark" "$WORK/shards/mathmatch_qwen3"
rsync -a "$SRC/base_fixmark/" "$WORK/base_fixmark/"
rsync -a "$SRC/shards/mathmatch_qwen3/" "$WORK/shards/mathmatch_qwen3/"
echo "=== staged ($(date +%T)) ==="
du -sh "$WORK/base_fixmark" "$WORK/shards/mathmatch_qwen3"
ls -la "$WORK/shards/mathmatch_qwen3"

cd "$REPO"
export PYTHONPATH=src:ctc/src
echo "=== marker audit on mooney's OWN copy ==="
$PY src/scripts/ctc/fix_marker_embeddings.py --base "$WORK/base_fixmark/model_and_optim" \
    --check-only --model qwen3_0_6B --tokenizer qwen3
echo "=== STAGE DONE ==="
