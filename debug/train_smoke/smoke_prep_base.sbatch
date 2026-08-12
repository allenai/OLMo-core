#!/bin/bash
# Prerequisites for the multi-rank real-base smoke run (task #25). CPU only -- no GPU is held while
# this runs, which matters because horton is fully allocated and the training job needs 2 cards.
#
# Three things happen here, in order:
#   1. AUDIT the candidate base's marker embedding rows and print the cosine and the norm ratio.
#   2. REPAIR it into a copy (never in place -- the original stays a reference).
#   3. REBUILD the mathmatch shards with the **qwen3** marker set and the **Qwen3-0.6B** tokenizer.
#      The existing debug/train_smoke shards are qwen3_5-tokenized (vocab 248320); the base is a
#      Qwen3-0.6B (vocab 151936, verified from the distcp shard sizes), so those token ids mean
#      nothing to it and "did the base load" could not be read off the loss.
#SBATCH --job-name=ctc-smoke-prep
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/prasann/joblogs/ctc-smoke-prep-%j.log

set -euo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_smoke
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export TRITON_CACHE_DIR=$WORK/triton
export HF_HOME=/data/prasann/hf-cache
export HF_HUB_OFFLINE=1
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$WORK/shards" /data/prasann/joblogs

echo "=== node $(hostname) ==="
cd "$REPO"
export PYTHONPATH=src:ctc/src
echo "=== interpreter ==="; $PY -c "import sys, torch; print(sys.executable, torch.__version__)"

BASE=/data/prasann/dcmix_base/model_and_optim
FIXED=$WORK/base_fixmark
# The absolute snapshot directory, not the "Qwen/..." repo id. With HF_HUB_OFFLINE=1 the repo-id
# form raised LocalEntryNotFoundError against this cache even though every file is present, so the
# lookup is taken out of the picture entirely -- nothing here needs the hub.
TOK=/data/prasann/hf-cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
echo "=== HF_HOME=$HF_HOME  TOK=$TOK ==="; ls "$TOK" | tr '\n' ' '; echo

echo "=== 1. AUDIT $BASE ==="
# `|| true`: --check-only exits non-zero on a bad base *by design*, so it gates a launcher. Here
# a bad base is the expected finding -- that is what step 2 repairs -- so don't let it abort.
$PY src/scripts/ctc/fix_marker_embeddings.py --base "$BASE" --check-only \
    --model qwen3_0_6B --tokenizer qwen3 || true

echo "=== 2. REPAIR -> $FIXED ==="
$PY src/scripts/ctc/fix_marker_embeddings.py --base "$BASE" --out "$FIXED" \
    --model qwen3_0_6B --tokenizer qwen3 --hf-tokenizer "$TOK"

echo "=== 2b. RE-AUDIT the repaired copy (proves it round-trips through distcp) ==="
$PY src/scripts/ctc/fix_marker_embeddings.py --base "$FIXED/model_and_optim" --check-only \
    --model qwen3_0_6B --tokenizer qwen3

echo "=== 3. SHARDS: mathmatch, qwen3 marker set, Qwen3-0.6B tokenizer ==="
$PY src/scripts/ctc/convert_to_shards.py --task mathmatch --emit dense --chunk-by document \
    --marker-set qwen3 --tokenizer "$TOK" --query-position after --seq-len 4096 \
    --input-jsonl "$REPO/debug/train_smoke/data/mathmatch/train.jsonl" \
    --out-dir "$WORK/shards/mathmatch_qwen3"

echo "=== shard dir ==="; ls -la "$WORK/shards/mathmatch_qwen3/"
echo "=== metadata ==="; cat "$WORK/shards/mathmatch_qwen3/metadata.json"
echo "=== PREP DONE ==="
