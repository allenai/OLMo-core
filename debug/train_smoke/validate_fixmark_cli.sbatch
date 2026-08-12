#!/bin/bash
# Validate the PROMOTED src/scripts/ctc/fix_marker_embeddings.py against two real checkpoints.
#
# The unit test in ctc/tests/train/ pins the tensor logic on a synthetic matrix. What it cannot pin
# is the CLI: that the distcp load works, that the reserved-id lookup finds the right rows in a real
# 151936-row embedding matrix, and -- new in the promoted version -- that --check-only exits
# non-zero on a bad base so it can gate a launcher under `set -e`.
#
# CPU only. horton has both checkpoints: the raw base and the repaired copy.
#SBATCH --job-name=ctc-fixmark-val
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/data/prasann/joblogs/ctc-fixmark-val-%j.log

set -uo pipefail   # NOT -e: the exit codes are the thing under test.

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_smoke
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

export HOME=$WORK/home
export TMPDIR=$WORK/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

cd "$REPO"
export PYTHONPATH=src:ctc/src
echo "=== node $(hostname) ==="

echo "=== A. the RAW base -- expect POISONED and a NON-ZERO exit ==="
$PY src/scripts/ctc/fix_marker_embeddings.py --base /data/prasann/dcmix_base/model_and_optim \
    --check-only --model qwen3_0_6B --tokenizer qwen3
echo "EXIT_RAW=$?   (expected: non-zero)"

echo "=== B. the REPAIRED base -- expect OK and exit 0 ==="
$PY src/scripts/ctc/fix_marker_embeddings.py --base "$WORK/base_fixmark/model_and_optim" \
    --check-only --model qwen3_0_6B --tokenizer qwen3
echo "EXIT_FIXED=$?   (expected: 0)"

echo "=== VALIDATION DONE ==="
