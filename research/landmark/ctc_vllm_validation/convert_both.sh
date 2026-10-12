#!/bin/bash
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
export PYTHONPATH="/scratch/users/prasann/tf_qwen35_target:$REPO/research/common:$REPO/src"
export HF_HOME=/data/prasann/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

echo "=== FULL export START $(date) ==="
# TODO(port): depends on src/corpus_reasoning/train/export_olmo_to_hf.py, not ported
$PY "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
  --save-folder /data/prasann/ctc_suite/ckpts/ctc-1ep-contra-full \
  --hf-out /data/prasann/ctc_suite/hf_exports/ctc-1ep-contra-full \
  --base-model Qwen/Qwen3.5-0.8B-Base
echo "=== FULL export rc=$? END $(date) ==="

echo "=== CHUNKED-MIX export START $(date) ==="
# TODO(port): depends on src/corpus_reasoning/train/export_olmo_to_hf.py, not ported
$PY "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
  --save-folder /data/prasann/ctc_suite/ckpts/ctc-smoke-contra-3ep \
  --hf-out /data/prasann/ctc_suite/hf_exports/ctc-smoke-contra-3ep \
  --base-model Qwen/Qwen3.5-0.8B-Base
echo "=== CHUNKED-MIX export rc=$? END $(date) ==="
