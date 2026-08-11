#!/bin/bash
set -uo pipefail
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
BUNDLE=/data/prasann/ctc_local_bundle
CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b

mkdir -p "$BUNDLE/contra"
ln -sf /data/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_n100_k3.jsonl \
       "$BUNDLE/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"

export HOME=/data/prasann/hometmp
export TMPDIR=/data/prasann/hometmp
export HF_HOME=/data/prasann/hf-cache
export PYTHONPATH="$REPO/src:$REPO/ctc/src"
cd "$REPO"

echo "=== dry-run ==="
$PY -m ctc.eval.cli --tasks contradiction --rungs 2k --bundle "$BUNDLE" --dry-run

echo "=== real eval (limit 12) ==="
$PY -m ctc.eval.cli --ckpt "$CKPT" --tasks contradiction --rungs 2k \
    --bundle "$BUNDLE" --out /data/prasann/ctc_local_results \
    --limit 12 --max-length 8192 --attn full --ignore-format-fingerprint \
    --tokenizer Qwen/Qwen3.5-0.8B-Base
echo "=== exit=$? ==="
