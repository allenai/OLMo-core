#!/bin/bash
#SBATCH --job-name=prefill-chunked
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/prefill_parity/logs/chunked_%j.log
#
# Prefill reuse under the document-chunked mask, on a checkpoint that actually HAS one.
#
# The first chunked probe used ctc-s5-contra-full-4b, whose attention is the stock `default` mixer:
# `enable_document_chunk_attention` reconstructs chunk_ids and hands them down, but a mixer that does
# not accept the keyword ignores them, and the run came out bit-identical to plain causal. So it
# tested nothing. ctc-smoke-contra-3ep is `document_chunked` / `cross_doc_mode=chunked` with
# `document_chunk_attention` set in its config, i.e. the mask is real there.
#
# What this is looking for is a SECOND defect, independent of the GDN one: the reuse path feeds the
# suffix forward only the suffix's token ids, so chunk_ids are reconstructed from a stream whose
# opening <|box_start|> markers are all in the discarded prefix.
set -uo pipefail

PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/prefill_parity
CKPT=/data/prasann/ctc_suite/ckpts/ctc-smoke-contra-3ep
TOK=Qwen/Qwen3.5-0.8B-Base
CONTRA=$WORK/data/contradiction_rung_2048_tail50.jsonl

export HOME=/data/prasann/hometmp
export TMPDIR=/data/prasann/hometmp
export TRITON_CACHE_DIR=/data/prasann/triton_cache
export HF_HOME=/data/prasann/hf-cache
export PYTHONPATH="$REPO/src:$REPO/ctc/src"

cd "$REPO"
echo "=== host $(hostname)  $(date '+%F %T') ==="
echo "interpreter: $PY"

$PY debug/prefill_parity/parity_probe.py \
    --ckpt "$CKPT" --data "$CONTRA" --spec contradiction --tokenizer "$TOK" \
    --max-length 16384 --groups 2 --rows-per-group 6 \
    --query-positions both --attn full,chunked --family qwen3_5 \
    --out "$WORK/out/probe_chunked_ckpt.json"
echo "probe exit=$?"

echo; echo "=== done $(date '+%F %T') ==="
