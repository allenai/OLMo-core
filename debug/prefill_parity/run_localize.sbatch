#!/bin/bash
#SBATCH --job-name=prefill-localize
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/prefill_parity/logs/localize_%j.log
#
# Follow-up to run_parity.sbatch: WHY does prefill reuse change the output, and is `--attn chunked`
# reachable at all.
#
# Submitted with --dependency=afterany:<parity job> so only one GPU is ever held.
set -uo pipefail

PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/prefill_parity
CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b
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

# ── 1. localize: synthetic architectures, the GDN-state ablation, per-layer divergence ──────────
echo; echo "=== localize ==="
$PY debug/prefill_parity/localize.py \
    --ckpt "$CKPT" --data "$CONTRA" --spec contradiction --tokenizer "$TOK" \
    --max-length 16384 --rows 4 --query-position both \
    --experiments synthetic,ablate,layers \
    --out "$WORK/out/localize.json"
echo "localize exit=$?"

# ── 2. the chunked mask, now that enable_document_chunk_attention gets its ids ───────────────────
echo; echo "=== probe: contradiction, attn chunked (qwen3_5 ids) ==="
$PY debug/prefill_parity/parity_probe.py \
    --ckpt "$CKPT" --data "$CONTRA" --spec contradiction --tokenizer "$TOK" \
    --max-length 16384 --groups 2 --rows-per-group 6 \
    --query-positions both --attn chunked --family qwen3_5 \
    --out "$WORK/out/probe_contradiction_chunked.json"
echo "probe exit=$?"

# ── 3. re-time the two arms with a warm-up, so the speedup is not the first-forward tax ─────────
echo; echo "=== probe: contradiction, attn full (re-timed) ==="
$PY debug/prefill_parity/parity_probe.py \
    --ckpt "$CKPT" --data "$CONTRA" --spec contradiction --tokenizer "$TOK" \
    --max-length 16384 --groups 3 --rows-per-group 8 \
    --query-positions both --attn full \
    --out "$WORK/out/probe_contradiction_full_warm.json"
echo "probe exit=$?"

echo; echo "=== done $(date '+%F %T') ==="
ls -la "$WORK/out"
