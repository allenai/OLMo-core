#!/bin/bash
#SBATCH --job-name=prefill-localize2
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/prefill_parity/logs/localize2_%j.log
#
# Re-run of the localization after two fixes: the synthetic models are cast to bf16 (flash-attn
# takes fp16/bf16 only, and `init_weights` leaves fp32), and each experiment is now independent so a
# failing control cannot take the decisive GDN ablation down with it -- which is exactly what
# happened in job 3438595.
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

$PY debug/prefill_parity/localize.py \
    --ckpt "$CKPT" --data "$CONTRA" --spec contradiction --tokenizer "$TOK" \
    --max-length 16384 --rows 4 --query-position both \
    --experiments synthetic,ablate,layers \
    --out "$WORK/out/localize.json"
echo "localize exit=$?"

echo; echo "=== done $(date '+%F %T') ==="
