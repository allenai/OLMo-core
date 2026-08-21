#!/bin/bash
# Any-node vLLM 5-rung ladder for a FULL-ATTENTION olmo3-family checkpoint: stage distcp ckpt ->
# HF export on node-local /data -> per-rung prefills/vllm/grade. Unblocks vLLM evals from horton
# (whose berkeleynlp-partition quota pends for hours) by running on any jsteinhardt node that has
# /data/prasann/ctc_vllm_venv (sneetches confirmed 2026-08-18).
#
#   EVAL_TASK=<catalog key> LADDER=<rung dir> CKPT=<distcp dir> SCALE=<label> \
#   [CKPT_HOST=<node>] [RUNGS="2048 ..."] \
#   sbatch [-w node] [--qos=...] debug/ctc_olmo_hybrid/vllm_pair_eval_anynode.sbatch
#
# ⚠ full-attention arms ONLY: the HF export maps noswa-olmo3 onto the Olmo2 HF class; hybrid (GDN)
# has no HF/vLLM mapping and must keep using the native evaluator.
#SBATCH --job-name=vllm-pair-eval
#SBATCH --partition=jsteinhardt
#SBATCH --account=site
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/prasann/ctc_suite_logs/vllm_pair_eval_%j.log

set -uo pipefail
: "${EVAL_TASK:?}" "${LADDER:?}" "${CKPT:?}" "${SCALE:?}"
RUNGS="${RUNGS:-2048 4096 8192 16384 32768}"

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CPUENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -x "$CPUENV/bin/python" ] || CPUENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
VLLMPY=/data/prasann/ctc_vllm_venv/bin/python
[ -x "$VLLMPY" ] || { echo "FATAL: no vllm venv on $(hostname) /data"; exit 2; }
export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export HOME=/data/prasann/home_cache TMPDIR=/data/prasann/tmp FLASHINFER_CACHE_DIR=/data/prasann/flashinfer
mkdir -p "$TMPDIR" "$FLASHINFER_CACHE_DIR"

# ── stage the distcp ckpt to node-local /data (same logic as local_pair_eval.sbatch) ──
LOCAL_CKPT="$CKPT"
if [ ! -f "$CKPT/model_and_optim/.metadata" ] && [ -n "${CKPT_HOST:-}" ]; then
  LOCAL_CKPT="/data/prasann/ctc_pair_eval_ckpts/$(basename "$CKPT")"
  mkdir -p "$(dirname "$LOCAL_CKPT")"
  if [ ! -f "$LOCAL_CKPT/model_and_optim/.metadata" ]; then
    echo "staging $CKPT from $CKPT_HOST"
    if ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no "$CKPT_HOST" true 2>/dev/null; then
      ssh "$CKPT_HOST" "tar -C $(dirname "$CKPT") -cf - $(basename "$CKPT")" | tar -xf - -C "$(dirname "$LOCAL_CKPT")"
    else
      cp -r "/net/$CKPT_HOST$CKPT" "$(dirname "$LOCAL_CKPT")/"
    fi
  fi
fi
[ -f "$LOCAL_CKPT/model_and_optim/.metadata" ] || { echo "FATAL: no usable ckpt"; exit 3; }

# ── HF export on node-local /data (skipped when already present) ──
HF_MODEL=/data/prasann/hf_exports/$(basename "$LOCAL_CKPT")-hf
if [ ! -f "$HF_MODEL/config.json" ]; then
  mkdir -p /data/prasann/hf_exports
  "$CPUENV/bin/python" -u "$REPO/debug/ctc_olmo_hybrid/export_noswa_to_hf.py" \
    "$LOCAL_CKPT" "$HF_MODEL" || { echo "FATAL: HF export failed"; exit 4; }
fi

TOK=/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk
RUNGDIR=/scratch/users/prasann/ctc_suite_staged/eval_rungs/$LADDER
WORK=/data/prasann/vllm_ladder/${LADDER}__${SCALE}
DEST=/scratch/users/prasann/ctc_olmo3ns_results/${LADDER}__${SCALE}_full
mkdir -p "$WORK" "$DEST"

for R in $RUNGS; do
  EV="$RUNGDIR/rung_${R}.jsonl"
  [ -f "$EV" ] || { echo "skip rung $R (no file)"; continue; }
  echo "=== rung $R $(date '+%F %T') ==="
  "$CPUENV/bin/python" -u "$REPO/debug/ctc_vllm_validation/build_prefills_generic.py" \
    --task "$EVAL_TASK" --eval-data "$EV" --tokenizer "$TOK" \
    --doc-start-id 100266 --doc-end-id 100267 --eos-token-id 100257 \
    --out "$WORK/prefills_${R}.jsonl" || { echo "prefills rc=$? rung $R"; continue; }
  "$VLLMPY" -u "$REPO/debug/ctc_vllm_validation/run_vllm_eval_generic.py" \
    --hf-model "$HF_MODEL" --prefills "$WORK/prefills_${R}.jsonl" \
    --mode full --max-new-tokens 200 --max-model-len 40960 \
    --out "$WORK/responses_${R}.jsonl" || { echo "vllm rc=$? rung $R"; continue; }
  "$CPUENV/bin/python" -u "$REPO/debug/ctc_vllm_validation/grade_responses_generic.py" \
    --responses "$WORK/responses_${R}.jsonl" --eval-data "$EV" --task "$EVAL_TASK" \
    --out "$DEST/rung_${R}.vllm.json" || { echo "grade rc=$? rung $R"; continue; }
  grep -oE '"[a-z_]*f1[a-z_]*": *[0-9.]+' "$DEST/rung_${R}.vllm.json" | head -2
done
echo "=== VLLM-PAIR-EVAL DONE $(date '+%F %T') -> $DEST ==="
