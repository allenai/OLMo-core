#!/bin/bash
# General native (olmo_core) CTC eval: run_rung_eval on one checkpoint at one rung, for ANY task /
# variant / arm. Generalizes debug/obliq_synthetic/eval_obliq_viab.sbatch (which hardcoded
# retrieval/dense/full). Node-pin to the training host (-w) since checkpoints live on node-local
# /data. Node-local JIT caches (HOME/TRITON/FLASHINFER on /data) avoid the NFS deadlock.
#
# Env knobs: CKPT(dir w/ config.json+model_and_optim) TASK VARIANT ARM SCALE RUNG EVAL_JSONL
#            NGPU MAXLEN OUTROOT
#   sbatch -w sneetches --gres=gpu:H200:2 --qos=preemptive_high --account=site --partition=jsteinhardt \
#     --export=ALL,CKPT=...,TASK=oolong,VARIANT=chunked,ARM=chunked-mix,SCALE=qwen3.5-4b,RUNG=2048,\
#EVAL_JSONL=...,NGPU=2,MAXLEN=4096 research/landmark/ctc_vllm_validation/eval_ctc_native.sbatch
#SBATCH --job-name=ctc-native-eval
#SBATCH --partition=jsteinhardt
#SBATCH --account=site
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/data/prasann/joblogs/ctc_native_eval_%x_%j.log
set -uo pipefail
MIRROR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$MIRROR" 2>/dev/null || true
SRC=/data/prasann/joblogs/ctc_native_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
DST=$MIRROR/eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
( while true; do cp -f "$SRC" "$DST" 2>/dev/null || true; sleep 20; done ) &
trap 'cp -f "$SRC" "$DST" 2>/dev/null || true; kill %1 2>/dev/null' EXIT

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
NGPU="${NGPU:-2}"
TASK="${TASK:?}"; VARIANT="${VARIANT:-dense}"; ARM="${ARM:-full}"; SCALE="${SCALE:-qwen3.5-4b}"
RUNG="${RUNG:-2048}"; CKPT="${CKPT:?}"; EVAL_JSONL="${EVAL_JSONL:?}"
OUTROOT="${OUTROOT:-$REPO/research/landmark/ctc_vllm_validation/native_eval_out/${TASK}_${ARM}}"
MAXLEN="${MAXLEN:-}"

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/research/common:$REPO/src"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HOME=/data/prasann TRITON_CACHE_DIR=/data/prasann/triton_cache FLASHINFER_CACHE_DIR=/data/prasann/flashinfer_cache
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR" "$OUTROOT" 2>/dev/null || true
mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "=== HOST=$(hostname) TASK=$TASK VARIANT=$VARIANT ARM=$ARM CKPT=$CKPT RUNG=$RUNG CVD=$CUDA_VISIBLE_DEVICES START=$(date '+%F %T') ==="
cd "$REPO"
python -m scripts.eval.ctc_suite.run_rung_eval \
  --task "$TASK" --ckpt "$CKPT" --variant "$VARIANT" --arm "$ARM" --model-scale "$SCALE" \
  --rung-tokens "$RUNG" --eval-jsonl "$EVAL_JSONL" \
  ${MAXLEN:+--max-length "$MAXLEN"} \
  --out-root "$OUTROOT" --small-eval-ok --nproc "$NGPU"
echo "=== DONE $(date '+%F %T') rc=$? ==="
