#!/bin/bash
# One-shot: export the LOCALLY-trained scifact dense retrain checkpoint (already on cubbins
# /data from run_ctc_local.sbatch, no S3 round-trip needed) -> HF -> vLLM serving copy -> eval
# all 5 scifact rungs -> push grades to S3 ctc_dense_results/scifact/.
# Adapted from debug/ctc_vllm_validation/node_local/prep_task.sh + gpu_eval_task.sh, skipping
# prep_task.sh's step 0b (S3 checkpoint stage-in) since the checkpoint is already node-local.
#SBATCH --job-name=scifact-retrain-eval
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --account=site
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/prasann/joblogs/scifact_retrain_eval_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$MIRROR_DIR"
_SRC_LOG="/data/prasann/joblogs/scifact_retrain_eval_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/scifact_retrain_eval_${SLURM_JOB_ID}.log"
( while true; do cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null; sleep 20; done ) &
_MIRROR_PID=$!
trap '_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }; _mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

RUN="${RUN:?set RUN=ctc-4b-scifact-full-retrain-s1 (the training run-name / local ckpt dir)}"
# TASK_LABEL is what gpu_eval_task.sh's TASK derives SERVE/results/S3-push paths from
# (SERVE=vllm_serving_4b/ctc-4b-${TASK}-full, S3 push -> ctc_dense_results/${TASK}/). Default
# to the canonical "scifact" so grades land at the SAME S3/results location as every other task
# (ctc_dense_results/scifact/) -- this intentionally OVERWRITES the collapsed checkpoint's HF
# export + serving copy (the raw collapsed olmo-core checkpoint itself is untouched, still on
# S3/beaker/weka, so nothing is unrecoverably lost). Override TASK_LABEL to keep them side by
# side instead (but then also override S3_TASK_OVERRIDE below to still push to .../scifact/).
TASK_LABEL="${TASK_LABEL:-scifact}"
LOCAL_CKPT="/data/prasann/ctc_suite/ckpts/${RUN}"
HF_EXPORT="/data/prasann/ctc_suite/hf_exports_4b/ctc-4b-${TASK_LABEL}-full"
SERVE="/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-${TASK_LABEL}-full"
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
BASE_MODEL_ID="Qwen/Qwen3.5-4B-Base"
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
LOCAL_REPO=$REPO_ROOT

echo "=== [scifact-retrain-eval] HOST=$(hostname) START=$(date '+%F %T') RUN=$RUN TASK_LABEL=$TASK_LABEL ==="

if [ ! -f "$LOCAL_CKPT/model_and_optim/.metadata" ]; then
  echo "FATAL: no checkpoint at $LOCAL_CKPT"; exit 2
fi

PREAMBLE=/data/prasann/node_local_env.sh
[ -f "$PREAMBLE" ] || { echo "FATAL: $PREAMBLE missing -- run prep_task.sh for another task first"; exit 3; }
source "$PREAMBLE"
mkdir -p "$HOME" "$HOME/.cache" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$FLASHINFER_CACHE_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$HF_HOME" "$PIP_CACHE_DIR"
if [ ! -f "$HOME/.aws/credentials" ]; then
  mkdir -p "$HOME/.aws"
  cp /accounts/projects/berkeleynlp/prasann/.aws/config /accounts/projects/berkeleynlp/prasann/.aws/credentials "$HOME/.aws/" 2>/dev/null
  chmod 600 "$HOME/.aws/config" "$HOME/.aws/credentials" 2>/dev/null
fi

# ---- 0c. STALE-EXPORT GUARD: HF_EXPORT/SERVE at these paths may already exist from the
# ORIGINAL collapsed checkpoint (same TASK_LABEL=scifact path convention). A plain
# "skip if exists" check would silently eval the OLD const-[1] model instead of our retrain.
# Tag each export with a SOURCE_RUN marker; if it doesn't match $RUN, move the stale dir aside
# (never delete) and force a fresh export/serve build below.
if [ -d "$HF_EXPORT" ] && [ "$(cat "$HF_EXPORT/SOURCE_RUN" 2>/dev/null)" != "$RUN" ]; then
  echo "[scifact-retrain-eval] stale HF export at $HF_EXPORT (not from RUN=$RUN) -- archiving aside"
  mv "$HF_EXPORT" "${HF_EXPORT}-preretrain-orig-$(date +%s)"
fi
if [ -d "$SERVE" ] && [ "$(cat "$SERVE/SOURCE_RUN" 2>/dev/null)" != "$RUN" ]; then
  echo "[scifact-retrain-eval] stale serving copy at $SERVE (not from RUN=$RUN) -- archiving aside"
  mv "$SERVE" "${SERVE}-preretrain-orig-$(date +%s)"
fi

# ---- 1. export to HF (network needed once for the hybrid-config snapshot_download) ----
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
unset TRANSFORMERS_CACHE
export PYTHONPATH="/data/prasann/tf_qwen35_target:$PYTHONPATH"

if [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "[scifact-retrain-eval] exporting ckpt -> HF $(date '+%F %T')"
  # TODO(port): depends on src/corpus_reasoning/train/export_olmo_to_hf.py, not ported
  python "$LOCAL_REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
    --save-folder "$LOCAL_CKPT" --ckpt "$LOCAL_CKPT" \
    --hf-out "$HF_EXPORT" --base-model "$BASE_MODEL_ID"
  rc=$?
  if [ $rc -ne 0 ]; then echo "!!! export FAILED rc=$rc"; exit 1; fi
  echo "$RUN" > "$HF_EXPORT/SOURCE_RUN"
else
  echo "[scifact-retrain-eval] HF export already exists: $HF_EXPORT"
fi

# ---- 2. vLLM serving copy ----
if [ ! -f "$SERVE/model.safetensors" ] || [ ! -f "$SERVE/preprocessor_config.json" ]; then
  echo "[scifact-retrain-eval] building vLLM serving copy $(date '+%F %T')"
  python "$LOCAL_REPO/research/landmark/ctc_vllm_validation/make_vllm_serving_copy.py" \
    --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out "$SERVE"
  python "$LOCAL_REPO/research/landmark/ctc_vllm_validation/make_vl_weights.py" \
    --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out-dir "$SERVE"
  rc=$?
  if [ $rc -ne 0 ]; then echo "!!! serving-copy build FAILED rc=$rc"; exit 1; fi
  echo "$RUN" > "$SERVE/SOURCE_RUN"
else
  echo "[scifact-retrain-eval] serving copy already built: $SERVE"
fi

# ---- 3. eval all rungs via the general vLLM driver ----
export TASK="${TASK_LABEL}"   # names SERVE dir (matches what we just built) + results dir
export EVAL_TASK=beir_scifact # canonical task alias for build_prefills_any/grade_any
export RUNG_TASK=scifact      # reuse the existing eval_rungs/scifact/ dir
export RUNGS="2048 4096 8192 16384 32768"
export KEEP_RESPONSES=1       # keep generations so we can dump/verify const-pred is gone

# TODO(port): depends on debug/ctc_vllm_validation/node_local/, not ported
bash "$LOCAL_REPO/debug/ctc_vllm_validation/node_local/gpu_eval_task.sh"
rc=$?
echo "=== [scifact-retrain-eval] DONE $(date '+%F %T') rc=$rc ==="
exit $rc
