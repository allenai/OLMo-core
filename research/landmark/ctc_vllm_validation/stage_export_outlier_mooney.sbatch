#!/bin/bash
# CPU-only stage: S3 checkpoint -> HF export -> vLLM serving-copy pieces 1-2 (no GPU held).
# Split out from the eval job per coordinator note: don't idle an H200 during S3 staging +
# HF export (~10min on horton before it wedged). Piece 3 (dummy visual, needs the vLLM venv's
# transformers) and the actual GPU eval are separate jobs chained by --dependency.
#SBATCH --job-name=stage-export-outlier-mooney
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/prasann/joblogs/stage_export_outlier_mooney_%j.log
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
VDIR="$REPO/research/landmark/ctc_vllm_validation"

TASK=outlier
CKPT_S3_DIR=ctc-4b-outlier-full-20260719T225805-0700
CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-4b-outlier-full
HF_EXPORT=/data/prasann/ctc_suite/hf_exports_4b/ctc-4b-outlier-full
SERVE=/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-outlier-full
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
BASE_SNAP_SRC=/scratch/users/prasann/hf_models/Qwen3.5-4B-Base
BASE_MODEL_ID="Qwen/Qwen3.5-4B-Base"

echo "=== HOST=$(hostname) START=$(date '+%F %T') TASK=$TASK ==="
mkdir -p /data/prasann/joblogs

# ---- 0a. copy the (metadata-only, ~23MB) base snapshot subset from shared /scratch to
# node-local /data (avoid repeated slow-NFS reads; exclude the incomplete-download .cache) ----
if [ -f "$BASE_SNAP/config.json" ]; then
  echo "base snapshot already local: $BASE_SNAP"
else
  echo "=== copying base snapshot subset from /scratch $(date '+%F %T') ==="
  mkdir -p "$BASE_SNAP"
  rsync -a --exclude='.cache' "$BASE_SNAP_SRC/" "$BASE_SNAP/"
fi

# ---- 0b. stage checkpoint from S3 ----
if [ -f "$CKPT/model_and_optim/.metadata" ]; then
  echo "checkpoint already staged: $CKPT"
else
  echo "=== staging ckpt from S3 $(date '+%F %T') ==="
  export AWS_PROFILE=S3
  mkdir -p "$CKPT"
  aws s3 sync "s3://ai2-llm/checkpoints/prasanns/_transfer/$CKPT_S3_DIR/model_and_optim" \
    "$CKPT/model_and_optim" --only-show-errors
  aws s3 cp "s3://ai2-llm/checkpoints/prasanns/_transfer/$CKPT_S3_DIR/config.json" \
    "$CKPT/config.json" --only-show-errors
fi
ls -la "$CKPT"

# ---- 1. export to HF (needs egress for the one-time hybrid-config snapshot_download) ----
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
source "$REPO/recipes/env/local_env.sh"
export PYTHONPATH="$REPO/research/common:${PYTHONPATH:-}"  # corpus_reasoning.* lives in research/common
unset TRANSFORMERS_CACHE
export HF_HOME=/data/prasann/hf_dl_4b
mkdir -p "$HF_HOME"
export PYTHONPATH="/scratch/users/prasann/tf_qwen35_target:$PYTHONPATH"

if [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "=== exporting ckpt -> HF $(date '+%F %T') ==="
  # TODO(port): depends on src/corpus_reasoning/train/export_olmo_to_hf.py, not ported
  python "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
    --save-folder "$CKPT" --ckpt "$CKPT" \
    --hf-out "$HF_EXPORT" --base-model "$BASE_MODEL_ID"
  rc=$?
  if [ $rc -ne 0 ]; then echo "!!! export FAILED rc=$rc"; exit 1; fi
else
  echo "HF export already exists: $HF_EXPORT"
fi
ls -la "$HF_EXPORT"

# ---- 2. build vLLM serving copy pieces 1-2 (no GPU, no vllm venv needed) ----
if [ ! -f "$SERVE/model.safetensors" ] || [ ! -f "$SERVE/preprocessor_config.json" ]; then
  echo "=== building vLLM serving copy $(date '+%F %T') ==="
  python "$VDIR/make_vllm_serving_copy.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out "$SERVE"
  python "$VDIR/make_vl_weights.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out-dir "$SERVE"
  rc=$?
  if [ $rc -ne 0 ]; then echo "!!! serving-copy build FAILED rc=$rc"; exit 1; fi
else
  echo "serving copy already built: $SERVE"
fi
ls -la "$SERVE"
echo "=== STAGE+EXPORT DONE $(date '+%F %T') (piece 3 dummy-visual + GPU eval are separate jobs) ==="
