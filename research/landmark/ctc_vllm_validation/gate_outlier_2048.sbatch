#!/bin/bash
# GATING VALIDATION: dense (full-attention) vLLM eval for a NON-contra task (outlier),
# rung_2048, on horton. Proves the GENERAL driver (build_prefills_any.py / run_vllm_eval.py
# --task / grade_any.py) works end-to-end on GPU before fanning the roster out across nodes.
#SBATCH --job-name=gate-outlier-2048
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/gate_outlier_2048_%j.log
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
VDIR="$REPO/research/landmark/ctc_vllm_validation"
# TODO(port): depends on debug/ctc_vllm_validation/general/ (build_prefills_any.py, grade_any.py), not ported
GDIR="$VDIR/general"

TASK=outlier
CKPT_S3_DIR=ctc-4b-outlier-full-20260719T225805-0700
CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-4b-outlier-full
HF_EXPORT=/data/prasann/ctc_suite/hf_exports_4b/ctc-4b-outlier-full
SERVE=/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-outlier-full
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
BASE_MODEL_ID="Qwen/Qwen3.5-4B-Base"
RUNG_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/outlier/rung_2048.jsonl
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python

echo "=== HOST=$(hostname) START=$(date '+%F %T') TASK=$TASK ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# ---- 0. stage checkpoint from S3 (skip if already present) ----
mkdir -p /data/prasann/joblogs
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
else
  echo "HF export already exists: $HF_EXPORT"
fi

# ---- 2. build vLLM serving copy (7-piece recipe pieces 1-3) ----
if [ ! -f "$SERVE/model.safetensors" ] || [ ! -f "$SERVE/preprocessor_config.json" ]; then
  echo "=== building vLLM serving copy $(date '+%F %T') ==="
  python "$VDIR/make_vllm_serving_copy.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out "$SERVE"
  python "$VDIR/make_vl_weights.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out-dir "$SERVE"
else
  echo "serving copy already built: $SERVE"
fi

# piece 3: dummy visual.* weights -- required so vLLM's track_weights_loading doesn't error
# on uninitialized visual.* params. Must run with the vLLM venv's native-qwen3_5 transformers.
NEED_DUMMY_VISUAL=1
if $VLLM_PY -c "
from safetensors import safe_open
with safe_open('$SERVE/model.safetensors', framework='pt') as f:
    keys = f.keys()
import sys
sys.exit(0 if any(k.startswith('visual.') for k in keys) else 1)
"; then
  NEED_DUMMY_VISUAL=0
fi
if [ "$NEED_DUMMY_VISUAL" = "1" ]; then
  echo "=== grafting dummy visual weights $(date '+%F %T') ==="
  $VLLM_PY "$VDIR/add_dummy_visual.py" --serve "$SERVE"
else
  echo "visual.* weights already present in $SERVE/model.safetensors"
fi

# ---- 3. build prefills (corpus-reasoning-olmo env python, CPU-only) ----
PREFILLS=$VDIR/prefills_outlier_2048_4b.json
echo "=== building prefills $(date '+%F %T') ==="
python "$GDIR/build_prefills_any.py" \
  --tokenizer "$BASE_SNAP" --task "$TASK" --eval-data "$RUNG_JSONL" \
  --max-test-samples 100000 --out "$PREFILLS"
rc=$?
if [ $rc -ne 0 ]; then echo "!!! build_prefills_any FAILED rc=$rc"; exit 1; fi

# ---- 4. run vLLM (venv python, GPU) ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8
RESP=$VDIR/responses_outlier_2048_vllm.json
echo "=== vLLM full-attention eval $(date '+%F %T') ==="
PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode full --task "$TASK" \
  --max-new-tokens 256 --max-model-len 4096 --gpu-mem-util 0.85 \
  --out "$RESP"
rc=$?
if [ $rc -ne 0 ]; then echo "!!! run_vllm_eval FAILED rc=$rc"; exit 1; fi

# ---- 5. grade ----
GRADE=$VDIR/grade_outlier_2048_vllm.json
echo "=== grading $(date '+%F %T') ==="
python "$GDIR/grade_any.py" \
  --responses "$RESP" --eval-data "$RUNG_JSONL" --task "$TASK" \
  --max-test-samples 100000 --out "$GRADE"
rc=$?
if [ $rc -ne 0 ]; then echo "!!! grade_any FAILED rc=$rc"; exit 1; fi

echo "=== DONE $(date '+%F %T') ==="
echo "--- $GRADE ---"
cat "$GRADE"
