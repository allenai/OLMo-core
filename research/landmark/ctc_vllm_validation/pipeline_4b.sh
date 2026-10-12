#!/bin/bash
# 4B FULL-arm vLLM parity check on horton (mirrors pipeline.sh's 0.8B validation, but against
# the REAL target scale + the native 4B contradiction reference that just landed:
# results/ctc_suite/contradiction/qwen3.5-4b_full/rung_2048.json, f1=0.5713, eval_size=500).
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
VDIR="$REPO/research/landmark/ctc_vllm_validation"

# Allow network egress for the ONE-TIME snapshot_download inside export_olmo_to_hf.py's
# hybrid-Qwen3.5 path (needs the real HF repo id, not a bare local dir, to read text_config) --
# override local_env.sh's default offline flags BEFORE sourcing it (its own documented escape
# hatch), then source for REPO/PATH/PYTHONPATH conventions.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
source "$REPO/recipes/env/local_env.sh"
export PYTHONPATH="$REPO/research/common:${PYTHONPATH:-}"  # corpus_reasoning.* lives in research/common
unset TRANSFORMERS_CACHE
export HF_HOME=/data/prasann/hf_dl_4b
mkdir -p "$HF_HOME"
# export_olmo_to_hf.py's hybrid (Qwen3.5) path calls transformers.Qwen3_5ForCausalLM /
# AutoConfig with model_type="qwen3_5" -- the corpus-reasoning-olmo env's own transformers
# (4.57.6) predates real Qwen3.5 support and raises KeyError('qwen3_5'). convert_both.sh
# (the 0.8B export) works around this with a "shadow" transformers build on PYTHONPATH;
# do the same here (does not affect the vLLM venv -- that has its own transformers 5.14.1
# with native qwen3_5 support already).
export PYTHONPATH="/scratch/users/prasann/tf_qwen35_target:$PYTHONPATH"

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
$VLLM_PY -c "import vllm; print('vllm', vllm.__version__)"

CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b
HF_EXPORT=/data/prasann/ctc_suite/hf_exports_4b/ctc-s5-contra-full-4b
BASE_MODEL_ID="Qwen/Qwen3.5-4B-Base"
BASE_SNAP_LOCAL=/data/prasann/hf_models/Qwen3.5-4B-Base   # flat local copy (used for the VL wrapper's vision tower + preprocessor files)

if [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "=== exporting 4B ckpt -> HF $(date '+%F %T') ==="
  # TODO(port): depends on src/corpus_reasoning/train/export_olmo_to_hf.py, not ported
  python "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
    --save-folder "$CKPT" --ckpt "$CKPT" \
    --hf-out "$HF_EXPORT" --base-model "$BASE_MODEL_ID"
else
  echo "HF export already exists: $HF_EXPORT"
fi

SERVE=/data/prasann/ctc_suite/vllm_serving_4b_v3/ctc-s5-contra-full-4b
if [ ! -f "$SERVE/model.safetensors" ] || [ ! -f "$SERVE/preprocessor_config.json" ]; then
  echo "=== building vLLM serving copy $(date '+%F %T') ==="
  python "$VDIR/make_vllm_serving_copy.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP_LOCAL" --out "$SERVE"
  python "$VDIR/make_vl_weights.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP_LOCAL" --out-dir "$SERVE"
else
  echo "serving copy already built: $SERVE"
fi

RUNG_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction/rung_2048.jsonl
PREFILLS=$VDIR/prefills_rung2048_4b.json
echo "=== building prefills (tokenizer=local snapshot, avoids HF-hub cache lookups) $(date '+%F %T') ==="
python "$VDIR/build_prefills.py" \
  --tokenizer "$BASE_SNAP_LOCAL" \
  --contra-data "$RUNG_JSONL" \
  --max-test-samples 100000 \
  --cot-mode none \
  --out "$PREFILLS"

echo "=== vLLM FULL arm (4B) $(date '+%F %T') ==="
$VLLM_PY "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode full \
  --max-new-tokens 512 --max-model-len 16384 \
  --out "$VDIR/responses_full_4b.json"

python "$VDIR/grade_responses.py" \
  --responses "$VDIR/responses_full_4b.json" --contra-data "$RUNG_JSONL" \
  --max-test-samples 100000 --out "$VDIR/grade_full_4b.json"

echo "=== DONE $(date '+%F %T') ==="
echo "--- grade_full_4b.json ---"; cat "$VDIR/grade_full_4b.json"
