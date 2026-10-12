#!/bin/bash
# Full vLLM parity pipeline for CTC-suite contradiction (0.8B first, fast + low-risk proof).
# Runs INSIDE a GPU allocation. Reuses build_prefills.py / run_vllm_eval.py / grade_responses.py /
# make_vllm_serving_copy.py / make_vl_weights.py from this directory (prior agent's validation gate).
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
VDIR="$REPO/research/landmark/ctc_vllm_validation"
source "$REPO/recipes/env/local_env.sh"
export PYTHONPATH="$REPO/research/common:${PYTHONPATH:-}"  # corpus_reasoning.* lives in research/common
# The tokenizer/base-model snapshot lives under this node-local HF cache (set by the
# prior export job's convert_both.sh) -- local_env.sh's offline flags stay on, but
# without overriding HF_HOME here we'd look in ~/.cache/huggingface (empty, offline) and
# fail even though the snapshot is present locally. Also: the shell profile exports a
# stale TRANSFORMERS_CACHE (deprecated var, still honored ahead of HF_HOME in transformers
# 4.57) pointing at an empty /scratch dir -- unset it or it silently shadows HF_HOME.
unset TRANSFORMERS_CACHE
export HF_HOME=/data/prasann/hf_cache

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# ---- locate the vLLM venv ----
VLLM_PY=""
for cand in /data/prasann/ctc_vllm_venv/bin/python /scratch/users/prasann/cc_vllm_venv/bin/python; do
  if [ -x "$cand" ]; then VLLM_PY="$cand"; break; fi
done
if [ -z "$VLLM_PY" ]; then
  echo "FATAL: no vLLM venv found (checked /data/prasann/ctc_vllm_venv, /scratch/users/prasann/cc_vllm_venv)"
  exit 1
fi
echo "VLLM_PY=$VLLM_PY"
$VLLM_PY -c "import vllm, transformers; print('vllm', vllm.__version__, 'transformers', transformers.__version__)"

# ---- locate HF exports (prior agent already ran export_olmo_to_hf.py for these two 0.8B ckpts) ----
HF_FULL=/data/prasann/ctc_suite/hf_exports/ctc-1ep-contra-full
HF_CMIX=/data/prasann/ctc_suite/hf_exports/ctc-smoke-contra-3ep
for d in "$HF_FULL" "$HF_CMIX"; do
  if [ ! -f "$d/config.json" ]; then echo "FATAL: missing HF export $d"; exit 1; fi
done

# ---- locate Qwen3.5-0.8B base snapshot (VL config; needed for the vLLM serving-copy wrapper) ----
BASE_SNAP=$(find /data/prasann/hf_cache /data/prasann/hf ~/.cache/huggingface -maxdepth 6 -iname "snapshots" -path "*Qwen3.5-0.8B*" 2>/dev/null \
  | xargs -I{} find {} -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -z "$BASE_SNAP" ]; then
  BASE_SNAP=$(find /data/prasann -maxdepth 8 -iname "config.json" 2>/dev/null | xargs grep -l "qwen3_5" 2>/dev/null | xargs -I{} dirname {} | grep -i "0.8B\|0\.8b" | head -1)
fi
echo "BASE_SNAP=$BASE_SNAP"
if [ -z "$BASE_SNAP" ]; then echo "FATAL: could not locate Qwen3.5-0.8B base snapshot"; exit 1; fi

# ---- build vLLM-servable copies (idempotent) ----
build_serving_copy() {
  local hf_export="$1" out="$2"
  if [ -f "$out/model.safetensors" ] && [ -f "$out/config.json" ] && [ -f "$out/preprocessor_config.json" ]; then
    echo "  serving copy already built: $out"
    return 0
  fi
  python "$VDIR/make_vllm_serving_copy.py" --hf-export "$hf_export" --base-snapshot "$BASE_SNAP" --out "$out"
  python "$VDIR/make_vl_weights.py" --hf-export "$hf_export" --base-snapshot "$BASE_SNAP" --out-dir "$out"
}
SERVE_FULL=/data/prasann/ctc_suite/vllm_serving/ctc-1ep-contra-full
SERVE_CMIX=/data/prasann/ctc_suite/vllm_serving/ctc-smoke-contra-3ep
echo "=== building serving copies $(date '+%F %T') ==="
build_serving_copy "$HF_FULL" "$SERVE_FULL"
build_serving_copy "$HF_CMIX" "$SERVE_CMIX"

# ---- build native-identical prefills (corpus-reasoning-olmo env: has olmo_core) ----
RUNG_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction/rung_2048.jsonl
PREFILLS=$VDIR/prefills_rung2048.json
echo "=== building prefills $(date '+%F %T') ==="
python "$VDIR/build_prefills.py" \
  --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --contra-data "$RUNG_JSONL" \
  --max-test-samples 100000 \
  --cot-mode none \
  --out "$PREFILLS"

# ---- FULL arm: vLLM generate + grade ----
# max-model-len: the "rung_2048" file is NOT hard-capped at 2048 tokens (contradiction's
# n=20-doc prompts vary quite a bit) -- build_prefills.py reported max len 11161. The
# native evaluator's --max-length 4096 is just a KV-cache sizing hint there (it handled
# every example without an eval_size shortfall), but vLLM hard-rejects any prompt over
# max_model_len, so give it real headroom.
echo "=== vLLM FULL arm $(date '+%F %T') ==="
$VLLM_PY "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE_FULL" --prefills "$PREFILLS" --mode full \
  --max-new-tokens 512 --max-model-len 16384 \
  --out "$VDIR/responses_full.json"

python "$VDIR/grade_responses.py" \
  --responses "$VDIR/responses_full.json" --contra-data "$RUNG_JSONL" \
  --max-test-samples 100000 --out "$VDIR/grade_full.json"

# ---- native chunked-mix reference at the SAME rung file (needed for chunked-arm parity;
#      the only native rung_2048 reference on record today is the FULL arm) ----
# NOTE: pass the ckpt ROOT (config.json lives there; the loader finds the latest step's
# model_and_optim/ under it internally), matching how the existing full-arm native
# reference was invoked -- NOT a step subdir (that has model_and_optim/ but no config.json).
NATIVE_CMIX_CKPT=/data/prasann/ctc_suite/ckpts/ctc-smoke-contra-3ep
NATIVE_CMIX_RESULT="$REPO/results/ctc_suite/contradiction/qwen3.5-0.8b_chunked-mix/rung_2048.json"
echo "NATIVE_CMIX_CKPT=$NATIVE_CMIX_CKPT"
if [ -f "$NATIVE_CMIX_RESULT" ]; then
  echo "native chunked-mix reference already recorded: $NATIVE_CMIX_RESULT (skipping re-run)"
elif [ -n "$NATIVE_CMIX_CKPT" ]; then
  echo "=== native chunked-mix reference (bs=1) $(date '+%F %T') ==="
  cd "$REPO"
  python -u -m scripts.eval.ctc_suite.run_rung_eval \
    --task contradiction \
    --ckpt "$NATIVE_CMIX_CKPT" \
    --variant chunked \
    --rung-tokens 2048 \
    --eval-jsonl "$RUNG_JSONL" \
    --arm chunked-mix \
    --model-scale qwen3.5-0.8b \
    --tokenizer Qwen/Qwen3.5-0.8B-Base \
    --nproc 1 \
    --master-port "$MASTER_PORT" \
    2>&1 | tee "$VDIR/native_chunked_mix_rung2048.log"
else
  echo "WARNING: no native chunked-mix ckpt found; skipping native reference re-run"
fi

# ---- CHUNKED arm: vLLM generate + grade ----
echo "=== vLLM CHUNKED arm $(date '+%F %T') ==="
$VLLM_PY "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE_CMIX" --prefills "$PREFILLS" --mode chunked \
  --max-new-tokens 512 --max-model-len 16384 \
  --out "$VDIR/responses_chunked.json"

python "$VDIR/grade_responses.py" \
  --responses "$VDIR/responses_chunked.json" --contra-data "$RUNG_JSONL" \
  --max-test-samples 100000 --out "$VDIR/grade_chunked.json"

echo "=== DONE $(date '+%F %T') ==="
echo "--- grade_full.json ---"; cat "$VDIR/grade_full.json"
echo "--- grade_chunked.json ---"; cat "$VDIR/grade_chunked.json"
