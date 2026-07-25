#!/bin/bash
# Evaluate ONE length-mix arm on Beaker: olmo checkpoint -> HF -> vLLM serving copy -> vLLM
# generation at 3 rungs -> graded f1. Writes one JSON per arm to weka.
#
# Why vLLM and not the native backend: on this exact family the native path reads f1 0.571 at 2k
# where vLLM reads 0.849 -- a degraded harness, not a model property. Every number this experiment
# reports must come from the same (vLLM) path as the 0.335 baseline, or the comparison is fiction.
#
# Env: ARM (e.g. A3), CKPT (weka olmo save folder). Uses the SHARED venv built by lm-vllm-venv.
set -uo pipefail
echo "=== ARM=$ARM HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LM=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_length_mix
# vllm_venv_fix pins the cu129 triad (torch 2.11.0). The first venv ended at torch 2.13.0+cu130 --
# flashinfer drags torch upward, and vllm 0.25.1's compiled extensions are built against the
# 2.11.0 ABI, so imports succeed but model load does not. Override with VENV=... if needed.
VENV=${VENV:-/weka/oe-training-default/ai2-llm/checkpoints/prasanns/shared/vllm_venv_fix}
BASE_SNAP=$LM/base_snap_small
OUT=$LM/eval_results; mkdir -p "$OUT"
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v cache | head -1 | xargs -r dirname)
[ -z "$REPO" ] && { echo "FATAL: repo not found"; exit 1; }
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
VDIR=$REPO/debug/ctc_vllm_validation
echo "REPO=$REPO"

# gantry --no-python skips the usual `pip install -e .`, so the container python has olmo-core's
# transitive deps only if the image happens to ship them -- this one is missing
# dataclass_extensions, which killed the export 4 s in. Install the package (deps already present
# are satisfied instantly). Only the EXPORT uses this interpreter; vLLM runs from its own venv, so
# nothing here can perturb the inference stack.
python -c "import dataclass_extensions" 2>/dev/null || {
  echo "=== installing olmo-core deps for the export step $(date '+%F %T') ==="
  pip install -q -e "$REPO" 2>&1 | tail -5
  python -c "import dataclass_extensions, olmo_core; print('olmo_core importable')" \
    || { echo "FATAL: olmo_core still not importable after install"; exit 2; }
}

# JIT caches MUST be container-local, never on a shared FS: concurrent arms compiling the same
# flashinfer/triton kernels into one shared cache dir deadlock on the lock file.
export HOME=/root
export FLASHINFER_CACHE_DIR=/root/.cache/flashinfer TRITON_CACHE_DIR=/root/.cache/triton
mkdir -p "$FLASHINFER_CACHE_DIR" "$TRITON_CACHE_DIR"

# --- real system CUDA toolkit (pip metapackages give incoherent nvcc/cuda.h -> flashinfer JIT dies)
. /etc/os-release; UBU="ubuntu${VERSION_ID//./}"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq wget gnupg ca-certificates >/dev/null 2>&1
if ! dpkg -l cuda-keyring >/dev/null 2>&1; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/ck.deb && dpkg -i /tmp/ck.deb >/dev/null 2>&1
fi
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8 cuda-nvrtc-dev-12-8 >/dev/null 2>&1
export CUDA_HOME=/usr/local/cuda-12.8; export PATH="$CUDA_HOME/bin:$PATH"
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "FATAL: no nvcc at $CUDA_HOME"; exit 1; }
echo "nvcc OK: $($CUDA_HOME/bin/nvcc --version | tail -1)"
[ -x "$VENV/bin/python" ] || { echo "FATAL: shared venv missing at $VENV"; exit 1; }

HF=$LM/hf_exports/$ARM
SERVE=$LM/serving/$ARM
mkdir -p "$LM/hf_exports" "$LM/serving"

# --- 1) olmo distcp -> HF (container conda python has olmo_core + torch) ---
if [ ! -f "$HF/config.json" ]; then
  echo "=== [$ARM] export -> HF $(date '+%F %T') ==="
  python "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
    --save-folder "$CKPT" --hf-out "$HF" --base-model Qwen/Qwen3.5-4B-Base \
    || { echo "!!! [$ARM] EXPORT FAILED"; exit 3; }
else echo "[$ARM] HF export exists"; fi

# --- 2) vLLM serving copy (Qwen3.5 VL-wrapper recipe) ---
if [ ! -f "$SERVE/preprocessor_config.json" ]; then
  echo "=== [$ARM] serving copy $(date '+%F %T') ==="
  python "$VDIR/make_vllm_serving_copy.py" --hf-export "$HF" --base-snapshot "$BASE_SNAP" --out "$SERVE" \
    || { echo "!!! [$ARM] SERVING COPY FAILED"; exit 4; }
  python "$VDIR/make_vl_weights.py" --hf-export "$HF" --base-snapshot "$BASE_SNAP" --out-dir "$SERVE" \
    || { echo "!!! [$ARM] VL WEIGHTS FAILED"; exit 4; }
else echo "[$ARM] serving copy exists"; fi

# --- 3) per-rung: prefills -> vLLM generate -> grade ---
IDS="--doc-start-id 248049 --doc-end-id 248050 --eos-token-id 248044"
RES=$OUT/${ARM}.json
echo "{\"arm\": \"$ARM\", \"ckpt\": \"$CKPT\", \"rungs\": {}}" > "$RES.tmp"
for RUNG in 2048 8192 32768; do
  echo "=== [$ARM] rung $RUNG $(date '+%F %T') ==="
  PF=/root/prefills_${ARM}_${RUNG}.json
  RS=/root/resp_${ARM}_${RUNG}.json
  GR=$OUT/${ARM}_rung${RUNG}.grade.json
  python "$VDIR/build_prefills.py" --tokenizer "$BASE_SNAP" \
    --contra-data "$LM/eval_rungs/rung_${RUNG}.jsonl" --max-test-samples 100000 \
    --cot-mode none $IDS --out "$PF" || { echo "!!! [$ARM] rung $RUNG prefills FAILED"; continue; }
  # max-model-len is a floor; run_vllm_eval raises it to (longest prefill + gen + 256).
  "$VENV/bin/python" -u "$VDIR/run_vllm_eval.py" --hf-model "$SERVE" --prefills "$PF" \
    --mode full --model-family qwen3_5 --max-new-tokens 512 --max-model-len 8192 \
    --gpu-mem-util 0.90 --out "$RS" || { echo "!!! [$ARM] rung $RUNG VLLM FAILED"; continue; }
  python "$VDIR/grade_responses.py" --responses "$RS" \
    --contra-data "$LM/eval_rungs/rung_${RUNG}.jsonl" --max-test-samples 100000 --out "$GR" \
    || { echo "!!! [$ARM] rung $RUNG GRADE FAILED"; continue; }
  echo "--- [$ARM] rung $RUNG ---"; cat "$GR"
done

# --- 4) collect into one arm-level JSON ---
python - "$ARM" "$CKPT" "$OUT" <<'PY'
import json, os, sys, math
arm, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"arm": arm, "ckpt": ckpt, "eval_backend": "vllm", "rungs": {}}
for r in (2048, 8192, 32768):
    p = f"{out}/{arm}_rung{r}.grade.json"
    if not os.path.exists(p):
        res["rungs"][str(r)] = {"error": "missing"}; continue
    d = json.load(open(p)); m = d.get("contradiction", d)
    n = m.get("n") or m.get("eval_size") or 500
    f1 = m.get("f1")
    se = math.sqrt(max(f1, 1e-9) * (1 - f1) / n) if isinstance(f1, (int, float)) and n else None
    res["rungs"][str(r)] = {"f1": f1, "precision": m.get("precision"), "recall": m.get("recall"),
                            "parse_rate": m.get("parse_rate"), "eval_size": n, "binomial_se": se}
json.dump(res, open(f"{out}/{arm}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PY
rm -f "$RES.tmp"
echo "=== [$ARM] EVAL DONE $(date '+%F %T') ==="
