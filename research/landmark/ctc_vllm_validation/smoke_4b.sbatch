#!/bin/bash
# Smoke test: can vLLM (corpus-reasoning-eval env, vllm 0.18.1) load our converted 4B
# contra HF export directly (no VL-wrapper stitching, no base-snapshot vision weights)
# using the enforce_eager + gdn_prefill_backend=triton recipe, and generate something sane?
# This is the make-or-break step before the full dense-arm vLLM sweep.
#SBATCH --job-name=vllm-4b-smoke
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive_high_sewonm
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/prasann/joblogs/smoke_4b_%j.log
set -uo pipefail

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE
export HF_HOME=/data/prasann/hf
export PYTHONWARNINGS=ignore
export TMPDIR=/data/prasann/tmp
mkdir -p "$TMPDIR"

HF_EXPORT=/data/prasann/ctc_suite/hf_exports_4b/ctc-s5-contra-full-4b
echo "=== HF_EXPORT contents ==="
ls -la "$HF_EXPORT" 2>&1

$PY - "$HF_EXPORT" <<'EOF'
import sys, time
import vllm

hf_export = sys.argv[1]
print("vllm version:", vllm.__version__)
t0 = time.time()
llm = vllm.LLM(
    model=hf_export,
    max_model_len=4096,
    gpu_memory_utilization=0.6,
    enforce_eager=True,
    gdn_prefill_backend="triton",
)
print(f"LOAD_TIME={time.time()-t0:.1f}s", flush=True)
out = llm.generate(
    ["The capital of France is", "1 + 1 ="],
    vllm.SamplingParams(max_tokens=16, temperature=0.0),
)
for o in out:
    print("GENERATION:", repr(o.outputs[0].text))
print("SMOKE_TEST_OK")
EOF
rc=$?
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
