#!/bin/bash
# Is the forced BLOCK_N=32 what makes chunked eval 7x slower than dense?
#
# Profiling (job 3422856, 48 ex, contradiction clean 2k) attributed the runtime as:
#   build_block_mask 11.3s | orig_build 7.1s | build_chunk_ids 2.4s | make_mask_mod 0.0s
#   -> patch bookkeeping 20.7s of 123.5s (17%); kernel + forward = 102.8s (83%).
#   dense reference: 17.4s total.
# So the lever is the FlexAttention kernel. Qwen3.5 pads the attention page to 288 to align with
# the mamba page, and 288 = 2^5 * 9, so _patch_flex_kernel_options_pow2 clamps the triton kernel's
# KV tile to the largest power-of-2 divisor = 32 (a pure-softmax model would get 128+).
#
# This measures the SENSITIVITY: if 16 vs 32 barely moves the clock, tile size is not the lever and
# the slowdown is inherent to FlexAttention -- stop optimizing. If 16 is much worse than 32, then
# tile size dominates and it is worth pursuing a power-of-2 attention page so 128 becomes reachable.
#
# CORRECTNESS: BLOCK_N only sets the kernel's inner tile; the mask itself is unchanged. Each arm
# also re-grades, so any arm that alters the f1 is by definition unsafe and gets rejected.
#SBATCH --job-name=blockn-sweep
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/blockn_sweep_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
SRCREPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOCAL_REPO=/data/prasann/repo/OLMo-core
VD=$LOCAL_REPO/debug/ctc_vllm_validation
OUT=/data/prasann/chunk_profile
mkdir -p /data/prasann/joblogs "$OUT"
cp "$SRCREPO/src/corpus_reasoning/lib/vllm_chunked_patch.py" \
   "$LOCAL_REPO/src/corpus_reasoning/lib/vllm_chunked_patch.py"

source /data/prasann/node_local_env.sh
mkdir -p "$HOME" "$TMPDIR" "$FLASHINFER_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"
export PATH=$CUDA_HOME/bin:$PATH
PY=/data/prasann/ctc_vllm_venv/bin/python
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_clean/rung_2048.jsonl
SERVE_C=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix

for BN in 32 16 64; do
  echo ""
  echo "############ CHUNK_BLOCK_N=$BN ############"
  CHUNK_BLOCK_N=$BN CHUNK_PROFILE=1 $PY -u "$VD/run_vllm_eval.py" \
    --hf-model "$SERVE_C" --prefills "$OUT/prefills48.json" \
    --mode chunked --task contradiction --max-new-tokens 512 --max-model-len 4096 \
    --gpu-mem-util 0.85 --out "$OUT/resp_bn${BN}.json" 2>&1 \
    | grep -E "generated 48|stage profile|build_block_mask|orig_build|build_chunk_ids|Error|error" | head -8
  $PY "$VD/grade_responses.py" --responses "$OUT/resp_bn${BN}.json" --contra-data "$RUNG" \
    --max-test-samples 100000 --out "$OUT/grade_bn${BN}.json" >/dev/null 2>&1
done

echo ""
echo "=== BLOCK_N sweep: time and f1 (f1 MUST be identical across arms) ==="
for BN in 32 16 64; do
  python -c "
import json
try:
    r=json.load(open('$OUT/resp_bn${BN}.json')); g=json.load(open('$OUT/grade_bn${BN}.json'))
    print(f'  BLOCK_N=${BN}:  gen_s={r.get(\"gen_seconds\",0):7.1f}   f1={g.get(\"metric_value\",float(\"nan\")):.6f}')
except Exception as e:
    print(f'  BLOCK_N=${BN}:  FAILED ({e})')
"
done
