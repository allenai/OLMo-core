#!/bin/bash
# Chunked-eval speedup, round 2: is the FALLBACK BlockMask build's KV-cache term the long-rung cost?
#
# WHY THIS IS NOT A REPEAT. The 2026-08-04 investigation (paper-v2-todo-status.md, "CLOSED with a
# negative result") measured three things and refuted all three: the chunk_ids rebuild (0.4%),
# create_block_mask not being compiled (it is), and the BLOCK_N clamp (1.6%). Every one of those was
# measured at the **2k rung**, where each prompt prefills in a single 2048-token step and the KV
# cache sits at its 128-page floor. That is the rung where the mask build matters LEAST. Its verdict
# ("the 7x is inherent to FlexAttention") may still be right, but it was never tested where the pain
# actually is.
#
# THE UNTESTED TERM. We are always on the fallback path -- the pow2 BLOCK_N clamp forces
# kv_block_size != block_size, which sets vLLM's direct_build=False (flex_attention.py:882). The
# fallback is
#     create_block_mask_compiled(mask_mod, q_len=num_actual_tokens, kv_len=total_cache_tokens, ...)
# and kv_len is the ENTIRE allocated cache, not the live sequence lengths. So
#     mask cost per unit of work  ~  total_cache_tokens / kv_block_size
# which is independent of how many sequences are resident, and scales with the rung because
# n_pages = (max_model_len * 10) // 544 grows with the auto-bumped max_model_len. At the 32k rung
# that is ~10 x 33k = 330k cache tokens evaluated on EVERY forward, whether 8 sequences are running
# or 1. If that term dominates, shrinking the allocation is a large, free win -- and it would be
# invisible at 2k, which is exactly why the first pass missed it.
#
# Arms (24 examples, 8k rung, identical prompts). A is today's production config.
#SBATCH --job-name=chunk-kv-sweep
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_kv_sweep_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

VDIR=/data/prasann/repo/OLMo-core/debug/ctc_vllm_validation
source /data/prasann/node_local_env.sh 2>/dev/null || true
export HOME=/data/prasann/home
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONPATH=/data/prasann/repo/OLMo-core/src
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python

WORK=/data/prasann/chunk_kv_sweep
mkdir -p "$WORK" /data/prasann/joblogs
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
N=24

echo "=== [chunk-kv-sweep] HOST=$(hostname) START=$(date '+%F %T') ==="

PREFILLS="$WORK/prefills_8192_${N}.json"
if [ ! -f "$PREFILLS" ]; then
  python "$VDIR/general/build_prefills_any.py" \
    --tokenizer "$BASE_SNAP" --task contradiction --eval-data "$RUNG" \
    --max-test-samples $N --out "$PREFILLS" || exit 1
fi

run_arm () {  # name headroom batched seqs
  local name="$1" hr="$2" bt="$3" ms="$4"
  echo ""
  echo "######## ARM $name  headroom=$hr batched=$bt seqs=$ms  $(date '+%F %T') ########"
  CHUNK_SEQ_HEADROOM="$hr" CHUNK_MAX_BATCHED_TOKENS="$bt" CHUNK_MAX_NUM_SEQS="$ms" \
  CHUNK_PROFILE=1 \
  $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.85 \
    --out "$WORK/resp_${name}.json" 2>&1 | grep -E "gen_seconds|bumping max_model_len|chunked KV config|profile|build_block_mask|orig_build|build_chunk_ids|Error|Traceback|OOM|out of memory" | tail -20
  python - <<PYEOF
import json
try:
    d=json.load(open("$WORK/resp_${name}.json"))
    print(f"ARM $name  gen_seconds={d.get('gen_seconds'):.1f}  load={d.get('load_seconds',0):.0f}  n={d.get('eval_size')}")
except Exception as e:
    print(f"ARM $name  NO RESULT ({e})")
PYEOF
}

# A = production baseline. B/C isolate the cache-allocation term. D/E add the concurrency knobs.
run_arm A_baseline      10 2048  8
run_arm B_cache_small    2 2048  2
run_arm C_cache_mid      4 2048  4
run_arm D_bigbatch      10 8192  8
run_arm E_small_bigbatch  2 8192  2

echo ""
echo "=== SUMMARY ==="
for a in A_baseline B_cache_small C_cache_mid D_bigbatch E_small_bigbatch; do
  python - <<PYEOF
import json
try:
    d=json.load(open("$WORK/resp_${a}.json")); print(f"  {'$a':<18} gen_seconds={d.get('gen_seconds'):.1f}")
except Exception: print(f"  {'$a':<18} MISSING")
PYEOF
done
echo "=== [chunk-kv-sweep] DONE $(date '+%F %T') ==="
