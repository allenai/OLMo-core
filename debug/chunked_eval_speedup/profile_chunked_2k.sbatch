#!/bin/bash
# Attribute the chunked-eval slowdown to a stage, on the real model + real rung.
#
# Chunked eval is 20-29x slower than dense. A CPU microbenchmark
# (debug/chunked_eval_speedup/bench_chunk_ids.py) ruled OUT the chunk_ids rebuild as the cause:
# it costs ~4.5 s per 500 examples at 2k against a 1199 s total, i.e. ~0.4%. So the time is in
# create_block_mask_compiled and/or the FlexAttention kernel itself, and this measures which.
#
# CHUNK_PROFILE=1 accumulates per-stage timings inside vllm_chunked_patch and prints at exit:
#   orig_build       -- vLLM's own metadata build
#   build_chunk_ids  -- our per-step token scan (expected small)
#   make_mask_mod    -- building the per-step closure
#   build_block_mask -- create_block_mask_compiled over the chunked mask_mod
# Anything left over vs. gen_seconds is the attention kernel + model forward.
#
# Small on purpose: 48 examples at the 2k rung, so it lands in minutes.
#SBATCH --job-name=chunk-profile
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_profile_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
SRCREPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOCAL_REPO=/data/prasann/repo/OLMo-core
VD=$LOCAL_REPO/debug/ctc_vllm_validation
OUT=/data/prasann/chunk_profile
mkdir -p /data/prasann/joblogs "$OUT"

# Ship the instrumented patch to the node-local repo the driver imports from.
cp "$SRCREPO/src/corpus_reasoning/lib/vllm_chunked_patch.py" \
   "$LOCAL_REPO/src/corpus_reasoning/lib/vllm_chunked_patch.py"
echo "=== instrumented patch synced: $(grep -c _prof_add "$LOCAL_REPO/src/corpus_reasoning/lib/vllm_chunked_patch.py") call sites ==="

source /data/prasann/node_local_env.sh
mkdir -p "$HOME" "$TMPDIR" "$FLASHINFER_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"
export PATH=$CUDA_HOME/bin:$PATH
PY=/data/prasann/ctc_vllm_venv/bin/python
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_clean/rung_2048.jsonl
SERVE_C=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
SERVE_D=/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-contradiction-full
BASE=/data/prasann/hf_models/Qwen3.5-4B-Base

python "$VD/general/build_prefills_any.py" --tokenizer "$BASE" --task contradiction \
  --eval-data "$RUNG" --max-test-samples 48 --out "$OUT/prefills48.json" || exit 1

echo ""
echo "############ DENSE reference (48 ex) ############"
$PY -u "$VD/run_vllm_eval.py" --hf-model "$SERVE_D" --prefills "$OUT/prefills48.json" \
  --mode full --task contradiction --max-new-tokens 512 --max-model-len 4096 \
  --gpu-mem-util 0.85 --out "$OUT/resp_dense48.json" 2>&1 | grep -E "gen_seconds|Processed|load_seconds" | tail -3

echo ""
echo "############ CHUNKED profiled (48 ex) ############"
CHUNK_PROFILE=1 $PY -u "$VD/run_vllm_eval.py" --hf-model "$SERVE_C" --prefills "$OUT/prefills48.json" \
  --mode chunked --task contradiction --max-new-tokens 512 --max-model-len 4096 \
  --gpu-mem-util 0.85 --out "$OUT/resp_chunked48.json" 2>&1 | tail -40

echo ""
echo "=== gen_seconds comparison ==="
for f in resp_dense48 resp_chunked48; do
  [ -f "$OUT/$f.json" ] && python -c "
import json;d=json.load(open('$OUT/$f.json'))
print(f\"  $f gen_s={d.get('gen_seconds',0):8.1f} patch={d.get('patch_debug')}\")"
done
