#!/bin/bash
# HEAD-TO-HEAD: the HuggingFace/native chunked evaluator vs the vLLM chunked path.
#
# WHY THIS, AFTER FOUR ROUNDS OF TUNING vLLM. Rounds 1-4 established that vLLM's chunked path has a
# STRUCTURAL ceiling, not a tuning one:
#   * the gap to dense is 7.0x = 5.1x step count  x  1.36x per-step cost
#   * step count only falls with concurrency
#   * concurrency is walled by the fallback create_block_mask, which is rebuilt EVERY step over the
#     WHOLE allocated cache -- at seqs=512 that is a 20.34 GiB transient on top of a 72 GiB KV cache,
#     and it OOMs (round 4, job 3440933)
#   * the FREE-query fast path fires but buys 0.2%: at seqs=128 only 19% of steps are all-decode
#     (mixed prefill/decode batches break the condition), and the mask is only ~12% of runtime
#
# The native evaluator does not have that ceiling, because it does what the corpus-reasoning HF eval
# always did (scripts/eval/evaluate.py:621-637): build the chunked mask ONCE for the prefill, then
# decode from the KV cache with plain causal attention --
#     "Dense / full: prefill once (chunked mask applied + K,V cached), then single-token greedy
#      decode over the cache (plain causal since new tokens are FREE)."
#     -- eval_lc_native_docchunk.py:494
# One mask build per EXAMPLE instead of one per STEP. There is no full-grid rebuild to OOM on.
#
# ⚠ VARIANT NAMING IS INVERTED AND IT IS A REAL FOOTGUN. run_beaker_multirung_eval.sh documents it:
# the DRIVER's arms are dense = full attention / chunked = doc-chunked mask, but the UNDERLYING
# evaluator calls full attention `full` and THE CHUNKED MASK `dense`. So the chunked arm here is
# `--variant dense`. Passing `full` would silently measure full attention and look like a huge
# speedup with better f1 -- the exact shape of a false positive.
#
# TASK CHOICE: groups4 and outlier_amzn are the only tasks with BOTH a local distcp step dir (what
# native needs) and a local chunked serving copy (what vLLM needs) on the same node, so neither arm
# pays a staging cost the other does not. contradiction has no chunked distcp left on disk or S3.
#
# THE COMPARISON IS SPEED FIRST, AGREEMENT SECOND. Native and vLLM are known to disagree on this
# model family (~0.079 f1, ~2.8x SE), so a score difference here does NOT invalidate either arm --
# but a LARGE one would mean they are not running the same task, and that must be checked before any
# migration. Both arms run the SAME rung file and the same eval_size.
#
#   sbatch debug/chunked_eval_speedup/native_vs_vllm_chunked.sbatch
#SBATCH --job-name=native-vs-vllm
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/native_vs_vllm_%j.log

set -uo pipefail
source /data/prasann/node_local_env.sh
REPO=/data/prasann/repo/OLMo-core
TASK="${TASK:-outlier_amzn}"
EVAL_TASK="${EVAL_TASK:-outlier_amazon}"
# ⚠ TWO DIFFERENT ALIAS NAMESPACES. build_prefills_any/run_vllm_eval resolve run_rung_eval's
# catalog aliases (outlier_amzn -> outlier_amazon), but eval_lc_native_docchunk.py accepts only
# TASK_CFG KEYS plus a tiny alias set (nq->retrieval, contra->contradiction). Passing
# `outlier_amazon` to the native arm fast-fails in 2s with "not in [...]" -- which the timing
# harness then reported as native being 1201x FASTER. A crash is not a speedup; NATIVE_TASK is
# separate from EVAL_TASK for exactly this reason.
NATIVE_TASK="${NATIVE_TASK:-outlier}"
RUNG="${RUNG:-8192}"
MAXTEST="${MAXTEST:-500}"
RUNG_JSONL=/data/prasann/ctc_suite_staged/eval_rungs/$TASK/rung_${RUNG}.jsonl
CKPT=/data/prasann/ctc_suite/ckpts_4b_chunked/ctc-4b-${TASK}-cmix
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-${TASK}-cmix
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
OUT=/data/prasann/native_vs_vllm/$TASK
mkdir -p "$OUT"

echo "=== HOST=$(hostname) TASK=$TASK RUNG=$RUNG START=$(date '+%F %T') ==="
[ -f "$RUNG_JSONL" ] || { echo "FATAL: no rung $RUNG_JSONL"; exit 2; }
[ -f "$SERVE/model.safetensors" ] || { echo "FATAL: no serving copy $SERVE"; exit 2; }
STEP=$(ls -d "$CKPT"/step* 2>/dev/null | sort -t p -k3 -n | tail -1)
[ -n "$STEP" ] || STEP="$CKPT"
[ -d "$STEP/model_and_optim" ] || { echo "FATAL: $STEP has no model_and_optim"; exit 2; }
echo "  native ckpt: $STEP"
echo "  vllm serve : $SERVE"

# ---------- ARM A: vLLM chunked, at the best config round 4 found stable ----------
echo ""
echo "######## ARM A: vLLM chunked (cache + seqs=128) $(date '+%F %T') ########"
PREFILLS="$OUT/prefills_${RUNG}.json"
[ -f "$PREFILLS" ] || python "$REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" \
  --tokenizer "$BASE_SNAP" --task "$EVAL_TASK" --eval-data "$RUNG_JSONL" \
  --max-test-samples "$MAXTEST" --out "$PREFILLS"
export CHUNK_CACHE_IDS=1 CHUNK_MAX_NUM_SEQS=128 CHUNK_SEQ_HEADROOM=130
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$REPO/debug/ctc_vllm_validation/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task "$EVAL_TASK" \
  --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.85 --out "$OUT/resp_vllm.json"
RC_V=$?; VLLM_S=$(( $(date +%s) - T0 ))
echo "  vLLM rc=$RC_V wall=${VLLM_S}s"
[ $RC_V -eq 0 ] && python "$REPO/debug/ctc_vllm_validation/general/grade_any.py" \
  --responses "$OUT/resp_vllm.json" --eval-data "$RUNG_JSONL" --task "$EVAL_TASK" \
  --max-test-samples "$MAXTEST" --out "$OUT/grade_vllm.json"

# ---------- ARM B: native/HF chunked -- ONE mask build per example ----------
# --variant dense IS the chunked mask (see the naming note above).
echo ""
echo "######## ARM B: native docchunk (variant=dense == chunked mask) $(date '+%F %T') ########"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=1 python -u "$REPO/src/corpus_reasoning/eval/eval_lc_native_docchunk.py" \
  --variant dense --model-path "$STEP" --out "$OUT/native.json" \
  --tokenizer "$BASE_SNAP" --max-length 12288 --max-test-samples "$MAXTEST" \
  --data "$RUNG_JSONL" --task "$NATIVE_TASK" \
  --doc-start-id 248049 --doc-end-id 248050 --eos-token-id 248044 2>&1 | tail -40
RC_N=$?; NAT_S=$(( $(date +%s) - T0 ))
echo "  native rc=$RC_N wall=${NAT_S}s"

echo ""
echo "=== HEAD-TO-HEAD ($TASK rung_$RUNG, eval_size<=$MAXTEST) ==="
python3 - "$OUT" "$VLLM_S" "$NAT_S" <<'PY'
import json, os, sys
out, vs, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def load(p):
    try: return json.load(open(os.path.join(out, p)))
    except Exception: return None
gv, gn = load("grade_vllm.json"), load("native.json")
print(f"  vLLM chunked  : {vs:>6}s   metric={gv.get('metric_value') if gv else 'FAILED'}")
print(f"  native chunked: {ns:>6}s   (see native.json for its own metric block)")
if ns < 30:
    print(f"  !! native arm ran for {ns}s -- that is a CRASH, not a result. No ratio reported.")
elif ns and vs:
    print(f"  ratio: native is {vs/ns:.2f}x the vLLM wall clock "
          f"({'FASTER' if ns < vs else 'SLOWER'})")
print("  NOTE: single-GPU each. The production native harness runs 8-way DP torchrun, so a")
print("        per-GPU parity here becomes ~8x throughput in the real runner.")
PY
echo "=== DONE $(date '+%F %T') ==="
