#!/bin/bash
# Round 3: does the validated 8k speedup config carry to 16k and 32k?
#
# ROUND2_FINDINGS.md §4 ships `CHUNK_CACHE_IDS=1 + CHUNK_MAX_NUM_SEQS=32 + CHUNK_SEQ_HEADROOM=34`
# as 1.78x, but explicitly scopes it to the 8k rung and names 32k as "the setting most likely to
# need lowering, and it has not been measured above 8k". The mechanism is concrete: the fallback
# mask build is O(num_actual_tokens x total_cache_tokens), and total_cache_tokens scales with
# max_model_len x CHUNK_SEQ_HEADROOM -- so headroom 10 -> 34 is a 3.4x tax on every step, and 8k ->
# 32k is another 4x on top. Whether the step savings still outrun that is an empirical question.
#
# So gpu_eval_task_chunked.sh currently gates the concurrency knobs at rung <= 8192. This sweep is
# what lifts (or confirms) that gate.
#
# TWO METHOD RULES INHERITED FROM ROUND 2, both learned the hard way:
#   * FULL eval_size, not a sample. Two of three knobs in round 2 had an eval_size-dependent SIGN:
#     bt=512 measured 1.09x FASTER at eval_size=48 and 0.93x SLOWER at 500; seqs=32 was 1.5x slower
#     at 24 and 1.19x faster at 48. A 24-example sweep would have shipped the wrong config.
#   * Reuse the SAME prefills pack across arms so wall clock and f1 are directly comparable and no
#     arm pays a different tokenization cost.
#
# Correctness gate per arm: parse_rate == 1.000 AND set_f1 within the binomial SE of the baseline
# arm at this rung. A config that is fast and wrong is the failure mode that matters here -- the
# chunked path can silently degrade to dense-causal numbers (ROUND2 §5a), so speed alone never
# certifies an arm.
#
#   sbatch debug/chunked_eval_speedup/rung_scaling_sweep.sbatch
#SBATCH --job-name=chunk-rungscale
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/chunk_rungscale_%j.log

set -uo pipefail
source /data/prasann/node_local_env.sh

LOCAL_REPO=/data/prasann/repo/OLMo-core
# TASK names the eval_rungs/<dir> and the serving copy; EVAL_TASK is the catalog key passed to
# --task. They are NOT the same string here: `contradiction_iid` is a data-source label with no
# entry in build_prefills_any's catalog, and passing it lands in the "not in [...]" fast-fail
# (which is how the first submission of this sweep, job 3440570, died on all 8 arms).
TASK=contradiction_iid
EVAL_TASK=contradiction
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
RUNGDIR=/data/prasann/ctc_suite_staged/eval_rungs/$TASK
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
OUT=/data/prasann/chunk_rungscale
mkdir -p "$OUT"

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$SERVE/model.safetensors" ] || { echo "FATAL: no serving copy at $SERVE"; exit 1; }

# rung : arm-label : CHUNK_CACHE_IDS : CHUNK_MAX_NUM_SEQS : CHUNK_SEQ_HEADROOM
# seqs=8/headroom=10 is the historical production default and is the baseline at each rung.
# seqs=16/headroom=18 is included because if 32 overshoots at long rungs, the turnover is likely
# BELOW it -- round 2 saw 64 already regress at 8k, so the curve peaks somewhere and moves left as
# L grows. Measuring only {8, 32} would find "32 is worse" and miss that 16 is better than both.
ARMS=(
  "16384:base:0:8:10"
  "16384:cache_only:1:8:10"
  "16384:cache_seqs16:1:16:18"
  "16384:cache_seqs32:1:32:34"
  "32768:base:0:8:10"
  "32768:cache_only:1:8:10"
  "32768:cache_seqs16:1:16:18"
  "32768:cache_seqs32:1:32:34"
)

for spec in "${ARMS[@]}"; do
  IFS=: read -r RUNG ARM CACHE SEQS HEAD <<< "$spec"
  RUNG_JSONL="$RUNGDIR/rung_${RUNG}.jsonl"
  [ -f "$RUNG_JSONL" ] || { echo "!!! rung=$RUNG missing $RUNG_JSONL -- skipping"; continue; }

  # Build the prefills pack ONCE per rung and reuse it for every arm at that rung.
  PREFILLS="$OUT/prefills_${RUNG}.json"
  if [ ! -f "$PREFILLS" ]; then
    python "$LOCAL_REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" \
      --tokenizer "$BASE_SNAP" --task "$EVAL_TASK" --eval-data "$RUNG_JSONL" \
      --max-test-samples 100000 --out "$PREFILLS" || { echo "!!! prefill build failed rung=$RUNG"; continue; }
  fi

  echo ""
  echo "######## rung=$RUNG arm=$ARM cache=$CACHE seqs=$SEQS headroom=$HEAD $(date '+%F %T') ########"
  RESP="$OUT/responses_${RUNG}_${ARM}.json"
  GRADE="$OUT/grade_${RUNG}_${ARM}.json"

  if [ "$CACHE" = "1" ]; then export CHUNK_CACHE_IDS=1; else unset CHUNK_CACHE_IDS; fi
  export CHUNK_MAX_NUM_SEQS=$SEQS
  export CHUNK_SEQ_HEADROOM=$HEAD
  # CHUNK_MAX_BATCHED_TOKENS is deliberately NOT set: round 2 §3 measured lowering it as a 1.39x
  # loss at seqs=32 and a 0.93x loss at production concurrency. It stays at its 2048 default.

  T0=$(date +%s)
  PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$LOCAL_REPO/debug/ctc_vllm_validation/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task "$EVAL_TASK" \
    --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.6 --out "$RESP"
  RC=$?; T1=$(date +%s); GENS=$((T1-T0))
  if [ $RC -ne 0 ]; then echo "!!! rung=$RUNG arm=$ARM run_vllm_eval FAILED rc=$RC (${GENS}s)"; continue; fi

  python "$LOCAL_REPO/debug/ctc_vllm_validation/general/grade_any.py" \
    --responses "$RESP" --eval-data "$RUNG_JSONL" --task "$EVAL_TASK" \
    --max-test-samples 100000 --out "$GRADE" || { echo "!!! grade failed rung=$RUNG arm=$ARM"; continue; }

  python3 - "$GRADE" "$RUNG" "$ARM" "$GENS" "$OUT/RESULTS.tsv" <<'PY'
import json, os, sys
grade, rung, arm, gens, tsv = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
d = json.load(open(grade))
mv, pr = d.get("metric_value"), d.get("parse_rate")
sz = d.get("eval_size", d.get("n"))
pd_ = d.get("patch_debug", {}) or {}
row = f"{rung}\t{arm}\t{gens}\t{mv}\t{pr}\t{sz}\tapplied={pd_.get('applied')}\tdirect={pd_.get('direct')}\tfallback={pd_.get('fallback')}"
print("  RESULT: " + row.replace("\t", "  "))
new = not os.path.exists(tsv)
with open(tsv, "a") as f:
    if new:
        f.write("rung\tarm\tgen_seconds\tmetric_value\tparse_rate\teval_size\tapplied\tdirect\tfallback\n")
    f.write(row + "\n")
PY
  rm -f "$RESP"
done

echo ""
echo "=== RESULTS ==="
cat "$OUT/RESULTS.tsv" 2>/dev/null
ACC=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/chunked_eval_speedup
cp "$OUT/RESULTS.tsv" "$ACC/ROUND3_RUNG_SCALING.tsv" 2>/dev/null
echo "=== DONE $(date '+%F %T') ==="
