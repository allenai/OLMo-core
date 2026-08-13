#!/bin/bash
# Round 4: turn OFF the per-decode-step mask rebuild, then see how far concurrency can go.
#
# THE INSIGHT (from the HF eval path in corpus-reasoning/scripts/eval/evaluate.py:621-637).
# That evaluator applies the chunked block mask to the PREFILL ONLY and then decodes with
# `attention_mask=None` -- "after prefill, both paths drop back to plain causal for the
# autoregressive decode (each new token attends to all real KV)". That is not a shortcut, it is the
# semantics: the chunked rule is
#     allowed = causal & not_pad & (context_ok | q_free | kv_free)
# so a FREE query short-circuits to `causal & not_pad`, which IS vLLM's default paged-causal mask.
# Every generated token is FREE (answer text, not a document).
#
# vLLM already ships this as CHUNK_FREE_QUERY_FASTPATH -- and it is OFF BY DEFAULT. Its own comment
# says it covers 93.7% of steps in the 8k production run, is checked per step against real chunk
# ids, and fails closed if any context query appears. So in production we rebuild, at
# O(num_actual_tokens x total_cache_tokens) each time, a mask vLLM had already built for free, on
# ~94% of steps.
#
# WHY ROUND 2 UNDER-RATED THIS. §1 attributed only 12.1% of runtime to the mask apparatus and moved
# on. But the 17.8x dense/chunked gap is almost entirely a CONCURRENCY gap:
#     32573 builder calls  ~=  (500 examples / 8 resident) x 521 steps
#     dense runs the same 500 at max_num_seqs=1024  ->  ~521 steps total
# and §2b found concurrency does not pay because the fallback mask build is O(N x N.L) -- QUADRATIC
# in resident sequences. That quadratic term IS the per-step rebuild. Remove it on the 94% of steps
# where it is provably redundant and concurrency should stop paying for itself, which is what makes
# seqs >> 32 worth testing for the first time. Round 2 measured the concurrency knob with the tax
# still on; this measures it with the tax off.
#
# METHOD, inherited from round 2 (both rules learned the hard way):
#   * FULL eval_size=500. Two of three knobs in round 2 had an eval_size-dependent SIGN.
#   * Reuse ONE prefills pack across all arms so wall clock and score are directly comparable.
# Reference to beat, same rung/checkpoint/pack: production baseline 2066.1s at set_f1 0.8039,
# parse_rate 1.000; the shipped cache+seqs32 config 1163.9s at 0.7999.
#
# CORRECTNESS IS THE GATE, NOT SPEED. The fast path claims exactness, so the arms that use it must
# reproduce the baseline within the 500-example binomial SE (0.018) AND hold parse_rate 1.000. A
# chunked run that silently degrades to dense-causal numbers is THE failure mode here (ROUND2 §5a),
# and it would look like a speedup with slightly better f1 -- so f1 moving UP is not reassuring,
# it is the thing to be suspicious of. `freepath` counts are logged per arm to prove the path fired.
#
# ⚠ RERUN NOTE: the first execution of this sweep measured NOTHING. The node-local copies of
# vllm_chunked_patch.py on cubbins and mooney predated the freepath feature (grep -c
# CHUNK_FREE_QUERY_FASTPATH -> 0), so the env var toggled code that was not present and all four
# arms reported freepath=0 -- fp_seqs32 came out byte-identical to the control. The module is now
# synced from the login repo. TARGET IS DENSE PARITY: dense runs this rung in 116.4s, i.e. 17.8x.
#
#   sbatch debug/chunked_eval_speedup/freepath_concurrency_sweep.sbatch
#SBATCH --job-name=chunk-freepath
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/chunk_freepath_%j.log

set -uo pipefail
source /data/prasann/node_local_env.sh

LOCAL_REPO=/data/prasann/repo/OLMo-core
TASK=contradiction_iid
EVAL_TASK=contradiction
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
RUNG=8192
RUNG_JSONL=/data/prasann/ctc_suite_staged/eval_rungs/$TASK/rung_${RUNG}.jsonl
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
OUT=/data/prasann/chunk_freepath
mkdir -p "$OUT"

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$SERVE/model.safetensors" ] || { echo "FATAL: no serving copy at $SERVE"; exit 1; }
[ -f "$RUNG_JSONL" ] || { echo "FATAL: no rung file $RUNG_JSONL"; exit 1; }

# Start at 8k because that is where a TRUSTED reference exists (2066.1s / 0.8039, full 500-example
# validation). Proving the fast path against a known-good number matters more than covering more
# rungs -- 16k/32k follow only once exactness is established here.
PREFILLS="$OUT/prefills_${RUNG}.json"
if [ ! -f "$PREFILLS" ]; then
  python "$LOCAL_REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" \
    --tokenizer "$BASE_SNAP" --task "$EVAL_TASK" --eval-data "$RUNG_JSONL" \
    --max-test-samples 100000 --out "$PREFILLS" || { echo "FATAL: prefill build failed"; exit 1; }
fi

# arm : CHUNK_CACHE_IDS : CHUNK_FREE_QUERY_FASTPATH : CHUNK_MAX_NUM_SEQS : CHUNK_SEQ_HEADROOM
# `shipped` reproduces what production runs today (the round-2 winner) as an in-run control, so the
# comparison does not depend on trusting a number measured on another day.
# Headroom must track concurrency or the KV allocation cannot hold the sequences.
ARMS=(
  "fp_seqs192:1:1:192:196"
  "fp_seqs256:1:1:256:260"
  "fp_seqs320:1:1:320:326"
)

for spec in "${ARMS[@]}"; do
  IFS=: read -r ARM CACHE FP SEQS HEAD <<< "$spec"
  echo ""
  echo "######## arm=$ARM cache=$CACHE freepath=$FP seqs=$SEQS headroom=$HEAD $(date '+%F %T') ########"
  RESP="$OUT/responses_${ARM}.json"
  GRADE="$OUT/grade_${ARM}.json"

  export CHUNK_CACHE_IDS=$CACHE
  if [ "$FP" = "1" ]; then export CHUNK_FREE_QUERY_FASTPATH=1; else unset CHUNK_FREE_QUERY_FASTPATH; fi
  export CHUNK_MAX_NUM_SEQS=$SEQS
  export CHUNK_SEQ_HEADROOM=$HEAD
  # CHUNK_MAX_BATCHED_TOKENS stays at its 2048 default: round 2 §3 measured lowering it as a 1.39x
  # loss at seqs=32 and a 0.93x loss at production concurrency.

  T0=$(date +%s)
  PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$LOCAL_REPO/debug/ctc_vllm_validation/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task "$EVAL_TASK" \
    --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.6 --out "$RESP"
  RC=$?; T1=$(date +%s); GENS=$((T1-T0))
  if [ $RC -ne 0 ]; then echo "!!! arm=$ARM FAILED rc=$RC after ${GENS}s"; continue; fi

  python "$LOCAL_REPO/debug/ctc_vllm_validation/general/grade_any.py" \
    --responses "$RESP" --eval-data "$RUNG_JSONL" --task "$EVAL_TASK" \
    --max-test-samples 100000 --out "$GRADE" || { echo "!!! arm=$ARM grade failed"; continue; }

  python3 - "$GRADE" "$ARM" "$GENS" "$OUT/RESULTS.tsv" <<'PY'
import json, os, sys
grade, arm, gens, tsv = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = json.load(open(grade))
pd_ = d.get("patch_debug", {}) or {}
BASE_S, BASE_F1, SE = 2066.1, 0.8039, 0.018
mv, pr = d.get("metric_value"), d.get("parse_rate")
# freepath > 0 proves the optimisation actually fired; an arm that claims the flag but shows
# freepath=0 measured nothing and must not be read as a speedup.
fp = pd_.get("freepath", 0)
ok = (pr == 1.0) and (mv is not None) and abs(mv - BASE_F1) <= SE
row = (f"{arm}\t{gens}\t{BASE_S/gens:.2f}x\t{mv}\t{(mv or 0)-BASE_F1:+.4f}\t{pr}\t"
       f"{d.get('eval_size', d.get('n'))}\tfreepath={fp}\tfallback={pd_.get('fallback')}\t"
       f"{'PASS' if ok else 'FAIL-CORRECTNESS'}")
print("  RESULT: " + row.replace("\t", "  "))
new = not os.path.exists(tsv)
with open(tsv, "a") as f:
    if new:
        f.write("arm\tgen_seconds\tspeedup_vs_2066s\tmetric\tdelta_f1\tparse_rate\teval_size\t"
                "freepath\tfallback\tverdict\n")
    f.write(row + "\n")
PY
  rm -f "$RESP"
done

echo ""
echo "=== RESULTS (gate: |delta_f1| <= 0.018 AND parse_rate 1.000 AND freepath > 0) ==="
cat "$OUT/RESULTS.tsv" 2>/dev/null
cp "$OUT/RESULTS.tsv" \
  /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/chunked_eval_speedup/ROUND4_FREEPATH.tsv 2>/dev/null
echo "=== DONE $(date '+%F %T') ==="
