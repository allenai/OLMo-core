#!/bin/bash
# ROUND 5 (2026-08-13): fresh look — "simple, working, FAST chunked eval", developed small.
#
# Prior state (ROUND2_FINDINGS.md §12): the validated tier-2 config
# (fp + cache + seqs64 + early stop) got the 8k rung from 2066s -> 312s (6.6x), still 4.0x
# slower than dense, and the residual is ~98% PREFILL: every prefill step pays
# create_block_mask over (max_num_batched_tokens x total_cache_tokens).
#
# This bench reproduces the problem SMALL — smoke model (0.5B GDN hybrid, 3.7G), rung 2560,
# eval_size=100 ⚠ (perf bench; scores gate corruption only, ±~0.05 SE) — and adds the two
# prefill-side arms the earlier rounds never ran in the post-early-stop regime:
#
#   * ONE-SHOT PREFILL: CHUNK_MAX_BATCHED_TOKENS >= longest prompt, so each prompt prefills
#     in a single step -> exactly ONE chunked BlockMask build per request instead of
#     ceil(len/2048), and per-step fixed CPU cost is paid once per request.
#   * RIGHT-SIZED CONCURRENCY: with early stop the decode tail is ~20 tokens, so the
#     seqs=64/headroom=66 setting tuned for the decode-bound regime mostly just inflates
#     total_cache_tokens, which the prefill mask build is linear in. seqs=16/headroom=18
#     shrinks the cache while still overlapping decode.
#
# Reference points at this rung from production (4B, eval_size=500):
#   chunked 1162.7s vs dense 46.0s — 25.3x. (32,386 steps: decode-bound, no early stop.)
#
#SBATCH --job-name=chunk-smallscale
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_smallscale_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

ACC=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
R2=/data/prasann/repo_r2/OLMo-core
mkdir -p "$R2/src" "$R2/debug/ctc_vllm_validation" /data/prasann/joblogs
rsync -a --delete /data/prasann/repo/OLMo-core/src/ "$R2/src/"
rsync -a --delete /data/prasann/repo/OLMo-core/debug/ctc_vllm_validation/ "$R2/debug/ctc_vllm_validation/"
cp "$ACC/src/corpus_reasoning/lib/vllm_chunked_patch.py" "$R2/src/corpus_reasoning/lib/vllm_chunked_patch.py"
cp "$ACC/debug/ctc_vllm_validation/run_vllm_eval.py"     "$R2/debug/ctc_vllm_validation/run_vllm_eval.py"

VDIR=$R2/debug/ctc_vllm_validation
GDIR=$VDIR/general
source /data/prasann/node_local_env.sh 2>/dev/null || true
export HOME=/data/prasann/home
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONPATH=$R2/src
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
GRADE_PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

WORK=/data/prasann/chunk_smallscale
mkdir -p "$WORK"
SERVE=/data/prasann/ctc_suite/vllm_serving/ctc-smoke-contra-3ep
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_2560.jsonl
PREFILLS_FULL=/data/prasann/ctc_suite_vllm_results_chunked/contradiction_iid/prefills_2560.json
PREFILLS=$WORK/prefills_2560_first100.json
EVAL_SIZE=100

# Truncate the production 500-row pack to the first N rows, keeping all metadata.
$GRADE_PY - "$PREFILLS_FULL" "$PREFILLS" "$EVAL_SIZE" <<'PYEOF'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
pack = json.load(open(src))
pack["rows"] = pack["rows"][:n]
pack["eval_size"] = n
json.dump(pack, open(dst, "w"))
print(f"[bench] wrote {n}-row pack -> {dst}")
PYEOF

SUMMARY=$WORK/SUMMARY.tsv
echo -e "arm\tmode\tload_s\tgen_s\tsteps\tfreepath\tfallback\tset_f1\tparse_rate" > "$SUMMARY"

run_arm() {
  local ARM=$1 MODE=$2; shift 2
  # remaining args are KEY=VAL env settings for this arm only
  echo ""
  echo "=== [smallscale] ARM=$ARM MODE=$MODE ENV: $* START=$(date '+%F %T') ==="
  local RESP="$WORK/resp_${ARM}.json"
  env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode "$MODE" --task contradiction \
    --max-new-tokens 512 --max-model-len 3072 --gpu-mem-util 0.85 \
    --out "$RESP" 2>&1 \
    | grep -E "gen_seconds|generated |chunked KV config|chunked flex tiles|extra stop|patch debug state|bumping max_model_len|LLM up|Traceback|Error|OOM|out of memory" \
    | tail -25
  if [ ! -s "$RESP" ]; then
    echo "*** ARM $ARM FAILED (no response file) ***"
    echo -e "${ARM}\t${MODE}\tFAIL\tFAIL\t-\t-\t-\t-\t-" >> "$SUMMARY"
    return 1
  fi
  $GRADE_PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RUNG" \
    --task contradiction --max-test-samples "$EVAL_SIZE" --out "$WORK/grade_${ARM}.json"
  $GRADE_PY - "$ARM" "$MODE" "$WORK" "$SUMMARY" <<'PYEOF'
import json, sys
arm, mode, work, summary = sys.argv[1:5]
r = json.load(open(f"{work}/resp_{arm}.json"))
g = json.load(open(f"{work}/grade_{arm}.json"))
pd = r.get("patch_debug") or {}
row = (f"{arm}\t{mode}\t{r.get('load_seconds', 0):.1f}\t{r['gen_seconds']:.1f}\t"
       f"{pd.get('calls','-')}\t{pd.get('freepath','-')}\t{pd.get('fallback','-')}\t"
       f"{g['metric_value']:.4f}\t{g.get('parse_rate')}")
open(summary, "a").write(row + "\n")
print(f"RESULT {row}")
PYEOF
}

STOP="CHUNK_EXTRA_STOP=]]"

# a0: dense reference, with the same early stop (the endgame dense config).
run_arm a0_dense_stop      full    "$STOP"
# a1: chunked, HISTORICAL production config — the thing being debugged. No stop, seqs8.
run_arm a1_chunk_prod      chunked
# a2: + early stop only (isolates the decode-waste term at this scale).
run_arm a2_chunk_prod_stop chunked "$STOP"
# a3: round-3 tier-2 validated config.
run_arm a3_tier2_seqs64    chunked "$STOP" CHUNK_FREE_QUERY_FASTPATH=1 CHUNK_CACHE_IDS=1 \
                                   CHUNK_MAX_NUM_SEQS=64 CHUNK_SEQ_HEADROOM=66
# a4: tier-2 with concurrency right-sized for the post-stop (prefill-bound) regime.
run_arm a4_tier2_seqs16    chunked "$STOP" CHUNK_FREE_QUERY_FASTPATH=1 CHUNK_CACHE_IDS=1 \
                                   CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18
# a5: NEW — one-shot prefill (bt >= longest prompt) on the right-sized cache.
run_arm a5_oneshot_seqs16  chunked "$STOP" CHUNK_FREE_QUERY_FASTPATH=1 CHUNK_CACHE_IDS=1 \
                                   CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18 \
                                   CHUNK_MAX_BATCHED_TOKENS=4096
# a6: NEW — one-shot prefill at tier-2 concurrency (does the bigger cache erase the win?).
run_arm a6_oneshot_seqs64  chunked "$STOP" CHUNK_FREE_QUERY_FASTPATH=1 CHUNK_CACHE_IDS=1 \
                                   CHUNK_MAX_NUM_SEQS=64 CHUNK_SEQ_HEADROOM=66 \
                                   CHUNK_MAX_BATCHED_TOKENS=4096

echo ""
echo "=== [smallscale] SUMMARY ==="
column -t -s $'\t' "$SUMMARY"

# Cross-arm generation agreement (post-']]'-truncation), a1 as the semantic reference.
$GRADE_PY - "$WORK" <<'PYEOF'
import json, glob, os, sys
work = sys.argv[1]
def trunc(s):
    i = s.find("]]")
    return s[: i + 2] if i >= 0 else s
resps = {}
for p in sorted(glob.glob(f"{work}/resp_a*.json")):
    arm = os.path.basename(p)[5:-5]
    r = json.load(open(p))
    d = r["responses"]  # dict {idx_str: text}
    resps[arm] = [trunc(d[k]) for k in sorted(d, key=int)]
ref = resps.get("a1_chunk_prod")
if ref:
    print("\ngen_match vs a1_chunk_prod (post-']]' truncation):")
    for arm, gens in resps.items():
        n = sum(a == b for a, b in zip(gens, ref))
        print(f"  {arm:24s} {n}/{len(ref)}")
PYEOF
echo "=== [smallscale] DONE $(date '+%F %T') ==="
