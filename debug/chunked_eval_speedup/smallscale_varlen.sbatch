#!/bin/bash
# ROUND 5c (2026-08-13): first GPU run of CHUNK_VARLEN_PREFILL — the varlen rewrite of
# chunked prefill (flash varlen block-diagonal A + fp32 strips B/C + merge_attn_states,
# decode via paged flash). Exactness of the decomposition is proven in float64 by
# test_varlen_decomposition.py; THIS run checks the vLLM integration end-to-end.
#
# Baselines on this bench (smoke model, rung 2560, eval_size=100 ⚠ perf bench):
#   dense+stop 2.7s | b5 stop+oneshot+seqs16 15.4s | historical chunked 286.6s
# Gate: parse_rate 1.0, f1 == b5's, gen_match vs b5 high (numerics differ: flash vs flex).
#
#SBATCH --job-name=chunk-varlen
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_varlen_%j.log
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
PREFILLS=$WORK/prefills_2560_first100.json
EVAL_SIZE=100

SUMMARY=$WORK/SUMMARY_varlen.tsv
echo -e "arm\tmode\tload_s\tgen_s\tsteps\tvarlen\tfallback\tset_f1\tparse_rate" > "$SUMMARY"

run_arm() {
  local ARM=$1 MODE=$2; shift 2
  echo ""
  echo "=== [varlen] ARM=$ARM MODE=$MODE ENV: $* START=$(date '+%F %T') ==="
  local RESP="$WORK/resp_${ARM}.json"
  env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode "$MODE" --task contradiction \
    --max-new-tokens 512 --max-model-len 3072 --gpu-mem-util 0.85 \
    --out "$RESP" 2>&1 | tee "$WORK/log_${ARM}.txt" \
    | grep -E "gen_seconds|generated |chunked KV config|varlen step|extra stop|patch debug state|Traceback|RuntimeError|Error|OOM|out of memory" \
    | tail -30
  if [ ! -s "$RESP" ]; then
    echo "*** ARM $ARM FAILED (no response file) — see $WORK/log_${ARM}.txt ***"
    tail -40 "$WORK/log_${ARM}.txt"
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
       f"{pd.get('calls','-')}\t{pd.get('varlen','-')}\t{pd.get('fallback','-')}\t"
       f"{g['metric_value']:.4f}\t{g.get('parse_rate')}")
open(summary, "a").write(row + "\n")
print(f"RESULT {row}")
PYEOF
}

STOP="CHUNK_EXTRA_STOP=]]"
BASE="CHUNK_MAX_BATCHED_TOKENS=4096 CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18"

# c0: the validated simple config (b5) — control.
run_arm c0_simple  chunked "$STOP" $BASE
# c1: + the varlen rewrite.
run_arm c1_varlen  chunked "$STOP" $BASE CHUNK_VARLEN_PREFILL=1
# c2: varlen at higher concurrency — with no per-step mask cost, does seqs64 pay now?
run_arm c2_varlen64 chunked "$STOP" CHUNK_MAX_BATCHED_TOKENS=4096 CHUNK_MAX_NUM_SEQS=64 \
                             CHUNK_SEQ_HEADROOM=66 CHUNK_VARLEN_PREFILL=1

echo ""
echo "=== [varlen] SUMMARY ==="
column -t -s $'\t' "$SUMMARY"

# gen_match: c1/c2 vs c0 (post-']]' truncation).
$GRADE_PY - "$WORK" <<'PYEOF'
import json, sys
work = sys.argv[1]
def trunc(s):
    i = s.find("]]")
    return s[: i + 2] if i >= 0 else s
def load(arm):
    d = json.load(open(f"{work}/resp_{arm}.json"))["responses"]
    return [trunc(d[k]) for k in sorted(d, key=int)]
try:
    ref = load("c0_simple")
    for arm in ("c1_varlen", "c2_varlen64"):
        try:
            gens = load(arm)
            n = sum(a == b for a, b in zip(gens, ref))
            print(f"gen_match {arm} vs c0_simple: {n}/{len(ref)}")
        except FileNotFoundError:
            print(f"gen_match {arm}: MISSING")
except FileNotFoundError:
    print("no c0 responses; skipping gen_match")
PYEOF
echo "=== [varlen] DONE $(date '+%F %T') ==="
