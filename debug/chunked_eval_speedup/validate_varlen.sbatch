#!/bin/bash
# ROUND 5f (2026-08-13): PRODUCTION-SCALE validation of CHUNK_VARLEN_PREFILL on the 4B
# checkpoints, eval_size=500. Small-scale (smoke model, rung 2560, min over 3 reps):
#   simple config 15.4s -> varlen 14.3s -> varlen@seqs64 13.0s (dense 2.7s).
# Gate: set_f1 within 0.018 of known-good AND parse_rate == 1.0 (same as validate_simple).
# Known-good: rung 2560 f1 0.8613 (prod chunked 1162.7s); rung 8192 f1 0.8039 (2066.1s).
# Simple-config reference from job 3443611: 95.8s @2560, 223.5s @8192.
#
#SBATCH --job-name=chunk-valvarlen
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_valvarlen_%j.log
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

WORK=/data/prasann/chunk_valvarlen
mkdir -p "$WORK"
SUMMARY=$WORK/SUMMARY.tsv
echo -e "arm\trung\tgen_s\tprod_s\tspeedup\tsimple_s\tvs_simple\tsteps\tvarlen\tfallback\tset_f1\tknown_f1\tdelta\tparse\tverdict" > "$SUMMARY"

run_rung() {
  local ARM=$1 SERVE=$2 PREFILLS=$3 RUNG=$4 KNOWN_S=$5 KNOWN_F1=$6 SIMPLE_S=$7; shift 7
  echo ""
  echo "=== [valvarlen] ARM=$ARM ENV: $* START=$(date '+%F %T') ==="
  local RESP="$WORK/resp_${ARM}.json"
  env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.90 \
    --out "$RESP" 2>&1 | tee "$WORK/log_${ARM}.txt" \
    | grep -E "gen_seconds|generated |chunked KV config|varlen step|extra stop|patch debug state|bumping max_model_len|LLM up|Traceback|RuntimeError|Error|OOM|out of memory" \
    | tail -25
  if [ ! -s "$RESP" ]; then
    echo "*** ARM $ARM FAILED ***"; tail -30 "$WORK/log_${ARM}.txt"
    echo -e "${ARM}\t-\tFAIL\t${KNOWN_S}\t-\t${SIMPLE_S}\t-\t-\t-\t-\t-\t${KNOWN_F1}\t-\t-\tFAIL" >> "$SUMMARY"
    return 1
  fi
  $GRADE_PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RUNG" \
    --task contradiction --max-test-samples 500 --out "$WORK/grade_${ARM}.json"
  $GRADE_PY - "$ARM" "$WORK" "$SUMMARY" "$KNOWN_S" "$KNOWN_F1" "$SIMPLE_S" <<'PYEOF'
import json, sys
arm, work, summary, known_s, known_f1, simple_s = sys.argv[1:7]
known_s, known_f1, simple_s = float(known_s), float(known_f1), float(simple_s)
r = json.load(open(f"{work}/resp_{arm}.json"))
g = json.load(open(f"{work}/grade_{arm}.json"))
pd = r.get("patch_debug") or {}
s, f1, pr = r["gen_seconds"], g["metric_value"], g.get("parse_rate")
ok = abs(f1 - known_f1) <= 0.018 and pr == 1.0
rung = arm.split("_")[-1]
row = (f"{arm}\t{rung}\t{s:.1f}\t{known_s:.1f}\t{known_s/s:.2f}x\t{simple_s:.1f}\t"
       f"{simple_s/s:.2f}x\t{pd.get('calls','-')}\t{pd.get('varlen','-')}\t"
       f"{pd.get('fallback','-')}\t{f1:.4f}\t{known_f1:.4f}\t{f1-known_f1:+.4f}\t{pr}\t"
       f"{'REPRODUCES' if ok else '*** FAILS GATE ***'}")
open(summary, "a").write(row + "\n")
print(f"RESULT {row}")
PYEOF
}

STOP="CHUNK_EXTRA_STOP=]]"
S2560=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction_iid-cmix
S8192=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
P2560=/data/prasann/ctc_suite_vllm_results_chunked/contradiction_iid/prefills_2560.json
P8192=/data/prasann/ctc_suite_vllm_results_chunked/contradiction_iid/prefills_8192.json
R2560=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_2560.jsonl
R8192=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl

# Matched to the validated simple config (isolates the varlen delta):
run_rung vl16_2560 "$S2560" "$P2560" "$R2560" 1162.7 0.8613 95.8 \
  "$STOP" CHUNK_MAX_BATCHED_TOKENS=4096 CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18 \
  CHUNK_VARLEN_PREFILL=1
run_rung vl16_8192 "$S8192" "$P8192" "$R8192" 2066.1 0.8039 223.5 \
  "$STOP" CHUNK_MAX_BATCHED_TOKENS=9216 CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18 \
  CHUNK_VARLEN_PREFILL=1
# Bonus: higher concurrency at 8k (best small-scale arm; no mask cost to pay for the cache).
run_rung vl64_8192 "$S8192" "$P8192" "$R8192" 2066.1 0.8039 223.5 \
  "$STOP" CHUNK_MAX_BATCHED_TOKENS=9216 CHUNK_MAX_NUM_SEQS=64 CHUNK_SEQ_HEADROOM=66 \
  CHUNK_VARLEN_PREFILL=1

echo ""
echo "=== [valvarlen] SUMMARY ==="
column -t -s $'\t' "$SUMMARY"
echo "=== [valvarlen] DONE $(date '+%F %T') ==="
