#!/bin/bash
# ROUND 5d (2026-08-13): attribute the c1_varlen regression (134.9s vs c0's 15.5s despite
# 95/100 gen_match). Suspect ranking: (1) the paged-flash decode call (fired on 131 steps,
# including pure-decode ones), (2) the fp32 strips / per-request loop on prefill steps.
#   d0: c1 config + CHUNK_PROFILE=1        -> per-stage table (varlen_fwd_pf / varlen_fwd_dec)
#   d1: c1 config + CHUNK_VARLEN_DECODE=0  -> varlen on prefill-only steps, decode = old path
#   d2: d1 + bt8192                        -> 2 prompts/step, fewer partial-prefill fallbacks
#
#SBATCH --job-name=chunk-varlen-diag
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_varlen_diag_%j.log
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

SUMMARY=$WORK/SUMMARY_varlen_diag.tsv
echo -e "arm\tgen_s\tsteps\tvarlen\tfallback\tset_f1\tparse_rate" > "$SUMMARY"

run_arm() {
  local ARM=$1; shift
  echo ""
  echo "=== [vdiag] ARM=$ARM ENV: $* START=$(date '+%F %T') ==="
  local RESP="$WORK/resp_${ARM}.json"
  env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --max-model-len 3072 --gpu-mem-util 0.85 \
    --out "$RESP" 2>&1 | tee "$WORK/log_${ARM}.txt" \
    | grep -E "gen_seconds|generated |varlen step|patch debug state|stage profile|ms/call|^[a-z_]+ +[0-9]|Traceback|RuntimeError|Error|OOM" \
    | tail -40
  if [ ! -s "$RESP" ]; then
    echo "*** ARM $ARM FAILED ***"; tail -30 "$WORK/log_${ARM}.txt"
    echo -e "${ARM}\tFAIL\t-\t-\t-\t-\t-" >> "$SUMMARY"; return 1
  fi
  $GRADE_PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RUNG" \
    --task contradiction --max-test-samples "$EVAL_SIZE" --out "$WORK/grade_${ARM}.json"
  $GRADE_PY - "$ARM" "$WORK" "$SUMMARY" <<'PYEOF'
import json, sys
arm, work, summary = sys.argv[1:4]
r = json.load(open(f"{work}/resp_{arm}.json"))
g = json.load(open(f"{work}/grade_{arm}.json"))
pd = r.get("patch_debug") or {}
row = (f"{arm}\t{r['gen_seconds']:.1f}\t{pd.get('calls','-')}\t{pd.get('varlen','-')}\t"
       f"{pd.get('fallback','-')}\t{g['metric_value']:.4f}\t{g.get('parse_rate')}")
open(summary, "a").write(row + "\n")
print(f"RESULT {row}")
PYEOF
}

STOP="CHUNK_EXTRA_STOP=]]"
BASE="CHUNK_MAX_BATCHED_TOKENS=4096 CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18 CHUNK_VARLEN_PREFILL=1"

run_arm d0_varlen_prof  "$STOP" $BASE CHUNK_PROFILE=1
run_arm d1_pfonly       "$STOP" $BASE CHUNK_VARLEN_DECODE=0
run_arm d2_pfonly_bt8k  "$STOP" CHUNK_MAX_BATCHED_TOKENS=8192 CHUNK_MAX_NUM_SEQS=16 \
                         CHUNK_SEQ_HEADROOM=18 CHUNK_VARLEN_PREFILL=1 CHUNK_VARLEN_DECODE=0

echo ""
echo "=== [vdiag] SUMMARY ==="
column -t -s $'\t' "$SUMMARY"
echo "=== [vdiag] profile table from d0 ==="
grep -A 25 "stage profile" "$WORK/log_d0_varlen_prof.txt" | head -30
echo "=== [vdiag] DONE $(date '+%F %T') ==="
