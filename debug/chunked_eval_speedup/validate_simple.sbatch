#!/bin/bash
# ROUND 5c (2026-08-13): PRODUCTION-SCALE validation of the "simple fast chunked eval" config
# found by smallscale_bench/smallscale_isolate (jobs 3443501/3443561):
#
#     CHUNK_EXTRA_STOP=']]'            (contradiction / cot_mode=none only)
#     CHUNK_MAX_BATCHED_TOKENS >= longest prompt   -> ONE BlockMask build per request
#     CHUNK_MAX_NUM_SEQS=16 / CHUNK_SEQ_HEADROOM=18
#     NO CHUNK_FREE_QUERY_FASTPATH, NO CHUNK_CACHE_IDS (individually innocent, pathological
#     in combination at small scale; the winner needs neither)
#
# Small-scale result (smoke model, rung 2560, eval_size=100): 286.6s -> 15.4s (18.6x), dense 2.7s.
# This job re-runs the config on the 4B production checkpoints at eval_size=500 against the
# known-good numbers:
#     rung 2560: chunked 1162.7s  set_f1 0.8613  (dense 46.0s / 0.9895)
#     rung 8192: chunked 2066.1s  set_f1 0.8039  (dense 116.4s / 0.9760; round-3 best 311.9s)
# Gate: set_f1 within the 500-example binomial SE (0.018) of known-good AND parse_rate == 1.0.
#
#SBATCH --job-name=chunk-valsimple
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_valsimple_%j.log
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

WORK=/data/prasann/chunk_valsimple
mkdir -p "$WORK"
SUMMARY=$WORK/SUMMARY.tsv
echo -e "arm\trung\tgen_s\tknown_good_s\tspeedup\tsteps\tset_f1\tknown_f1\tdelta\tparse_rate\tverdict" > "$SUMMARY"

run_rung() {
  local RUNGTOK=$1 SERVE=$2 PREFILLS=$3 RUNG=$4 BT=$5 KNOWN_S=$6 KNOWN_F1=$7
  local ARM="v_${RUNGTOK}"
  echo ""
  echo "=== [valsimple] RUNG=$RUNGTOK BT=$BT START=$(date '+%F %T') ==="
  local RESP="$WORK/resp_${ARM}.json"
  env CHUNK_EXTRA_STOP=']]' CHUNK_MAX_BATCHED_TOKENS="$BT" \
      CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18 \
    $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.90 \
    --out "$RESP" 2>&1 | tee "$WORK/log_${ARM}.txt" \
    | grep -E "gen_seconds|generated |chunked KV config|chunked flex tiles|extra stop|patch debug state|bumping max_model_len|LLM up|Traceback|Error|OOM|out of memory" \
    | tail -25
  if [ ! -s "$RESP" ]; then
    echo "*** RUNG $RUNGTOK FAILED (no response file) ***"
    echo -e "${ARM}\t${RUNGTOK}\tFAIL\t${KNOWN_S}\t-\t-\t-\t${KNOWN_F1}\t-\t-\tFAIL" >> "$SUMMARY"
    return 1
  fi
  $GRADE_PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RUNG" \
    --task contradiction --max-test-samples 500 --out "$WORK/grade_${ARM}.json"
  $GRADE_PY - "$ARM" "$RUNGTOK" "$WORK" "$SUMMARY" "$KNOWN_S" "$KNOWN_F1" <<'PYEOF'
import json, sys
arm, rung, work, summary, known_s, known_f1 = sys.argv[1:7]
known_s, known_f1 = float(known_s), float(known_f1)
r = json.load(open(f"{work}/resp_{arm}.json"))
g = json.load(open(f"{work}/grade_{arm}.json"))
pd = r.get("patch_debug") or {}
s, f1, pr = r["gen_seconds"], g["metric_value"], g.get("parse_rate")
ok = abs(f1 - known_f1) <= 0.018 and pr == 1.0
verdict = "REPRODUCES" if ok else "*** FAILS GATE ***"
row = (f"{arm}\t{rung}\t{s:.1f}\t{known_s:.1f}\t{known_s/s:.2f}x\t{pd.get('calls','-')}\t"
       f"{f1:.4f}\t{known_f1:.4f}\t{f1-known_f1:+.4f}\t{pr}\t{verdict}")
open(summary, "a").write(row + "\n")
print(f"RESULT {row}")
PYEOF
}

# rung 2560: prompt max ~3113 -> BT 4096 covers one-shot.
run_rung 2560 /data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction_iid-cmix \
  /data/prasann/ctc_suite_vllm_results_chunked/contradiction_iid/prefills_2560.json \
  /data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_2560.jsonl \
  4096 1162.7 0.8613

# rung 8192: prompt max ~8875 -> BT 9216. Same ckpt+pack as fullrung_validate.sbatch.
run_rung 8192 /data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix \
  /data/prasann/ctc_suite_vllm_results_chunked/contradiction_iid/prefills_8192.json \
  /data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl \
  9216 2066.1 0.8039

echo ""
echo "=== [valsimple] SUMMARY ==="
column -t -s $'\t' "$SUMMARY"
echo "=== [valsimple] DONE $(date '+%F %T') ==="
