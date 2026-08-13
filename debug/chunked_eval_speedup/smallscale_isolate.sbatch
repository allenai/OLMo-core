#!/bin/bash
# ROUND 5b (2026-08-13): knob ISOLATION after smallscale_bench.sbatch (job 3443501) showed the
# round-3 tier-2 recipe REGRESSING at small scale:
#     a2 stop-only (seqs8, bt2048, no fp/cache)     21.2s   316 steps
#     a4 stop + fp + cache + seqs16 (same cache!)   87.0s   181 steps   <- 4x SLOWER, fewer steps
#     a5 = a4 + one-shot prefill (bt4096)           41.4s   169 steps
# Something in {fp, cache, seqs16} costs ~300-500ms per fallback build. Each arm below adds ONE
# knob to the a2 baseline (stop, seqs8, headroom10, bt2048). b6 profiles the fp arm.
# Suspect: _query_positions_all_free() calls query_start_loc.cpu() -> per-step GPU sync.
#
#SBATCH --job-name=chunk-isolate
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_isolate_%j.log
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

SUMMARY=$WORK/SUMMARY_isolate.tsv
echo -e "arm\tmode\tload_s\tgen_s\tsteps\tfreepath\tfallback\tset_f1\tparse_rate" > "$SUMMARY"

run_arm() {
  local ARM=$1 MODE=$2; shift 2
  echo ""
  echo "=== [isolate] ARM=$ARM MODE=$MODE ENV: $* START=$(date '+%F %T') ==="
  local RESP="$WORK/resp_${ARM}.json"
  env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode "$MODE" --task contradiction \
    --max-new-tokens 512 --max-model-len 3072 --gpu-mem-util 0.85 \
    --out "$RESP" 2>&1 | tee "$WORK/log_${ARM}.txt" \
    | grep -E "gen_seconds|generated |chunked KV config|chunked flex tiles|extra stop|patch debug state|stage profile|ms/call|^[a-z_]+\.[pd] |bumping max_model_len|LLM up|Traceback|Error|OOM|out of memory" \
    | tail -60
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

# b0: a2 exact re-run — run-to-run variance floor.
run_arm b0_stop_rerun   chunked "$STOP"
# b1: + freepath ONLY.
run_arm b1_fp           chunked "$STOP" CHUNK_FREE_QUERY_FASTPATH=1
# b2: + chunk-id cache ONLY.
run_arm b2_cache        chunked "$STOP" CHUNK_CACHE_IDS=1
# b3: + seqs16/headroom18 ONLY.
run_arm b3_seqs16       chunked "$STOP" CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18
# b4: + one-shot prefill ONLY (bt4096 at prod seqs8/headroom10).
run_arm b4_oneshot      chunked "$STOP" CHUNK_MAX_BATCHED_TOKENS=4096
# b5: one-shot + seqs16, NO fp, NO cache — candidate simple-fast config.
run_arm b5_oneshot16    chunked "$STOP" CHUNK_MAX_BATCHED_TOKENS=4096 \
                                 CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18
# b6: profile the fp arm — where do its ~370ms/step go?
run_arm b6_fp_profile   chunked "$STOP" CHUNK_FREE_QUERY_FASTPATH=1 \
                                 CHUNK_PROFILE=1 CHUNK_PROFILE_SPLIT=1

echo ""
echo "=== [isolate] SUMMARY ==="
column -t -s $'\t' "$SUMMARY"
echo "=== [isolate] DONE $(date '+%F %T') ==="
