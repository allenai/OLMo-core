#!/bin/bash
# Extend the CHUNK_VARLEN_PREFILL validation to 16k/32k -- the rungs the speedup is actually needed at.
#
# WHAT IS ALREADY ESTABLISHED (other session, research/landmark/chunked_eval_speedup/validate_varlen.sbatch):
#   rung 2560: 1162.7s -> 59.8s   19.45x   set_f1 0.8627 vs 0.8613 known-good (+0.0014)  parse 1.0
#   rung 8192: 2066.1s -> 133.7s  15.45x   set_f1 0.8019 vs 0.8039 (-0.0020)             parse 1.0
# and that seqs=64 is WORSE than seqs=16 at 8k (9.86x vs 15.45x) because its varlen path mostly
# falls back: 31 varlen / 455 fallback steps, against 719/184 at seqs=16.
#
# WHY 16k/32k CANNOT BE ASSUMED FROM THAT. The fallback mask build costs
# O(num_actual_tokens x total_cache_tokens), so the varlen-vs-fallback split is rung-dependent by
# construction -- and the seqs=64 result at 8k is a direct demonstration that a config which looks
# good at one setting can collapse into fallback at another. My own rung-gate sweep found the same
# shape from the other direction: concurrency gave 1.32x at 16k but was a REGRESSION at seqs=128,
# where 128 is optimal at 8k. Two independent knobs have now shown rung-dependent sign in this
# codebase; extrapolating a 15x speedup from 8k to 32k without measuring would be the third mistake.
#
# SELF-CONTAINED BASELINE. Rather than compare against a historical wall-clock from a different
# task/config, each rung runs BOTH arms back to back in this job: the current production config and
# the varlen config, on the identical prefills. That makes the speedup a controlled measurement and
# makes the f1 comparison an arm-vs-arm parity check rather than a comparison against a number
# measured under conditions I would have to assume still hold.
#
# ⚠ RUNS FROM A SEPARATE REPO COPY. `vllm_chunked_patch.py` is read live by the chunked eval jobs
# currently in flight (ev4b-outlierM-cmix is 2h into its 32k rung; the OOLONG chunked arm is
# mid-ladder). Editing it in place would change their generations underneath them. This job copies
# the tree to repo_r3 and points PYTHONPATH there, exactly as the other session did.
#
#   sbatch research/landmark/chunked_eval_speedup/validate_varlen_16k32k.sbatch
# ⚠ MUST be cubbins: the chunked serving copy (ctc-4b-contradiction-cmix), the rung files and the
# other session's validated repo_r2 tree are all node-local there and nowhere else. The first
# attempt (3443866) had no --nodelist, landed elsewhere, and died in 3s on the missing serve dir.
#SBATCH --nodelist=cubbins
#SBATCH --job-name=chunk-val1632
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/chunk_val1632_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
ACC=$REPO_ROOT
R3=/data/prasann/repo_r3/OLMo-core
mkdir -p "$R3/src" "$R3/research/landmark/ctc_vllm_validation" /data/prasann/joblogs
rsync -a --delete $REPO_ROOT/src/ "$R3/src/"
rsync -a --delete $REPO_ROOT/research/landmark/ctc_vllm_validation/ "$R3/research/landmark/ctc_vllm_validation/"
# Take the patched copies from /accounts -- that is where the other session's varlen work lives.
cp "$ACC/research/common/corpus_reasoning/lib/vllm_chunked_patch.py" "$R3/research/common/corpus_reasoning/lib/vllm_chunked_patch.py"
cp "$ACC/research/landmark/ctc_vllm_validation/run_vllm_eval.py"     "$R3/research/landmark/ctc_vllm_validation/run_vllm_eval.py"
grep -q "CHUNK_VARLEN_PREFILL" "$R3/research/common/corpus_reasoning/lib/vllm_chunked_patch.py" \
  || { echo "FATAL: CHUNK_VARLEN_PREFILL absent from the patch copy -- nothing to validate"; exit 2; }

VDIR=$R3/research/landmark/ctc_vllm_validation
# TODO(port): depends on debug/ctc_vllm_validation/general/ (build_prefills_any.py, grade_any.py), not ported
GDIR=$VDIR/general
source /data/prasann/node_local_env.sh 2>/dev/null || true
export HOME=/data/prasann/home CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONPATH=$R3/research/common:$R3/src
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
GRADE_PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

WORK=/data/prasann/chunk_val1632; mkdir -p "$WORK"
SUMMARY=$WORK/SUMMARY.tsv
echo -e "arm\trung\tgen_s\tsteps\tvarlen\tfallback\tset_f1\tparse\tnote" > "$SUMMARY"

SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
RUNGDIR=/data/prasann/ctc_suite_staged/eval_rungs/contradiction
BASE=/data/prasann/hf_models/Qwen3.5-4B-Base
[ -d "$SERVE" ] || { echo "FATAL: no serving copy at $SERVE"; exit 2; }

run_arm() {
  local ARM=$1 RUNG=$2 PREFILLS=$3 RJ=$4; shift 4
  echo ""; echo "=== [val1632] ARM=$ARM rung=$RUNG ENV: $* START=$(date '+%F %T') ==="
  local RESP="$WORK/resp_${ARM}.json"
  env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --gpu-mem-util 0.90 --out "$RESP" 2>&1 | tee "$WORK/log_${ARM}.txt" \
    | grep -E "gen_seconds|generated |patch debug|bumping max_model_len|Traceback|RuntimeError|OOM" | tail -12
  [ -s "$RESP" ] || { echo "*** ARM $ARM FAILED ***"; tail -25 "$WORK/log_${ARM}.txt"; return 1; }
  $GRADE_PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RJ" \
    --task contradiction --max-test-samples 500 --out "$WORK/grade_${ARM}.json" >/dev/null 2>&1
  $GRADE_PY - "$ARM" "$WORK" "$SUMMARY" "$RUNG" <<'PYEOF'
import json, sys
arm, work, summary, rung = sys.argv[1:5]
r = json.load(open(f"{work}/resp_{arm}.json")); g = json.load(open(f"{work}/grade_{arm}.json"))
pd = r.get("patch_debug") or {}
row = (f"{arm}\t{rung}\t{r['gen_seconds']:.1f}\t{pd.get('calls','-')}\t{pd.get('varlen','-')}\t"
       f"{pd.get('fallback','-')}\t{g['metric_value']:.4f}\t{g.get('parse_rate')}\t")
open(summary, "a").write(row + "\n"); print("RESULT " + row)
PYEOF
}

for RUNG in 16384 32768; do
  RJ="$RUNGDIR/rung_${RUNG}.jsonl"
  [ -f "$RJ" ] || { echo "!!! missing $RJ -- skipping"; continue; }
  P="$WORK/prefills_${RUNG}.json"
  [ -f "$P" ] || $GRADE_PY "$GDIR/build_prefills_any.py" --tokenizer "$BASE" --task contradiction \
    --eval-data "$RJ" --max-test-samples 500 --out "$P" || { echo "!!! prefill build failed"; continue; }
  # max_batched_tokens tracks the rung (the other session used rung + ~1k headroom).
  MBT=$(( RUNG + 1024 ))
  # PRODUCTION arm first so a preemption leaves the baseline rather than only the fast arm.
  run_arm "prod_${RUNG}" "$RUNG" "$P" "$RJ" CHUNK_EXTRA_STOP=']]' CHUNK_CACHE_IDS=1
  run_arm "vl16_${RUNG}" "$RUNG" "$P" "$RJ" CHUNK_EXTRA_STOP=']]' \
    CHUNK_MAX_BATCHED_TOKENS=$MBT CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18 CHUNK_VARLEN_PREFILL=1
done

echo ""; echo "=== [val1632] SUMMARY (speedup = prod_s / vl16_s at the same rung) ==="
column -t -s $'\t' "$SUMMARY"
echo "GATE: vl16 f1 must be within 0.018 of prod f1 at the same rung, parse_rate 1.0."
echo "=== [val1632] DONE $(date '+%F %T') ==="
