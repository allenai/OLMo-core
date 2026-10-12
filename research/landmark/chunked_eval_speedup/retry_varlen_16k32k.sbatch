#!/bin/bash
# Retry the CHUNK_VARLEN_PREFILL measurement at 16k/32k. Job 3443907 got the production baselines
# and then OOM'd on BOTH varlen arms, so the speedup is still unmeasured at the rungs that need it.
#
# WHY IT OOM'D, AND WHY THAT IS NOT A VARLEN BUG. The contradiction rung files carry huge outlier
# prefills -- rung_16384 maxes at 106,031 tokens and rung_32768 at 214,446, against p50s of 29k and
# 59k. The driver raises max_model_len to cover the LONGEST prefill, so the engine sized itself for
# 215k positions; at --gpu-mem-util 0.90 that left ~1.3 GiB free, and the varlen path's larger batch
# then asked for 1.85 GiB. The production path survived only because it batches less aggressively.
# (Rung labels not tracking token counts is a known property of these ladders, not a new problem.)
#
# WHAT THIS RUN CHANGES -- two knobs, both in the direction of headroom:
#   --gpu-mem-util 0.90 -> 0.80    leaves ~14 GiB for activations instead of ~1.3
#   CHUNK_MAX_NUM_SEQS 16 -> 8 at 32k only  (16 stays at 16k, where it fit)
# seqs is NOT lowered globally: the other session measured seqs=64 as WORSE than seqs=16 at 8k
# (9.86x vs 15.45x) because its varlen path mostly fell back, so this knob has a measured optimum and
# is only being moved where the memory forces it.
#
# ⚠ REUSES THE PREFILLS AND THE PROD BASELINES FROM JOB 3443907. /data/prasann/chunk_val1632 still
# holds prefills_{16384,32768}.json and a SUMMARY.tsv with the production timings
# (16k: 1336.8s f1 0.3186; 32k: 3762.6s f1 0.2012), both measured on exactly these prefills. Re-running
# the production arm would cost another ~85 minutes to reproduce numbers that are already controlled,
# so this job runs the varlen arms ONLY and compares against the recorded baselines. That is sound
# precisely because the prefills are byte-identical -- it would not be if they were rebuilt.
#
#   sbatch research/landmark/chunked_eval_speedup/retry_varlen_16k32k.sbatch
#SBATCH --nodelist=cubbins
#SBATCH --job-name=chunk-retry1632
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/chunk_retry1632_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
ACC=$REPO_ROOT
R3=/data/prasann/repo_r3/OLMo-core
mkdir -p "$R3/src" "$R3/research/landmark/ctc_vllm_validation" /data/prasann/joblogs
rsync -a --delete $REPO_ROOT/src/ "$R3/src/"
rsync -a --delete $REPO_ROOT/research/landmark/ctc_vllm_validation/ "$R3/research/landmark/ctc_vllm_validation/"
cp "$ACC/research/common/corpus_reasoning/lib/vllm_chunked_patch.py" "$R3/research/common/corpus_reasoning/lib/vllm_chunked_patch.py"
cp "$ACC/research/landmark/ctc_vllm_validation/run_vllm_eval.py"     "$R3/research/landmark/ctc_vllm_validation/run_vllm_eval.py"
grep -q "CHUNK_VARLEN_PREFILL" "$R3/research/common/corpus_reasoning/lib/vllm_chunked_patch.py" \
  || { echo "FATAL: CHUNK_VARLEN_PREFILL absent from the patch copy"; exit 2; }

VDIR=$R3/research/landmark/ctc_vllm_validation
# TODO(port): depends on debug/ctc_vllm_validation/general/ (build_prefills_any.py, grade_any.py), not ported
GDIR=$VDIR/general
source /data/prasann/node_local_env.sh 2>/dev/null || true
export HOME=/data/prasann/home CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONPATH=$R3/research/common:$R3/src
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
GRADE_PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

WORK=/data/prasann/chunk_val1632          # same dir: the prefills and baselines live here
OUT=/data/prasann/chunk_retry1632; mkdir -p "$OUT"
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
RUNGDIR=/data/prasann/ctc_suite_staged/eval_rungs/contradiction
[ -d "$SERVE" ] || { echo "FATAL: no serving copy at $SERVE"; exit 2; }

# Baselines as recorded by 3443907, keyed by rung: gen_seconds and set_f1.
declare -A PROD_S=( [16384]=1336.8 [32768]=3762.6 )
declare -A PROD_F=( [16384]=0.3186 [32768]=0.2012 )
declare -A SEQS=(   [16384]=16     [32768]=8 )

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
SUMMARY=$OUT/SUMMARY.tsv
echo -e "rung\tarm\tgen_s\tspeedup\tvarlen\tfallback\tset_f1\tprod_f1\tdelta\tparse" > "$SUMMARY"

for RUNG in 16384 32768; do
  P="$WORK/prefills_${RUNG}.json"
  RJ="$RUNGDIR/rung_${RUNG}.jsonl"
  [ -f "$P" ]  || { echo "!!! missing prefills $P -- cannot compare against the recorded baseline"; continue; }
  [ -f "$RJ" ] || { echo "!!! missing $RJ"; continue; }
  MBT=$(( RUNG + 1024 ))
  RESP="$OUT/resp_vl_${RUNG}.json"
  echo ""; echo "=== rung=$RUNG varlen seqs=${SEQS[$RUNG]} mem-util 0.80 $(date '+%T') ==="
  env CHUNK_EXTRA_STOP=']]' CHUNK_MAX_BATCHED_TOKENS=$MBT CHUNK_MAX_NUM_SEQS=${SEQS[$RUNG]} \
      CHUNK_SEQ_HEADROOM=18 CHUNK_VARLEN_PREFILL=1 \
    $VLLM_PY -u "$VDIR/run_vllm_eval.py" --hf-model "$SERVE" --prefills "$P" --mode chunked \
      --task contradiction --max-new-tokens 512 --gpu-mem-util 0.80 --out "$RESP" 2>&1 \
    | tee "$OUT/log_vl_${RUNG}.txt" \
    | grep -E "gen_seconds|generated |patch debug|bumping max_model_len|OutOfMemory|Traceback" | tail -8
  if [ ! -s "$RESP" ]; then
    echo "*** rung=$RUNG STILL FAILED ***"; tail -20 "$OUT/log_vl_${RUNG}.txt"
    echo -e "$RUNG\tvarlen\tFAILED\t-\t-\t-\t-\t${PROD_F[$RUNG]}\t-\t-" >> "$SUMMARY"
    continue
  fi
  $GRADE_PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RJ" --task contradiction \
    --max-test-samples 500 --out "$OUT/grade_vl_${RUNG}.json" >/dev/null 2>&1
  $GRADE_PY - "$RUNG" "$OUT" "$SUMMARY" "${PROD_S[$RUNG]}" "${PROD_F[$RUNG]}" <<'PYEOF'
import json, sys
rung, out, summary, prod_s, prod_f = sys.argv[1:6]
r = json.load(open(f"{out}/resp_vl_{rung}.json")); g = json.load(open(f"{out}/grade_vl_{rung}.json"))
pd = r.get("patch_debug") or {}
s, f = r["gen_seconds"], g["metric_value"]
row = (f"{rung}\tvarlen\t{s:.1f}\t{float(prod_s)/s:.2f}x\t{pd.get('varlen','-')}\t{pd.get('fallback','-')}\t"
       f"{f:.4f}\t{prod_f}\t{f - float(prod_f):+.4f}\t{g.get('parse_rate')}")
open(summary, "a").write(row + "\n"); print("RESULT " + row)
PYEOF
done

echo ""; echo "=== SUMMARY (speedup vs job 3443907's production arm on identical prefills) ==="
column -t -s $'\t' "$SUMMARY"
echo "GATE: |delta| <= 0.018 and parse_rate 1.0, else the speedup is not free."
echo "=== DONE $(date '+%F %T') ==="
