#!/bin/bash
# Measure the CHUNK_VARLEN_PREFILL speedup at 16k/32k on the contradiction_iid ladder.
#
# WHY THE PREVIOUS TWO ATTEMPTS COULD NEVER HAVE WORKED. Jobs 3443907 and 3444725 both ran against
# `eval_rungs/contradiction` -- the old both-mode ladder -- whose prefill lengths do not track its
# rung labels at all:
#     rung_16384   len min/mean/max = 15334 / 29078 / 106031     (max = 6.5x the label)
#     rung_32768   len min/mean/max = 31386 / 59063 / 214446     (max = 6.5x the label)
# Two independent consequences, and the second is fatal on its own:
#   1. MEMORY. The driver raises max_model_len to cover the LONGEST prefill, so the engine sized
#      itself for 107k / 215k positions. That is what left 4.7 GiB free (with 64.8 GiB sitting in
#      CUDA-graph private pools) and OOM'd on a 7.12 GiB activation. Lowering --gpu-mem-util from
#      0.90 to 0.80 did not help, because the cost is driven by max_model_len, not by the fraction.
#   2. VARLEN CANNOT ENGAGE ANYWAY. The recipe requires CHUNK_MAX_BATCHED_TOKENS >= the longest
#      prompt; a prefill longer than MBT is split across steps and drops to the fallback path. The
#      recipe sets MBT = rung + 1024 = 17408 / 33792, which is ~6x SHORT of 106031 / 214446. So even
#      with infinite memory those runs would have measured the fallback path, not varlen.
# No amount of memory tuning fixes (2). The eval set was wrong, not the knobs.
#
# WHY contradiction_iid IS THE RIGHT LADDER. Same task, same chunked serving copy, and its lengths
# actually track the labels:
#     rung_16384   15578 / 17096 / 18628     (max = 1.14x)
#     rung_32768   32180 / 34275 / 36519     (max = 1.11x)
# So max_model_len stays near the rung, MBT can cover the longest prompt, and varlen is eligible on
# every example. It is also the ladder the suite actually reports contradiction on, so a speedup
# measured here applies to the runs we care about.
#
# ⚠ MBT IS DERIVED FROM THE BUILT PREFILLS, NOT FROM THE RUNG LABEL. That is the whole lesson above.
# Each rung's MBT is read off the prefill file after it is built.
#
#   sbatch research/landmark/chunked_eval_speedup/varlen_16k32k_iid.sbatch
#SBATCH --nodelist=cubbins
#SBATCH --job-name=varlen-iid
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --output=/data/prasann/joblogs/varlen_iid_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
ACC=$REPO_ROOT
R3=/data/prasann/repo_r3/OLMo-core
mkdir -p "$R3/src" "$R3/research/landmark/ctc_vllm_validation" /data/prasann/joblogs
rsync -a --delete $REPO_ROOT/src/ "$R3/src/"
rsync -a --delete $REPO_ROOT/research/landmark/ctc_vllm_validation/ "$R3/research/landmark/ctc_vllm_validation/"
cp "$ACC/recipes/eval/legacy_native/ctc_eval/lib/vllm_chunked_patch.py" "$R3/recipes/eval/legacy_native/ctc_eval/lib/vllm_chunked_patch.py"
cp "$ACC/research/landmark/ctc_vllm_validation/run_vllm_eval.py"     "$R3/research/landmark/ctc_vllm_validation/run_vllm_eval.py"
grep -q "CHUNK_VARLEN_PREFILL" "$R3/recipes/eval/legacy_native/ctc_eval/lib/vllm_chunked_patch.py" \
  || { echo "FATAL: CHUNK_VARLEN_PREFILL absent from the patch copy"; exit 2; }

VDIR=$R3/research/landmark/ctc_vllm_validation
# TODO(port): depends on debug/ctc_vllm_validation/general/ (build_prefills_any.py, grade_any.py), not ported
GDIR=$VDIR/general
source /data/prasann/node_local_env.sh 2>/dev/null || true
export HOME=/data/prasann/home CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONPATH=$R3/research/common:$R3/src
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

WORK=/data/prasann/varlen_iid; mkdir -p "$WORK"
RESULTS=$ACC/research/landmark/chunked_eval_speedup/results_iid; mkdir -p "$RESULTS"
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction_iid-cmix
[ -d "$SERVE" ] || SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
RUNGDIR=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid
BASE=/data/prasann/hf_models/Qwen3.5-4B-Base
[ -d "$SERVE" ]   || { echo "FATAL: no chunked serving copy"; exit 2; }
[ -d "$RUNGDIR" ] || { echo "FATAL: no contradiction_iid rung dir on $(hostname)"; exit 2; }
echo "=== HOST=$(hostname) SERVE=$SERVE START=$(date '+%F %T') ==="

SUMMARY=$WORK/SUMMARY.tsv
echo -e "rung\tarm\tgen_s\tspeedup\tvarlen\tfallback\tset_f1\tparse\tmaxlen\tMBT" > "$SUMMARY"

for RUNG in 16384 32768; do
  RJ="$RUNGDIR/rung_${RUNG}.jsonl"
  [ -f "$RJ" ] || { echo "!!! missing $RJ -- skipping"; continue; }
  P="$WORK/prefills_${RUNG}.json"
  [ -f "$P" ] || $PY "$GDIR/build_prefills_any.py" --tokenizer "$BASE" --task contradiction \
      --eval-data "$RJ" --max-test-samples 500 --out "$P" 2>&1 | tail -1
  [ -f "$P" ] || { echo "!!! prefill build failed rung=$RUNG"; continue; }

  # Derive MBT from the ACTUAL longest prefill. Hardcoding rung+1024 is precisely what made the two
  # previous attempts measure the fallback path while believing they measured varlen.
  # build_prefills_any.py writes {"rows": [{"idx", "prefill": [token ids], "n_chunks", "gold"}, ...]}
  # and the length it reports is len(row["prefill"]). The first version of this guessed at
  # "input_ids"/"num_tokens"/... , found nothing, and correctly refused to run rather than invent an
  # MBT -- which is the behaviour we want, but the key is not a guess, it is in the builder.
  MAXLEN=$($PY - "$P" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
print(max(len(r["prefill"]) for r in rows))
PY
)
  if ! [ "$MAXLEN" -gt 0 ] 2>/dev/null; then
    echo "!!! could not read prefill lengths for rung=$RUNG -- refusing to guess MBT"; continue
  fi
  MBT=$(( MAXLEN + 2048 ))
  echo ""; echo "######## rung=$RUNG maxlen=$MAXLEN MBT=$MBT $(date '+%T') ########"

  run_arm() {
    local ARM=$1; shift
    local RESP="$WORK/resp_${ARM}_${RUNG}.json"
    echo "--- arm=$ARM $* $(date '+%T') ---"
    env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" --hf-model "$SERVE" --prefills "$P" \
      --mode chunked --task contradiction --max-new-tokens 512 --gpu-mem-util 0.85 --out "$RESP" \
      2>&1 | tee "$WORK/log_${ARM}_${RUNG}.txt" \
      | grep -E "gen_seconds|generated |patch debug|bumping max_model_len|OutOfMemory" | tail -6
    [ -s "$RESP" ] || { echo "*** arm=$ARM rung=$RUNG FAILED ***"; tail -12 "$WORK/log_${ARM}_${RUNG}.txt"; return 1; }
    $PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RJ" --task contradiction \
      --max-test-samples 500 --out "$WORK/grade_${ARM}_${RUNG}.json" >/dev/null 2>&1
    cp "$WORK/grade_${ARM}_${RUNG}.json" "$RESULTS/" 2>/dev/null
    $PY - "$ARM" "$RUNG" "$WORK" "$SUMMARY" "$MAXLEN" "$MBT" <<'PYEOF'
import json, os, sys
arm, rung, work, summary, maxlen, mbt = sys.argv[1:7]
r = json.load(open(f"{work}/resp_{arm}_{rung}.json")); g = json.load(open(f"{work}/grade_{arm}_{rung}.json"))
pd = r.get("patch_debug") or {}
prod = None
if os.path.exists(f"{work}/gen_prod_{rung}.txt"):
    prod = float(open(f"{work}/gen_prod_{rung}.txt").read().strip())
s = r["gen_seconds"]
if arm == "prod":
    open(f"{work}/gen_prod_{rung}.txt", "w").write(str(s))
sp = f"{prod/s:.2f}x" if (prod and arm != "prod") else "-"
row = (f"{rung}\t{arm}\t{s:.1f}\t{sp}\t{pd.get('varlen','-')}\t{pd.get('fallback','-')}\t"
       f"{g['metric_value']:.4f}\t{g.get('parse_rate')}\t{maxlen}\t{mbt}")
open(summary, "a").write(row + "\n"); print("RESULT " + row)
PYEOF
  }

  # Production first, so a preemption leaves the baseline rather than only the fast arm.
  run_arm prod CHUNK_EXTRA_STOP=']]' CHUNK_CACHE_IDS=1
  run_arm varlen CHUNK_EXTRA_STOP=']]' CHUNK_MAX_BATCHED_TOKENS=$MBT CHUNK_MAX_NUM_SEQS=16 \
                 CHUNK_SEQ_HEADROOM=18 CHUNK_VARLEN_PREFILL=1
done

echo ""; echo "=== SUMMARY ==="
column -t -s $'\t' "$SUMMARY"
echo ""
echo "  GATE 1 (validity): the varlen arm must show varlen >> fallback. If fallback dominates, the"
echo "  plan did not engage and the TIMING is meaningless -- the metric is still valid."
echo "  GATE 2 (fidelity): |set_f1 varlen - set_f1 prod| <= 0.018 at the same rung, parse_rate 1.0."
echo "=== DONE $(date '+%F %T') -> $RESULTS ==="
