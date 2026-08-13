#!/bin/bash
# Lift the rung gate: is the round-4 concurrency speedup usable at 16k/32k, or does the mask tax eat it?
#
# WHERE THE GAP ACTUALLY IS NOW. Matched full-vs-cmix wall-clock from the newrung sweep
# (same driver, same rung files, eval_size=500), chunked/dense ratio:
#   msmarco   2k 7.1x  4k 5.3x  8k 5.0x  16k 2.4x  32k 2.3x
#   nq                                    16k 1.9x  32k 2.3x
#   hotpotqa                                        32k 2.4x
#   rerank    2k 3.7x  4k 6.6x  8k 7.7x  16k 11.4x  32k 11.2x   <-- the one bad family
# Everything except rerank has converged to ~2.3x at 32k. rerank has not, and rerank is also the
# task with the most decode steps (it emits a ranked list, not one id), which is exactly the axis
# `max_num_seqs` controls. 16k/32k still run at the historical seqs=8 because gpu_eval_task_chunked.sh
# gates the round-4 knobs on `RUNG <= 8192`; that gate is why rerank 32k costs 4253s.
#
# WHY THE GATE EXISTS AND WHY THIS IS NOT JUST "SET IT TO 128". The fallback mask build is
# O(num_actual_tokens x total_cache_tokens), and total_cache_tokens ~ max_model_len x CHUNK_SEQ_HEADROOM.
# So seqs=128/headroom=130 at 32k asks for ~16x the resident cache of the validated 8k point --
# and 512 already CRASHED at 8k (20.34 GiB mask transient). The sweep therefore holds the
# RESIDENT TOKEN BUDGET roughly constant instead of the sequence count:
#     8k x 128 = 1.05M tokens (validated)  ->  16k x 64 = 1.05M  ->  32k x 32 = 1.05M
# and brackets that point one step either side, so a turnover is visible rather than assumed.
#
# METRIC PARITY IS THE GATE, NOT SPEED. Round 2 §3 found two of three knobs have an eval_size- or
# rung-dependent SIGN, so every arm runs the production eval_size=500 and is compared against the
# already-measured baseline on the SAME rung file:
#   rerank rung_16384  mrr@10 0.9813  2056.9s     rerank rung_32768  mrr@10 0.9765  4253.7s
# 500-example SE at mrr~0.98 is ~0.006; an arm that moves the metric by more than ~0.012 is rejected
# no matter what it does to wall-clock. (Those rung files are the padded `eval_rungs_tokenaccurate`
# ladder -- fine as a fixed timing/parity substrate, since both sides read the identical file, but
# the mrr values themselves are not a difficulty measurement. See the hard-negative audit.)
#
#   sbatch debug/chunked_eval_speedup/rung_gate_16k32k.sbatch
#SBATCH --job-name=rung-gate
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/rung_gate_%j.log

set -uo pipefail
source /data/prasann/node_local_env.sh

# The driver lives in the node-local repo CLONE, not a loose script dir. cubbins' copy was verified
# byte-identical to repo HEAD before this was launched -- the stale-driver miss on mooney already
# cost four re-evals, so the guard below is a hard fail, not a warning.
DRIVER=/data/prasann/repo/OLMo-core/debug/ctc_vllm_validation/node_local/gpu_eval_task_chunked.sh
TASK=rerank
export SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-rerank-cmix
export RUNGDIR=/data/prasann/ctc_suite_staged/eval_rungs_tokenaccurate/rerank
TSV=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/chunked_eval_speedup/RUNG_GATE.tsv

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$DRIVER" ] || { echo "FATAL: no driver at $DRIVER"; exit 2; }
[ -d "$SERVE" ] || { echo "FATAL: no serving copy at $SERVE"; exit 2; }
for R in 16384 32768; do
  [ -f "$RUNGDIR/rung_${R}.jsonl" ] || { echo "FATAL: missing rung_${R}"; exit 2; }
done

# The driver is the node-local copy, and this node may be running a stale one -- that exact miss
# (RUNGDIR unsupported here but supported in the dense twin) already cost four re-evals tonight.
if ! grep -q "CHUNK_FAST_MAX_RUNG" "$DRIVER"; then
  echo "FATAL: driver on $(hostname) predates CHUNK_FAST_MAX_RUNG -- sync it before sweeping"; exit 2
fi

[ -f "$TSV" ] || printf 'rung\tseqs\tresident_tok\tgen_seconds\tspeedup\tmetric\tdelta\tparse_rate\teval_size\tverdict\n' > "$TSV"

# rung:seqs -- constant ~1.05M resident-token budget in the middle, bracketed either side.
for spec in "16384:32" "16384:64" "16384:128" "32768:16" "32768:32" "32768:64"; do
  RUNG="${spec%%:*}"; SEQS="${spec##*:}"
  case "$RUNG" in 16384) BASE_S=2056.9; BASE_M=0.9813 ;; 32768) BASE_S=4253.7; BASE_M=0.9765 ;; esac
  OUT=/data/prasann/rung_gate/r${RUNG}_s${SEQS}
  mkdir -p "$OUT"
  echo ""
  echo "######## rung=$RUNG seqs=$SEQS (resident ~$((RUNG*SEQS/1000))k tok) $(date '+%F %T') ########"

  # CHUNK_FAST_MAX_RUNG opens the gate; the *_FAST vars are what the driver then exports. Headroom
  # tracks seqs+2, matching the validated 128/130 pair -- headroom below seqs would starve the
  # scheduler and measure a different thing than the label says.
  T0=$(date +%s)
  OUTDIR="$OUT" RUNGDIR="$RUNGDIR" SERVE="$SERVE" \
  CHUNK_FAST_MAX_RUNG=999999 \
  CHUNK_MAX_NUM_SEQS_FAST="$SEQS" CHUNK_SEQ_HEADROOM_FAST="$((SEQS+2))" \
  TASK="$TASK" RUNGS="$RUNG" bash "$DRIVER" 2>&1 | tail -25
  WALL=$(( $(date +%s) - T0 ))
  echo "  arm wall (incl. engine init + prefill build) = ${WALL}s"

  python3 - "$OUT/grade_${RUNG}.json" "$RUNG" "$SEQS" "$BASE_S" "$BASE_M" "$TSV" <<'PY'
import json, sys, os
p, rung, seqs, base_s, base_m, tsv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
row = lambda *c: open(tsv, "a").write("\t".join(str(x) for x in c) + "\n")
if not os.path.exists(p):
    print(f"  rung={rung} seqs={seqs}: NO GRADE FILE -- arm crashed (likely the mask OOM)")
    row(rung, seqs, rung * seqs, "", "", "", "", "", "", "CRASH")
    raise SystemExit(0)
d = json.load(open(p))
# gen_seconds is the generation phase only -- the same clock the baselines were measured on.
# Comparing an arm's total job wall against a baseline's gen_seconds would credit engine init to
# the speedup and inflate every ratio.
g, m = d["gen_seconds"], d["metric_value"]
delta = m - base_m
# 500-example SE at mrr~0.98 is ~0.006; reject beyond 2 SE regardless of the speedup.
ok = abs(delta) <= 0.012
print(f"  rung={rung} seqs={seqs}: {g:.0f}s ({base_s/g:.2f}x)  {d['metric_name']}={m:.4f} "
      f"(base {base_m:.4f}, delta {delta:+.4f})  parse_rate={d.get('parse_rate')}  "
      f"{'PASS' if ok else 'METRIC MOVED -- REJECT'}")
row(rung, seqs, rung * seqs, f"{g:.0f}", f"{base_s/g:.2f}x", f"{m:.4f}", f"{delta:+.4f}",
    d.get("parse_rate"), d.get("eval_size"), "PASS" if ok else "REJECT")
PY
done

echo ""
echo "=== SUMMARY (baselines: 16k 2056.9s/0.9813, 32k 4253.7s/0.9765 at seqs=8) ==="
column -t "$TSV" 2>/dev/null || cat "$TSV"
echo "=== DONE $(date '+%F %T') ==="
