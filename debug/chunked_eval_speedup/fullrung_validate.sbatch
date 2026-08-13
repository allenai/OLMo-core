#!/bin/bash
# Chunked-eval speedup, ROUND 2 FINAL GATE: reproduce the known-good number at FULL eval_size.
#
# WHY THIS RUN EXISTS. Every sweep so far used eval_size 24 or 48, and one of them proved that is
# not enough: `d04_seqs32` (max_num_seqs=32, eval_size=24) came out 1.5x SLOWER, while the identical
# knob at eval_size=48 was 1.19x FASTER. The reason is structural -- when eval_size <= max_num_seqs
# every request is resident at once, so you pay the full O(N x N.L) mask cost with no queue left to
# drain and no step-count saving. The concurrency knob can therefore only be *sized* against the
# eval_size it will actually run at, which in production is 500.
#
# This run uses the EXACT prefills pack that produced the known-good production numbers
# (ctc_suite_vllm_results_chunked/contradiction_iid/prefills_8192.json, 500 rows), so the comparison
# is apples-to-apples against:
#     chunked  gen_seconds=2066.1  set_f1=0.8039  parse_rate=1.000   (production config)
#     dense    gen_seconds= 116.4  set_f1=0.9760  parse_rate=1.000
# A candidate is only acceptable if set_f1 lands at 0.8039 +/- the 500-example binomial SE (0.018)
# AND parse_rate stays 1.000.
#
#SBATCH --job-name=chunk-fullrung
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_fullrung_%j.log
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

WORK=/data/prasann/chunk_fullrung
mkdir -p "$WORK"
MODE=${MODE:-chunked}
if [ "$MODE" = "full" ]; then
  SERVE=/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-contradiction_iid-full
else
  SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
fi
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl
# The very pack the known-good 2066.1s / f1 0.8039 run consumed.
PREFILLS=/data/prasann/ctc_suite_vllm_results_chunked/contradiction_iid/prefills_8192.json
ARM=${ARM:?set ARM}

echo "=== [chunk-fullrung] ARM=$ARM HOST=$(hostname) START=$(date '+%F %T') ==="
env | grep -E '^CHUNK_' | sort

RESP="$WORK/resp_${ARM}.json"
$VLLM_PY -u "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode "$MODE" --task contradiction \
  --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.90 \
  --out "$RESP" 2>&1 \
  | grep -E "gen_seconds|generated |chunked KV config|chunked flex tiles|patch debug state|Traceback|Error|OOM|out of memory" \
  | tail -20

$GRADE_PY "$GDIR/grade_any.py" --responses "$RESP" --eval-data "$RUNG" \
  --task contradiction --max-test-samples 500 --out "$WORK/grade_${ARM}.json"

$GRADE_PY - "$ARM" "$WORK" <<'PYEOF'
import json, sys
arm, work = sys.argv[1], sys.argv[2]
r = json.load(open(f"{work}/resp_{arm}.json"))
g = json.load(open(f"{work}/grade_{arm}.json"))
import os
BASE_S, BASE_F1 = (116.36, 0.976) if os.environ.get("MODE")=="full" else (2066.1, 0.8038666666666667)
s, f1 = r["gen_seconds"], g["metric_value"]
print("")
print(f"RESULT arm={arm}")
print(f"  gen_seconds = {s:.1f}   (production baseline 2066.1)  speedup = {BASE_S/s:.2f}x")
print(f"  steps       = {r.get('patch_debug', {}).get('calls')}  (baseline 32573)")
print(f"  set_f1      = {f1:.4f}  (known-good 0.8039, 500-example SE 0.018)  delta = {f1-BASE_F1:+.4f}")
print(f"  parse_rate  = {g.get('parse_rate')}  (must be 1.0)")
print(f"  patch_debug = {r.get('patch_debug')}")
ok = abs(f1 - BASE_F1) <= 0.018 and g.get("parse_rate") == 1.0
print(f"  VERDICT: {'REPRODUCES known-good' if ok else '*** DOES NOT REPRODUCE ***'}")
PYEOF
echo "=== [chunk-fullrung] DONE $(date '+%F %T') ==="
