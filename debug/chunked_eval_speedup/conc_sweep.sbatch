#!/bin/bash
# Chunked-eval speedup, ROUND 2c: STEP COUNT is the other half of the gap.
#
# THE ASYMMETRY NOBODY COSTED. In run_vllm_eval.py the chunked arm is capped to
#     max_num_seqs=8, max_num_batched_tokens=2048
# while the dense arm passes `extra={}` and therefore runs at vLLM's DEFAULTS (max_num_seqs=1024).
# The production 8k contradiction runs make the consequence explicit: chunked needed 32573 forward
# steps for 500 examples (patch_debug.calls) at 63 ms/step = 2066 s; dense finished the same 500 in
# 116 s. Much of the 17.8x is therefore not "flex is slow" at all -- it is chunked doing ~30x more
# forward steps because only 8 sequences are resident at a time.
#
# WHY THE CAP EXISTS, AND WHAT IT ACTUALLY BINDS. On the fallback path the BlockMask comes from
# torch's create_block_mask, which materializes a dense (num_actual_tokens x total_cache_tokens)
# boolean before reducing it to blocks. At max_num_batched_tokens=2048 and a 94k-token cache that
# is a ~192 MB temporary per PREFILL step -- which is what the cap is really protecting against.
# DECODE steps have num_actual_tokens = num_reqs (8), so they materialize ~750 KB. So the two knobs
# should never have been tied: max_num_batched_tokens must stay small (it bounds the prefill
# temporary), but max_num_seqs can grow (it only widens decode, the cheap direction).
#
# Round 2a swept them TOGETHER and so could only observe the two moving in opposite directions
# (max_num_seqs 8->2 cost 2.3x). This sweep holds max_num_batched_tokens at 2048 and moves
# max_num_seqs alone, with the page allocation scaled to actually hold that many sequences.
#
#SBATCH --job-name=chunk-conc-sweep
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_conc_sweep_%j.log
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

WORK=/data/prasann/chunk_conc_sweep
mkdir -p "$WORK"
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
N=${N_EX:-48}
# Best tile config found by tile_sweep.sbatch; override from the launch env.
KVB=${KVB:-16}
QB=${QB:-}

echo "=== [chunk-conc-sweep] HOST=$(hostname) START=$(date '+%F %T') N=$N KVB=$KVB QB=${QB:-default} ==="

PREFILLS="$WORK/prefills_8192_${N}.json"
if [ ! -f "$PREFILLS" ]; then
  $GRADE_PY "$GDIR/build_prefills_any.py" \
    --tokenizer "$BASE_SNAP" --task contradiction --eval-data "$RUNG" \
    --max-test-samples $N --out "$PREFILLS" || exit 1
fi

run_arm () {
  local name="$1"; shift
  local resp="$WORK/resp_${name}.json"
  echo ""
  echo "######## ARM $name  [$*]  $(date '+%F %T') ########"
  env CHUNK_KV_BLOCK="$KVB" ${QB:+CHUNK_Q_BLOCK=$QB} "$@" \
  $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.90 \
    --out "$resp" 2>&1 \
    | grep -E "gen_seconds|generated |chunked KV config|chunked flex tiles|patch debug state|Traceback|Error|OOM|out of memory" \
    | tail -20
  $GRADE_PY "$GDIR/grade_any.py" --responses "$resp" --eval-data "$RUNG" \
    --task contradiction --max-test-samples "$N" --out "$WORK/grade_${name}.json" 2>&1 | tail -1
  $GRADE_PY - "$name" "$WORK" <<'PYEOF'
import json, sys
name, work = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(f"{work}/resp_{name}.json"))
except Exception as e:
    print(f"ARM {name}: NO RESULT ({e})"); raise SystemExit
try:
    g = json.load(open(f"{work}/grade_{name}.json"))
except Exception:
    g = {}
agree = ""
try:
    b = json.load(open(f"{work}/resp_c00_base.json"))["responses"]
    a = r["responses"]; k = set(a) & set(b)
    agree = f"  gen_match_vs_base={sum(1 for x in k if a[x]==b[x])}/{len(k)}"
except Exception:
    pass
print(f"ARM {name}  gen_seconds={r.get('gen_seconds',-1):.1f}  steps={r.get('patch_debug',{}).get('calls')}"
      f"  {g.get('metric_name')}={g.get('metric_value')}  parse_rate={g.get('parse_rate')}{agree}",
      flush=True)
PYEOF
}

# c00 is the production concurrency (8 seqs, headroom 10).
#
# BUDGETING THE PREFILL TEMPORARY. torch's create_block_mask materializes a dense
# (num_actual_tokens x total_cache_tokens) BOOL before reducing it to blocks
# (torch/nn/attention/flex_attention.py:1276 -> create_mask -> _convert_mask_to_block_mask), and
# that materialization is at FULL resolution -- q_block_size/kv_block_size shrink the reduced
# output, not this temporary. So the byte cost of a step is num_actual_tokens * total_cache_tokens.
# Widening max_num_seqs grows total_cache_tokens, which makes PREFILL steps (num_actual_tokens =
# max_num_batched_tokens) quadratically expensive while DECODE steps (num_actual_tokens = num_reqs)
# stay cheap. Each arm therefore lowers max_num_batched_tokens to hold that temporary near the
# ~190 MB the production config already pays, and only decode gets wider.
#   seqs=8   cache  94k tok, bt=2048 -> 192 MB   (production)
#   seqs=16  cache 178k tok, bt=1024 -> 182 MB
#   seqs=32  cache 318k tok, bt=512  -> 163 MB
#   seqs=48  cache 458k tok, bt=512  -> 234 MB
run_arm c00_base    CHUNK_MAX_NUM_SEQS=8  CHUNK_SEQ_HEADROOM=10 CHUNK_MAX_BATCHED_TOKENS=2048
run_arm c01_seqs16  CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18 CHUNK_MAX_BATCHED_TOKENS=1024
run_arm c02_seqs32  CHUNK_MAX_NUM_SEQS=32 CHUNK_SEQ_HEADROOM=34 CHUNK_MAX_BATCHED_TOKENS=512
run_arm c03_seqs48  CHUNK_MAX_NUM_SEQS=48 CHUNK_SEQ_HEADROOM=50 CHUNK_MAX_BATCHED_TOKENS=512
# Control: does the SMALLER prefill budget alone cost anything at production concurrency? If
# c05 ~= c00 then every difference above is attributable to decode width, not to the bt change.
run_arm c05_base_bt512 CHUNK_MAX_NUM_SEQS=8 CHUNK_SEQ_HEADROOM=10 CHUNK_MAX_BATCHED_TOKENS=512

echo ""
echo "=== SUMMARY ==="
$GRADE_PY - "$WORK" <<'PYEOF'
import json, os, sys
work = sys.argv[1]
rows = []
for fn in sorted(os.listdir(work)):
    if not (fn.startswith("resp_") and fn.endswith(".json")):
        continue
    arm = fn[5:-5]
    try:
        r = json.load(open(f"{work}/{fn}"))
    except Exception:
        continue
    try:
        g = json.load(open(f"{work}/grade_{arm}.json"))
    except Exception:
        g = {}
    rows.append((arm, r.get("gen_seconds", -1), r.get("patch_debug", {}).get("calls"),
                 g.get("metric_value"), g.get("parse_rate"), r))
b = next((s for a, s, *_ in rows if a == "c00_base"), None)
base = next((r for a, *_x, r in rows if a == "c00_base"), None)
print(f"{'arm':14s} {'gen_s':>8s} {'speedup':>8s} {'steps':>7s} {'score':>7s} {'parse':>6s} {'gen_match':>10s}")
for arm, s, st, sc, pr, r in rows:
    sp = f"{b/s:.2f}x" if (b and s and s > 0) else "-"
    m = "-"
    if base is not None:
        k = set(r["responses"]) & set(base["responses"])
        m = f"{sum(1 for x in k if r['responses'][x] == base['responses'][x])}/{len(k)}"
    print(f"{arm:14s} {s:8.1f} {sp:>8s} {str(st):>7s} "
          f"{(f'{sc:.4f}' if isinstance(sc,(int,float)) else '-'):>7s} "
          f"{(f'{pr:.3f}' if isinstance(pr,(int,float)) else '-'):>6s} {m:>10s}")
PYEOF
echo "=== [chunk-conc-sweep] DONE $(date '+%F %T') ==="
