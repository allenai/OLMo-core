#!/bin/bash
# Chunked-eval speedup, ROUND 2b: the FlexAttention TILE SIZE is the cost, not the KV allocation.
#
# WHAT ROUND 2a (job 3437892) SETTLED. Shrinking the KV allocation made things WORSE
# (111.4s -> 252.5s), because CHUNK_SEQ_HEADROOM and CHUNK_MAX_NUM_SEQS were swept together: with
# max_num_seqs=2 the same work needs 6205 forward steps instead of 1578. The allocation term is
# real but small; the STEP COUNT and the PER-STEP COST are what matter. Hypothesis refuted.
#
# WHAT THIS TESTS. In vLLM 0.25.1, `get_kernel_options` (flex_attention.py:1396) on the
# NON-direct-build branch -- the branch chunked mode is permanently on -- does:
#     preferred_block = 64                       # bf16
#     BLOCK_M = ensure_divisible(64, q_block_size)
#     BLOCK_N = ensure_divisible(64, kv_block_size)
# with `ensure_divisible(c, b) = min(c, b)` when b divides c. We pass flex_attn_kv_block_size=16 and
# never set flex_attn_q_block_size, whose small-blocks default is also 16. So the triton flex kernel
# runs **16x16 tiles** on an H200. That is the smallest tile the kernel supports, and it is where
# the 7x-vs-dense gap should live: dense mode uses FlashAttention, not flex at all.
#
# WHY RAISING THEM IS EXACT, NOT AN APPROXIMATION. On the fallback path q_block_size/kv_block_size
# are only the BlockMask's granularity. create_block_mask marks a block FULL only when mask_mod is
# true for every element in it, and flex_attention re-applies mask_mod elementwise inside every
# PARTIAL block. Coarser blocks therefore change which blocks are skipped, never which (q, kv)
# pairs are allowed. The documented "kv_block_size must divide the 528-token page or output is
# silently corrupted" trap comes from `_build_block_mask_direct`, which maps pages to blocks 1:1 --
# and that path is unreachable here (direct_build is False, and the patch now raises on it).
# THE CLAIM IS STILL CHECKED, NOT ASSUMED: every arm is graded and diffed against the baseline
# arm's generations, and any arm that emits "!!!!" shows up as parse_rate 0.
#
#SBATCH --job-name=chunk-combo
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_combo_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

# ---------------------------------------------------------------------------------------------
# Private repo copy. Job 3437406 (production chunked contradiction eval) is live on this node and
# runs one fresh python per rung out of /data/prasann/repo/OLMo-core -- editing that tree mid-run
# would silently change a production eval. Work from a node-local clone instead.
# ---------------------------------------------------------------------------------------------
ACC=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
R2=/data/prasann/repo_r2/OLMo-core
mkdir -p "$R2/src" "$R2/debug/ctc_vllm_validation" /data/prasann/joblogs
rsync -a --delete /data/prasann/repo/OLMo-core/src/ "$R2/src/"
rsync -a --delete /data/prasann/repo/OLMo-core/debug/ctc_vllm_validation/ "$R2/debug/ctc_vllm_validation/"
cp "$ACC/src/corpus_reasoning/lib/vllm_chunked_patch.py" "$R2/src/corpus_reasoning/lib/vllm_chunked_patch.py"
cp "$ACC/debug/ctc_vllm_validation/run_vllm_eval.py"     "$R2/debug/ctc_vllm_validation/run_vllm_eval.py"
echo "[setup] private repo at $R2"

VDIR=$R2/debug/ctc_vllm_validation
GDIR=$VDIR/general
source /data/prasann/node_local_env.sh 2>/dev/null || true
export HOME=/data/prasann/home
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONPATH=$R2/src
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
GRADE_PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

WORK=/data/prasann/chunk_combo
mkdir -p "$WORK"
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
N=${N_EX:-24}

echo "=== [chunk-combo] HOST=$(hostname) START=$(date '+%F %T') N=$N ==="

PREFILLS="$WORK/prefills_8192_${N}.json"
if [ ! -f "$PREFILLS" ]; then
  $GRADE_PY "$GDIR/build_prefills_any.py" \
    --tokenizer "$BASE_SNAP" --task contradiction --eval-data "$RUNG" \
    --max-test-samples $N --out "$PREFILLS" || exit 1
fi

# run_arm <name> <env assignments...>
run_arm () {
  local name="$1"; shift
  local resp="$WORK/resp_${name}.json"
  echo ""
  echo "######## ARM $name  [$*]  $(date '+%F %T') ########"
  env "$@" \
  $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --max-model-len 4096 --gpu-mem-util 0.85 \
    --out "$resp" 2>&1 \
    | grep -E "gen_seconds|generated |bumping max_model_len|chunked KV config|chunked flex tiles|stage profile|^[a-z_]+\.[pd] |^sync_head|^build_|^orig_build|^make_mask|patch debug state|Error|Traceback|OOM|out of memory|CUDA graph|Capturing" \
    | tail -40
  $GRADE_PY "$GDIR/grade_any.py" \
    --responses "$resp" --eval-data "$RUNG" --task contradiction \
    --max-test-samples "$N" --out "$WORK/grade_${name}.json" 2>&1 | tail -2
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
# agreement with the baseline arm's generations (greedy -> should be near-identical)
agree = ""
try:
    b = json.load(open(f"{work}/resp_arm00_base.json"))["responses"]
    a = r["responses"]
    keys = set(a) & set(b)
    same = sum(1 for k in keys if a[k] == b[k])
    agree = f"  gen_match_vs_base={same}/{len(keys)}"
except Exception:
    pass
print(f"ARM {name}  gen_seconds={r.get('gen_seconds', -1):.1f}  load={r.get('load_seconds', 0):.0f}"
      f"  eval_size={r.get('eval_size')}  {g.get('metric_name')}={g.get('metric_value')}"
      f"  parse_rate={g.get('parse_rate')}{agree}", flush=True)
PYEOF
}

# d00 is the production config re-run ALONE (the tile sweep shared cubbins with the concurrency
# sweep, which inflated its per-step cost); every number below is relative to this one.
run_arm arm00_base
# CUDA graphs, second attempt. The default mode captured FULL decode graphs and died with a
# device-side assert on first replay -- a full graph freezes the BlockMask it captured while the
# patch rebuilds one every step. PIECEWISE splits at the attention op, so the block_mask is never
# captured; NONE keeps torch.compile but no graphs at all, isolating how much of the win (if any)
# is compilation rather than graph capture.
run_arm d01_piecewise CHUNK_ENFORCE_EAGER=0 CHUNK_CUDAGRAPH_MODE=PIECEWISE
run_arm d02_compile   CHUNK_ENFORCE_EAGER=0 CHUNK_CUDAGRAPH_MODE=NONE
# The two levers that DID work, separately and together.
run_arm d03_cache        CHUNK_CACHE_IDS=1
run_arm d04_seqs32       CHUNK_MAX_NUM_SEQS=32 CHUNK_SEQ_HEADROOM=34 CHUNK_MAX_BATCHED_TOKENS=512
run_arm d05_cache_seqs32 CHUNK_CACHE_IDS=1 CHUNK_MAX_NUM_SEQS=32 CHUNK_SEQ_HEADROOM=34 CHUNK_MAX_BATCHED_TOKENS=512
run_arm d06_cache_seqs64 CHUNK_CACHE_IDS=1 CHUNK_MAX_NUM_SEQS=64 CHUNK_SEQ_HEADROOM=66 CHUNK_MAX_BATCHED_TOKENS=256

echo ""
echo "=== SUMMARY ==="
$GRADE_PY - "$WORK" <<'PYEOF'
import json, os, sys
work = sys.argv[1]
base = None
rows = []
for name in sorted(os.listdir(work)):
    if not (name.startswith("resp_") and name.endswith(".json")):
        continue
    arm = name[5:-5]
    try:
        r = json.load(open(f"{work}/{name}"))
    except Exception:
        continue
    try:
        g = json.load(open(f"{work}/grade_{arm}.json"))
    except Exception:
        g = {}
    rows.append((arm, r.get("gen_seconds", -1), g.get("metric_value"), g.get("parse_rate"), r))
    if arm == "arm00_base":
        base = r
print(f"{'arm':22s} {'gen_s':>8s} {'speedup':>8s} {'score':>7s} {'parse':>6s} {'gen_match':>10s}")
b_s = next((s for a, s, *_ in rows if a == "arm00_base"), None)
for arm, s, score, pr, r in rows:
    sp = f"{b_s/s:.2f}x" if (b_s and s and s > 0) else "-"
    m = "-"
    if base is not None:
        ka = set(r["responses"]) & set(base["responses"])
        m = f"{sum(1 for k in ka if r['responses'][k] == base['responses'][k])}/{len(ka)}"
    sc = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
    pv = f"{pr:.3f}" if isinstance(pr, (int, float)) else "-"
    print(f"{arm:22s} {s:8.1f} {sp:>8s} {sc:>7s} {pv:>6s} {m:>10s}")
PYEOF
echo "=== [chunk-combo] DONE $(date '+%F %T') ==="
