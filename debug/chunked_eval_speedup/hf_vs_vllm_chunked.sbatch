#!/bin/bash
# HF chunked-flex vs vLLM chunked-vllm: is the 0.079 drift the ENGINE, or the checkpoint path?
#
# WHAT THE EXISTING EVIDENCE ACTUALLY SAYS. results/ctc_suite/vllm_eval_status.md records ONE row:
# contradiction / qwen3.5-0.8b / full / rung2048, olmo-core-native 0.684 vs vLLM 0.763, |Δ|=0.079
# (~2.8x SE). Its chunked row (0.036 vs 0.049) agrees within ~1 SE but both arms sit near the floor,
# so it discriminates nothing. That single dense cell is the entire basis for the plan's "evaluator
# comparability -- must stay on vLLM" rule, and it has been quoted since (by me, repeatedly) as a
# property of "this model family". It is not established as one.
#
# WHY THAT MEASUREMENT CANNOT ASSIGN BLAME. It compared olmo-core native against vLLM -- and those
# two differ in TWO ways at once: different inference kernel AND different checkpoint path
# (olmo-core reads distcp; vLLM reads an HF export that went through export_olmo_to_hf.py plus the
# VL-wrapper repackaging). The write-up guessed "Gated-DeltaNet recurrent state compounding
# differently between olmo-core's kernel and vLLM's Triton/FLA kernel" and explicitly stopped:
# "Root cause not diagnosed further per the stop directive."
#
# WHY THIS COMPARISON SEPARATES THEM. vLLM does NOT use a bespoke format -- its serving copy IS an
# HF checkpoint, repackaged from the same `hf_exports_4b_chunked/` export this HF arm loads (key
# rename model.* -> model.language_model.* plus a synthesized visual tower; identical weights).
# So HF-flex vs vLLM-chunked holds the weights and the mask fixed and varies ONLY the engine:
#   HF ~= vLLM   -> the engines agree; the 0.079 is olmo-core's distcp path or its kernel, and the
#                   "must stay on vLLM" constraint on the model-scale study is unnecessary.
#   HF ~= olmo-core, vLLM apart -> vLLM is the outlier, and the ENTIRE 4B reference table (all vLLM)
#                   needs re-examination. That is the expensive outcome, which is why it is worth
#                   one job to find out.
#
# TASK CHOICE: outlier_amzn rung_8192, because a vLLM chunked number already exists for exactly this
# checkpoint+rung -- set_f1 0.8859 (measured twice: 2402s and 2315s) -- and an olmo-core-native arm
# is running now. Mid-range score, so any real gap is visible; unlike the contradiction 2k chunked
# row where both engines sat near 0.04.
#
#   sbatch debug/chunked_eval_speedup/hf_vs_vllm_chunked.sbatch
#SBATCH --job-name=hf-vs-vllm
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/hf_vs_vllm_%j.log

set -uo pipefail
source /data/prasann/node_local_env.sh

CR=/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning
HF_CKPT=/data/prasann/ctc_suite/hf_exports_4b_chunked/ctc-4b-outlier_amzn-cmix
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/outlier_amzn/rung_8192.jsonl
OUT=/data/prasann/hf_vs_vllm
mkdir -p "$OUT"

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -d "$HF_CKPT" ] || { echo "FATAL: no HF export at $HF_CKPT"; exit 2; }
[ -f "$RUNG" ] || { echo "FATAL: no rung $RUNG"; exit 2; }
ls "$HF_CKPT" | head -5

# The HF path is corpus-reasoning's own evaluator, run from that repo so its `scripts.lib.*` imports
# resolve. It needs torch + transformers; the olmo env has both.
cd "$CR"
export PYTHONPATH="$CR:${PYTHONPATH:-}"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

# --backend chunked-flex is the HF FlexAttention arm. --attention-pattern defaults to match the
# backend (chunked-* -> 'chunked'), but it is passed explicitly: this is the one knob that decides
# whether the chunked mask is applied at all, and a silent default flip would turn this into a
# full-attention run wearing a chunked label -- fast, higher-scoring, and completely wrong.
#
# --use-alpaca is REQUIRED and the script asserts on it: "Default helmet (2-shot) eval format is
# disabled -- training uses alpaca 0-shot." The vLLM path renders with use_alpaca=True inside
# load_unified_examples, so this is what makes the two arms the SAME PROMPT, not just the same
# weights. Without it the comparison would be engine AND prompt format at once -- the exact
# confound that made the original olmo-core-vs-vLLM number unattributable.
T0=$(date +%s)
$PY -u scripts/eval/evaluate.py \
  --backend chunked-flex --attention-pattern chunked \
  --use-alpaca \
  --base-model "$HF_CKPT" \
  --task outlier --max-test-samples 500 \
  --eval-data "$RUNG" \
  --doc-start-token '<|box_start|>' \
  --output-file "$OUT/hf_chunked_outlier_amzn_8192.json" 2>&1 | tail -30
RC=$?; HF_S=$(( $(date +%s) - T0 ))
echo "  HF chunked-flex rc=$RC wall=${HF_S}s"

echo ""
echo "=== THREE-WAY (outlier_amzn rung_8192, ctc-4b-outlier_amzn-cmix, eval_size=500) ==="
echo "  vLLM chunked      : set_f1 0.8859   2315s   [already measured, job 3442289]"
echo "  olmo-core native  : see job 3442289 ARM B"
$PY - "$OUT/hf_chunked_outlier_amzn_8192.json" "$HF_S" <<'PY'
import json, sys
p, secs = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(p))
except Exception as e:
    print(f"  HF chunked-flex   : FAILED to produce output ({e})")
    raise SystemExit(0)
# Scrape whatever metric block this evaluator wrote; do not assume a key name.
def find(o, depth=0):
    if depth > 3 or not isinstance(o, dict):
        return None
    for k in ("set_f1", "f1", "metric_value"):
        if k in o and isinstance(o[k], (int, float)):
            return k, o[k]
    for v in o.values():
        r = find(v, depth + 1)
        if r:
            return r
    return None
hit = find(d)
if hit:
    k, v = hit
    print(f"  HF chunked-flex   : {k} {v:.4f}   {secs}s")
    print(f"  |HF - vLLM| = {abs(v - 0.8859):.4f}   (500-example binomial SE at f1~0.89 is ~0.014;")
    print(f"   the disputed olmo-core-vs-vLLM gap was 0.079)")
else:
    print(f"  HF chunked-flex   : ran, but no metric key found -- inspect {p}")
PY
echo "=== DONE $(date '+%F %T') ==="
