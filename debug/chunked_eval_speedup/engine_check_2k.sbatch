#!/bin/bash
# FAST engine check at the 2k rung: HF chunked-flex vs vLLM chunked, same checkpoint, same prompt.
#
# WHY 2k. The 8k arms take ~40 min EACH, and the question -- do two engines running the same weights
# and the same mask agree? -- does not need a long rung. 2k also matches the rung of the single
# historical parity row (contradiction/0.8b/full/2048: olmo-core 0.684 vs vLLM 0.763, |Δ|=0.079),
# so a 2k result here sits at the same context length as the number under dispute.
#
# Both arms load the SAME weights: vLLM's serving copy is repackaged from the same HF export the HF
# arm reads (key rename + dummy visual tower). Both render alpaca 0-shot. So the ONLY difference is
# the inference engine -- which the original olmo-core-vs-vLLM comparison could not isolate, since
# it also changed checkpoint format (distcp vs HF export).
#
#SBATCH --job-name=engine-2k
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/joblogs/engine_2k_%j.log
set -uo pipefail
source /data/prasann/node_local_env.sh
REPO=/data/prasann/repo/OLMo-core
CR=/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning
HF_CKPT=/data/prasann/ctc_suite/hf_exports_4b_chunked/ctc-4b-outlier_amzn-cmix
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-outlier_amzn-cmix
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/outlier_amzn/rung_2048.jsonl
BASE=/data/prasann/hf_models/Qwen3.5-4B-Base
OUT=/data/prasann/engine_2k; mkdir -p "$OUT"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="

# Pre-flight: prove the HF arm's interpreter can resolve THIS checkpoint's config before spending
# ~10 min on the vLLM arm. transformers 5 is a major version and corpus-reasoning's evaluate.py was
# written against 4.x, so an import- or API-level break is a live possibility -- and it should
# surface in seconds, at the top of the log, not after the other arm has finished.
$VLLM_PY - "$HF_CKPT" <<'PRE'
import sys, transformers
from transformers import AutoConfig
print(f"  HF arm interpreter: transformers {transformers.__version__}")
try:
    c = AutoConfig.from_pretrained(sys.argv[1])
    print(f"  AutoConfig OK: {type(c).__name__} model_type={c.model_type}")
except Exception as e:
    print(f"  PRE-FLIGHT FAIL: {type(e).__name__}: {str(e)[:160]}")
    sys.exit(3)
PRE
[ $? -ne 0 ] && { echo "FATAL: HF arm cannot load this checkpoint; not spending GPU on the vLLM arm"; exit 3; }

echo ""; echo "######## ARM 1: vLLM chunked (2k) ########"
P="$OUT/prefills_2048.json"
[ -f "$P" ] || $PY "$REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" \
  --tokenizer "$BASE" --task outlier_amazon --eval-data "$RUNG" \
  --max-test-samples 500 --out "$P"
export CHUNK_CACHE_IDS=1 CHUNK_MAX_NUM_SEQS=128 CHUNK_SEQ_HEADROOM=130
# Job 3442605 already produced this arm (set_f1 0.9122, 636s) before the HF arm crashed on the wrong
# interpreter. Reuse it rather than burning another 10 min re-deriving a number that is not in
# dispute -- and reuse the RESPONSES too, so the comparison is against the identical generations.
if [ -f "$OUT/grade_vllm.json" ]; then
  V_S=636
  echo "  vLLM arm: reusing job 3442605 output (wall=${V_S}s)"
  $PY -c "import json;d=json.load(open('$OUT/grade_vllm.json'));print('  ',{k:d[k] for k in d if k in ('metric_name','metric_value','parse_rate','eval_size')})"
else
  T0=$(date +%s)
  PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$REPO/debug/ctc_vllm_validation/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$P" --mode chunked --task outlier_amazon \
    --max-new-tokens 256 --max-model-len 4096 --gpu-mem-util 0.85 --out "$OUT/resp_vllm.json" >/dev/null 2>&1
  V_S=$(( $(date +%s) - T0 ))
  $PY "$REPO/debug/ctc_vllm_validation/general/grade_any.py" \
    --responses "$OUT/resp_vllm.json" --eval-data "$RUNG" --task outlier_amazon \
    --max-test-samples 500 --out "$OUT/grade_vllm.json" 2>&1 | tail -2
  echo "  vLLM wall=${V_S}s"
fi

echo ""; echo "######## ARM 2: HF chunked-flex (2k) ########"
# ⚠ THE INTERPRETER IS THE WHOLE ARM. The first attempt (jobs 3442562/3442590/3442605) ran this
# under the olmo conda env and died with "model type `qwen3_5_text` ... not recognized", which I
# misread as "transformers has no Qwen3.5". It is an ENV fact, not a library fact:
#   corpus-reasoning-olmo  transformers 4.57.6  -> no models/qwen3_5/ at all
#   ctc_vllm_venv          transformers 5.14.1  -> models/qwen3_5/ with Qwen3_5TextConfig
#                                                  (model_type = "qwen3_5_text") and Qwen3_5ForCausalLM
# i.e. transformers 5 natively declares exactly the model_type and architecture our text-only
# export_olmo_to_hf.py checkpoint writes. The VL-wrapper + key-rename + dummy-visual recipe in
# CLAUDE.md is a vLLM-0.25.1 quirk and does NOT apply here -- the HF arm reads the raw export.
#
# (The census that "confirmed" the wrong conclusion grepped models/auto/configuration_auto.py in
# both envs and got 0 hits each. transformers v5 moved the auto mapping out of that file -- 444
# lines vs 1404 -- so the probe was vacuous for v5, not negative.)
cd "$CR"; export PYTHONPATH="$CR:${PYTHONPATH:-}"
T0=$(date +%s)
PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u scripts/eval/evaluate.py \
  --backend chunked-flex --attention-pattern chunked --use-alpaca \
  --base-model "$HF_CKPT" --task outlier --max-test-samples 500 \
  --eval-data "$RUNG" --doc-start-token '<|box_start|>' \
  --output-file "$OUT/hf_2048.json" 2>&1 | tail -12
H_S=$(( $(date +%s) - T0 ))
echo "  HF wall=${H_S}s"

echo ""; echo "=== ENGINE CHECK (outlier_amzn rung_2048, ctc-4b-outlier_amzn-cmix, eval_size=500) ==="
$PY - "$OUT" "$V_S" "$H_S" <<'PY2'
import json, os, sys
out, vs, hs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def metric(p):
    try: d = json.load(open(os.path.join(out, p)))
    except Exception: return None
    def find(o, dep=0):
        if dep > 3 or not isinstance(o, dict): return None
        for k in ("set_f1", "metric_value", "f1"):
            if k in o and isinstance(o[k], (int, float)): return o[k]
        for v in o.values():
            r = find(v, dep+1)
            if r is not None: return r
        return None
    return find(d)
v, h = metric("grade_vllm.json"), metric("hf_2048.json")
print(f"  vLLM chunked   : {v if v is None else round(v,4)}   {vs}s")
print(f"  HF chunked-flex: {h if h is None else round(h,4)}   {hs}s")
if v is not None and h is not None:
    d = abs(v-h)
    print(f"  |HF - vLLM| = {d:.4f}")
    print(f"  historical olmo-core-vs-vLLM gap was 0.079; 500-example SE here ~0.02")
    print(f"  -> engines {'AGREE' if d < 0.03 else 'DISAGREE'}")
PY2
echo "=== DONE $(date '+%F %T') ==="
