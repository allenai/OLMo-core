#!/bin/bash
# THREE-WAY ENGINE PARITY on 10 examples: vLLM chunked vs olmo-core native vs HF chunked-flex.
#
# WHY 10 EXAMPLES IS THE RIGHT SIZE, NOT A COMPROMISE. The question "do these engines run the same
# model?" is answered by comparing GENERATED STRINGS, not metrics. Exact string match is a per-token
# test: a 10-example set at ~40 generated tokens each is ~400 independent agreement checks, and two
# engines that genuinely share weights+mask+prompt produce byte-identical greedy output. A metric on
# 500 examples is a far WEAKER instrument for this -- it compresses all of that to one scalar with a
# +/-0.02 SE, which is exactly why the historical |Δ|=0.079 row could never say whether the engines
# disagreed a little everywhere or a lot on a few examples.
#
# So the acceptance test here is: identical strings -> engines are the same. If strings differ, the
# script reports WHERE the first token diverges per example, which localizes it (prompt rendering
# vs. sampling vs. numerics) in a way no aggregate score can.
#
# GREEDY IS MANDATORY. Any temperature > 0 makes string comparison meaningless -- two identical
# engines would differ by RNG alone. All three arms are pinned to greedy/temp=0.
#
# ⚠ INTERPRETERS DIFFER PER ARM AND THAT IS THE POINT:
#   vLLM arm   -> ctc_vllm_venv (vllm 0.25.1)
#   HF arm     -> ctc_vllm_venv TOO -- transformers 5.14.1 is the only env here with models/qwen3_5/
#                 (Qwen3_5TextConfig, model_type "qwen3_5_text" == what our export writes).
#                 The olmo conda env is transformers 4.57.6 and has NO qwen3_5 at all; running the
#                 HF arm there is what produced three "not recognized" crashes.
#   native arm -> olmo conda env (it imports olmo_core, which the vllm venv does not have)
#
#   sbatch debug/chunked_eval_speedup/engine_parity_10.sbatch
#SBATCH --job-name=parity10
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/parity10_%j.log

set -uo pipefail
source /data/prasann/node_local_env.sh
REPO=/data/prasann/repo/OLMo-core
CR=/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning
HF_CKPT=/data/prasann/ctc_suite/hf_exports_4b_chunked/ctc-4b-outlier_amzn-cmix
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-outlier_amzn-cmix
STEP=/data/prasann/ctc_suite/ckpts_4b_chunked/ctc-4b-outlier_amzn-cmix
RUNG=/data/prasann/ctc_suite_staged/eval_rungs/outlier_amzn/rung_2048.jsonl
BASE=/data/prasann/hf_models/Qwen3.5-4B-Base
OUT=/data/prasann/parity10; mkdir -p "$OUT"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
N=10

echo "=== HOST=$(hostname) START=$(date '+%F %T') N=$N ==="

# ---------- shared prefills: all three arms consume the SAME rendered prompts ----------
# Rendering per-arm would let a prompt difference masquerade as an engine difference -- the exact
# confound that made the original olmo-core-vs-vLLM row unattributable.
P="$OUT/prefills.json"
$PY "$REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" \
  --tokenizer "$BASE" --task outlier_amazon --eval-data "$RUNG" \
  --max-test-samples "$N" --out "$P" || { echo "FATAL: prefill build failed"; exit 2; }

# ---------- ARM 1: vLLM chunked ----------
echo ""; echo "######## ARM 1: vLLM chunked ########"
export CHUNK_CACHE_IDS=1
T0=$(date +%s)
PATH=$CUDA_HOME/bin:$PATH CUDA_VISIBLE_DEVICES=0 $VLLM_PY -u "$REPO/debug/ctc_vllm_validation/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$P" --mode chunked --task outlier_amazon \
  --max-new-tokens 256 --max-model-len 4096 --gpu-mem-util 0.85 \
  --out "$OUT/resp_vllm.json" 2>&1 | tail -5
echo "  vLLM wall=$(( $(date +%s) - T0 ))s"

# ---------- ARM 2: olmo-core native ----------
echo ""; echo "######## ARM 2: olmo-core native (variant=dense == chunked mask) ########"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=1 $PY -u "$REPO/src/corpus_reasoning/eval/eval_lc_native_docchunk.py" \
  --variant dense --model-path "$STEP" --out "$OUT/native.json" \
  --tokenizer "$BASE" --max-length 4096 --max-test-samples "$N" \
  --data "$RUNG" --task outlier \
  --doc-start-id 248049 --doc-end-id 248050 --eos-token-id 248044 2>&1 | tail -8
echo "  native wall=$(( $(date +%s) - T0 ))s"

# ---------- ARM 3: HF chunked-flex ----------
echo ""; echo "######## ARM 3: HF chunked-flex ########"
cd "$CR"; export PYTHONPATH="$CR:${PYTHONPATH:-}"
T0=$(date +%s)
PATH=$CUDA_HOME/bin:$PATH CUDA_VISIBLE_DEVICES=0 $VLLM_PY -u scripts/eval/evaluate.py \
  --backend chunked-flex --attention-pattern chunked --use-alpaca \
  --base-model "$HF_CKPT" --task outlier --max-test-samples "$N" \
  --eval-data "$RUNG" --doc-start-token '<|box_start|>' \
  --output-file "$OUT/hf.json" 2>&1 | tail -10
echo "  HF wall=$(( $(date +%s) - T0 ))s"

# ---------- compare generated strings ----------
echo ""; echo "=== STRING-LEVEL PARITY (outlier_amzn rung_2048, ctc-4b-outlier_amzn-cmix) ==="
$PY - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]

def load(fn):
    """Pull the list of generated strings out of whatever shape each arm wrote."""
    p = os.path.join(out, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # Each arm nests differently; search for the first list of dicts carrying a text-ish field,
    # rather than hardcoding three schemas that drift independently.
    def dig(o, depth=0):
        if depth > 4:
            return None
        if isinstance(o, list) and o and isinstance(o[0], dict):
            for k in ("generation", "response", "output", "text", "prediction", "completion"):
                if k in o[0]:
                    return [str(r.get(k, "")) for r in o]
        if isinstance(o, dict):
            for v in o.values():
                r = dig(v, depth + 1)
                if r:
                    return r
        return None
    return dig(d)

arms = {"vLLM": load("resp_vllm.json"), "native": load("native.json"), "HF": load("hf.json")}
for k, v in arms.items():
    print(f"  {k:8s}: {'MISSING/unparsed' if v is None else f'{len(v)} generations'}")
live = {k: v for k, v in arms.items() if v}
names = list(live)
if len(names) < 2:
    print("  -> fewer than two arms produced output; nothing to compare")
    raise SystemExit(0)

for i, a in enumerate(names):
    for b in names[i + 1:]:
        ga, gb = live[a], live[b]
        n = min(len(ga), len(gb))
        same = sum(1 for j in range(n) if ga[j].strip() == gb[j].strip())
        print(f"\n  {a} vs {b}: {same}/{n} generations byte-identical")
        if same == n:
            print("    -> IDENTICAL. These engines run the same model.")
            continue
        # Localize the first divergence: a shared prefix that ends mid-generation points at
        # numerics/kernel; divergence at token 0 points at prompt or sampling config.
        for j in range(n):
            if ga[j].strip() != gb[j].strip():
                x, y = ga[j].strip(), gb[j].strip()
                cp = next((t for t in range(min(len(x), len(y))) if x[t] != y[t]), min(len(x), len(y)))
                print(f"    first mismatch: example {j}, diverges at char {cp}")
                print(f"      {a:8s}: {x[:160]!r}")
                print(f"      {b:8s}: {y[:160]!r}")
                break
PY
echo "=== DONE $(date '+%F %T') ==="
