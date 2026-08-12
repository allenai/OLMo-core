#!/usr/bin/env bash
# Sanity-check that contradiction is fixed: Qwen3.5-4B on the v3 (iid) 2k rung vs the v2 (both-mode)
# 2k rung, same checkpoint, one job.
#
# Uses the PUSHED --ladder-version v3 path end-to-end (commit 79b969c01), not the --contra-data
# workaround the earlier hub job needed -- so this also validates the code that just went in:
# eval_lc_native.py's E5_CONTRA redirect and run_beaker_multirung_eval.sh's v3 gate.
#
# CHECKPOINT. ctc-s5-contra-full-4b is the CTC-suite Qwen3.5-4B contradiction model: config.json
# reports vocab_size 248320 (Qwen3.5, vs Qwen3's ~151936), GDN blocks present, d_model 2560 /
# 32 layers. This is the SAME model family as the paper. It is the checkpoint that scored 0.9895
# on the CTC iid ladder at n=56 and 0.843 on the both-mode ladder at n=44 -- via vLLM. Running it
# through the NATIVE evaluator here is an independent harness, so agreement also cross-checks the
# vLLM path.
#
# EXPECTED. v3 2k (n=100) ~0.98 (the local vLLM run on these exact files gave 0.9864); v2 2k
# (n=100, both-mode) ~0.83. A v3 number near the v2 one would mean the ladder wiring is wrong, not
# that the finding failed -- check EVAL500_CONTRA_ROOT resolution first.
set -euo pipefail

NAME="${NAME:-contra-v3-2k-qwen35}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
RUN="${RUN:-ctc-s5-contra-full-4b}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-4B-Base}"
MAX_TEST="${MAX_TEST:-500}"
MAX_LENGTH="${MAX_LENGTH:-16384}"   # n=100 measures ~4.4k tokens
BATCH_SIZE="${BATCH_SIZE:-2}"

read -r -d '' JOB <<'EOS' || true
set -uo pipefail
# `eval_lc_native.py` does `from ctc_eval.eval.evaluate import ...`; the package lives at
# src/scripts/ctc_eval, so it must be on PYTHONPATH. run_beaker_multirung_eval.sh:109 does exactly
# this -- omitting it is why the first run died with ModuleNotFoundError AFTER gantry reported success.
export PYTHONPATH="$PWD/src/scripts:$PWD/src:${PYTHONPATH:-}"
FAILED=0
P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns

echo "=== locating $RUN on weka ==="
# The run dir may hold step*/ subdirs or be flat (config.json + model_and_optim at the top).
CKPT=""
if [ -f "$P/$RUN/config.json" ] && [ -d "$P/$RUN/model_and_optim" ]; then
  CKPT="$P/$RUN"
else
  CKPT="$(ls -d "$P/$RUN"/step* 2>/dev/null | sort -V | tail -1)"
fi
if [ -z "$CKPT" ]; then
  echo "    '$RUN' not found; candidate contra/ctc dirs on weka:"
  ls -d "$P"/*contra* "$P"/*ctc* 2>/dev/null | head -30
  for c in $(ls -d "$P"/*contra*4b* "$P"/*ctc*contra* 2>/dev/null); do
    if [ -f "$c/config.json" ]; then CKPT="$c"; break; fi
    s="$(ls -d "$c"/step* 2>/dev/null | sort -V | tail -1)"
    if [ -n "$s" ] && [ -f "$s/config.json" ]; then CKPT="$s"; break; fi
  done
  [ -n "$CKPT" ] && echo "    auto-selected $CKPT"
fi
[ -n "$CKPT" ] && [ -f "$CKPT/config.json" ] || {
  echo "!!! no usable checkpoint under $P/$RUN"; ls -la "$P/$RUN" 2>/dev/null | head -20; exit 1; }
echo "    using $CKPT"
python - <<PY
import json
d=json.load(open("$CKPT/config.json"))
m=d.get("model",{})
print(f\"    vocab_size={m.get('vocab_size')} d_model={m.get('d_model')} n_layers={m.get('n_layers')}\")
print('    -> Qwen3.5' if (m.get('vocab_size') or 0) > 200000 else '    -> NOT Qwen3.5 (vocab too small) -- check the run name')
PY

V3="$P/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"
V2="$P/_eval_bundle_eval500_v2_clean/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"
for f in "$V3" "$V2"; do
  [ -f "$f" ] || { echo "!!! missing eval file: $f (did the weka staging job land?)"; exit 1; }
  echo "    $(wc -l < "$f") rows  $f"
done

mkdir -p /results
run_ver () {  # v2 | v3
  echo ""
  echo "######## ladder-version $1, contradiction 2k ########"
  python src/scripts/ctc_eval/eval/eval_lc_native.py \
    --model-path "$CKPT" \
    --ladder --ladder-version "$1" --ladder-tasks contradiction --ladder-rungs 2k \
    --tokenizer "$TOKENIZER" \
    --max-test-samples "$MAX_TEST" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --out "/results/contra_$1_2k.json" 2>&1 | tail -30
  [ "${PIPESTATUS[0]}" = "0" ] || { echo "!!! eval $1 FAILED"; FAILED=1; }
}
run_ver v3
run_ver v2

echo ""
echo "=== SUMMARY (same Qwen3.5 checkpoint, same n=100 rung, only the eval set differs) ==="
echo "    local vLLM reference on these exact files: v3 n=100 -> 0.9864 | both-mode n=92 -> 0.803"
python - <<'PY'
import json, glob, os
for v in ("v3", "v2"):
    f = f"/results/contra_{v}_2k.json"
    if not os.path.exists(f):
        print(f"  {v}: MISSING"); continue
    d = json.load(open(f))
    hits = {k: val for k, val in d.items() if "contra" in k.lower()}
    print(f"  {v}: {json.dumps(hits)[:400]}")
PY
exit $FAILED
EOS

echo "=== launching $NAME on $CLUSTER (run=$RUN, tokenizer=$TOKENIZER) ==="
gantry run \
  --name "${NAME}" \
  --description "Contradiction 2k: Qwen3.5-4B on v3 (iid) vs v2 (both-mode)" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  --cluster "${CLUSTER}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --gpus 1 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --timeout 0 \
  --env "RUN=${RUN}" \
  --env "TOKENIZER=${TOKENIZER}" \
  --env "MAX_TEST=${MAX_TEST}" \
  --env "MAX_LENGTH=${MAX_LENGTH}" \
  --env "BATCH_SIZE=${BATCH_SIZE}" \
  --yes \
  -- bash -c "${JOB}"

echo "Launched ${NAME}"
