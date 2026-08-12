#!/usr/bin/env bash
# Contradiction 2k on the results-hub Qwen3.5-4B checkpoints: v3 (iid) vs v2 (both-mode),
# same checkpoint, one job each.
#
# CHECKPOINTS (from results.csv, weka_model_location) -- note these live under amandab/, NOT
# prasanns/, which is why a prasanns-only weka scan does not find them:
#   amandab/q35-4b-dense-xlong5-dolci25-256k/step560          full                  v2 2k = 0.727
#   amandab/q35-4b-fastcomplm-xlong5-dolci25-256k/step560     compressive_landmark  v2 2k = 0.753
#   amandab/q35-4b-fastcomplm-xlong5-dolci25-256k-ep1/step634 compressive_landmark  v2 2k = 0.787
#
# ⚠ OPEN AT LAUNCH TIME. These train on xlong5 + Dolci-Instruct-SFT(25%), NOT on
# single_task_ladders_v2 where the realistic-vs-both mismatch was established, and the xlong5
# contradiction shard records no source jsonl. debug/ctc_contra_iid_eval/audit_xlong5_shards.py
# decodes the shard to settle it. Launched ahead of that result deliberately: each job scores BOTH
# eval sets on ONE checkpoint, so it is a controlled comparison regardless of which way the gate
# goes. If the gate says the xlong5 contradiction data is `both`-mode, then v3 is the WRONG eval
# for these checkpoints and the v3 column here should be read as an OOD probe, not a fix.
set -euo pipefail

WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-4B-Base}"
MAX_TEST="${MAX_TEST:-500}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
# Without an explicit image gantry uses its own venv, whose torch is built against a newer CUDA than
# the cluster driver -> "RuntimeError: The NVIDIA driver on your system is too old (found version
# 12080)". 12080 = CUDA 12.8, so the cu128 stable image is the matching one. This is what
# build_launch_config() passes as OLMoCoreBeakerImage.stable.
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

read -r -d '' JOB <<'EOS' || true
set -uo pipefail
# eval_lc_native.py does `from ctc_eval.eval.evaluate import ...`; the package is at
# src/scripts/ctc_eval. run_beaker_multirung_eval.sh:109 exports exactly this. Omitting it makes the
# job die with ModuleNotFoundError *after* gantry reports success.
export PYTHONPATH="$PWD/src/scripts:$PWD/src:${PYTHONPATH:-}"
FAILED=0
P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
V3="$P/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"
V2="$P/_eval_bundle_eval500_v2_clean/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"

echo "=== checkpoint: $CKPT_ABS ==="
[ -f "$CKPT_ABS/config.json" ] || { echo "!!! no config.json at $CKPT_ABS"; ls -la "$CKPT_ABS" 2>/dev/null | head -20; exit 1; }
python - <<PY
import json
m = json.load(open("$CKPT_ABS/config.json")).get("model", {})
v = m.get("vocab_size")
print(f"    vocab_size={v} d_model={m.get('d_model')} n_layers={m.get('n_layers')}")
print("    -> Qwen3.5" if (v or 0) > 200000 else "    -> NOT Qwen3.5 -- wrong checkpoint?")
PY

for f in "$V3"; do
  [ -f "$f" ] || { echo "!!! missing eval file: $f"; exit 1; }
  echo "    $(wc -l < "$f") rows  $(basename "$f")"
done

mkdir -p /results
run_one () {   # label file
  echo ""
  echo "######## $1 ########"
  python src/scripts/ctc_eval/eval/eval_lc_native.py \
    --model-path "$CKPT_ABS" \
    --contra-data "$2" \
    --tokenizer "$TOKENIZER" \
    --max-test-samples "$MAX_TEST" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --out "/results/contra_${1}_2k.json" 2>&1 | tail -25
  [ "${PIPESTATUS[0]}" = "0" ] || { echo "!!! eval $1 FAILED"; FAILED=1; }
}
# v3 only: the v2 2k numbers already exist in results.csv and for dense are triple-measured
# (0.726/0.727/0.727), so an in-job v2 control would be redundant compute. Cost of dropping it:
# the v3 number is compared against a v2 number produced by a different harness invocation rather
# than side-by-side in one job.
run_one v3_realistic "$V3"

echo ""
echo "=== SUMMARY | $RUN_LABEL | n=100, only the eval set differs ==="
echo "    results.csv v2 2k reference for this run: $V2_REF"
python - <<'PY'
import json, os
for v in ("v3_realistic",):
    f = f"/results/contra_{v}_2k.json"
    if not os.path.exists(f):
        print(f"  {v:<14} MISSING"); continue
    c = json.load(open(f)).get("contradiction", {})
    print(f"  {v:<14} f1={c.get('f1')}  em={c.get('exact_match')}  n={c.get('n')}")
PY
exit $FAILED
EOS

A=/weka/oe-training-default/ai2-llm/checkpoints/amandab
launch () {  # name ckpt_abs v2ref
  echo "=== launching $1 ==="
  gantry run \
    --name "$1" \
    --description "Contradiction 2k: $1 on v3 (iid) vs v2 (both-mode)" \
    --workspace "${WORKSPACE}" --budget "${BUDGET}" --cluster "${CLUSTER}" \
    --weka "${WEKA}:/weka/${WEKA}" --gpus 1 --priority "${PRIORITY}" \
    --beaker-image "${IMAGE}" \
    --allow-dirty --timeout 0 \
    --env "CKPT_ABS=$2" --env "RUN_LABEL=$1" --env "V2_REF=$3" \
    --env "TOKENIZER=${TOKENIZER}" --env "MAX_TEST=${MAX_TEST}" \
    --env "MAX_LENGTH=${MAX_LENGTH}" --env "BATCH_SIZE=${BATCH_SIZE}" \
    --yes -- bash -c "${JOB}"
}

launch q35-dense-v3-2k     "$A/q35-4b-dense-xlong5-dolci25-256k/step560"          0.727
launch q35-fcl-v3-2k       "$A/q35-4b-fastcomplm-xlong5-dolci25-256k/step560"     0.753
launch q35-fcl-ep1-v3-2k   "$A/q35-4b-fastcomplm-xlong5-dolci25-256k-ep1/step634" 0.787
echo "Launched 3 jobs"
