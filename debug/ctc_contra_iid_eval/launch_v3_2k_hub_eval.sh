#!/usr/bin/env bash
# 2k contradiction eval of the RESULTS-HUB checkpoint on the v3 (iid) eval set, with the v2
# (both-mode) file as an in-job control.
#
# WHY A HAND-ROLLED GANTRY CALL INSTEAD OF launch_beaker_multirung_eval.sh.
# Gantry runs PUSHED code, and the `--ladder-version v3` support is still uncommitted. The pushed
# on-node runner (run_beaker_multirung_eval.sh:61) hard-rejects any LADDER_VERSION that is not
# v2/fast, so the ladder route would need a commit+push. It does not need one: eval_lc_native.py
# has always had a non-ladder single-file path (`if not args.ladder:` -> `--contra-data`), which is
# exactly a one-rung eval. So this runs entirely on already-pushed code.
#
# WHAT IT ANSWERS. Every iid number so far is from ctc-4b-contradiction-full (Qwen3.5-4B, CTC
# suite). The hub's 5-task runs are Qwen3-4B and live only on weka. This scores THAT checkpoint,
# on weka, where it is mounted -- the comparison the local jobs could not make.
#
# EXPECTED. Hub dense-5task on the v2 both-mode 2k rung is 0.829 (results.csv). If the mismatch
# story holds across model families, v3 should come in far higher (~0.98 is what the Qwen3.5 CTC
# checkpoint gave at n=92/100). Both files are scored in ONE job on ONE checkpoint, so the
# comparison is internal and controlled -- no cross-run confound.
# A LOW v3 number would be the interesting result: it would mean the finding is Qwen3.5-specific.
#
# Prereq: the v3 bundle must be on weka (stage-eval500-v3-weka gantry job). Without it the v3 path
# will not exist and the job fails loudly rather than silently scoring nothing.
set -euo pipefail

NAME="${NAME:-contra-v3-2k-hub}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"          # standing directive: never below urgent
RUN="${RUN:-q4b-dense-5task-32k-nocpt-fixdata}"
MAX_TEST="${MAX_TEST:-500}"
MAX_LENGTH="${MAX_LENGTH:-16384}"       # n=100 measures ~4.4k tokens; ample headroom
BATCH_SIZE="${BATCH_SIZE:-2}"

read -r -d '' JOB <<'EOS' || true
set -uo pipefail
P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
V3="$P/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"
V2="$P/_eval_bundle_eval500_v2_clean/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"

echo "=== checkpoint discovery for $RUN ==="
CKPT="$(ls -d "$P/$RUN"/step* 2>/dev/null | sort -V | tail -1)"
[ -n "$CKPT" ] || { echo "!!! no step dir under $P/$RUN"; ls -la "$P/$RUN" 2>/dev/null | head; exit 1; }
echo "    using $CKPT"
ls "$CKPT" | head

for f in "$V3" "$V2"; do
  [ -f "$f" ] || { echo "!!! missing eval file: $f"; exit 1; }
  echo "    $(wc -l < "$f") rows  $f"
done

mkdir -p /results
run_one () {   # label file
  echo ""
  echo "######## $1 : $2 ########"
  python src/scripts/ctc_eval/eval/eval_lc_native.py \
    --model-path "$CKPT" \
    --contra-data "$2" \
    --tokenizer Qwen/Qwen3-4B \
    --max-test-samples "$MAX_TEST" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --out "/results/contra_${1}_2k.json" 2>&1 | tail -25
  python - <<PY
import json
try:
    d = json.load(open("/results/contra_${1}_2k.json"))
    c = d.get("contradiction", {})
    print(f"RESULT ${1}: f1={c.get('f1')} em={c.get('exact_match')} n={c.get('n', c.get('eval_size'))}")
except Exception as e:
    print(f"RESULT ${1}: could not read result ({e})")
PY
}

run_one v3_realistic "$V3"
run_one v2_both      "$V2"

echo ""
echo "=== SUMMARY (same checkpoint, same rung n=100, only the eval set differs) ==="
echo "    results.csv reference for this run on v2 2k: f1 0.829"
for l in v3_realistic v2_both; do
  python - <<PY
import json
try:
    d=json.load(open("/results/contra_${l}_2k.json")); c=d.get("contradiction",{})
    print(f"  {'$l':<14} f1={c.get('f1')}  em={c.get('exact_match')}")
except Exception as e:
    print(f"  {'$l':<14} MISSING ({e})")
PY
done
EOS

echo "=== launching $NAME on $CLUSTER (run=$RUN) ==="
gantry run \
  --name "${NAME}" \
  --description "Contradiction 2k: hub checkpoint on v3 (iid) vs v2 (both-mode) eval" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  --cluster "${CLUSTER}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --gpus 1 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --timeout 0 \
  --env "RUN=${RUN}" \
  --env "MAX_TEST=${MAX_TEST}" \
  --env "MAX_LENGTH=${MAX_LENGTH}" \
  --env "BATCH_SIZE=${BATCH_SIZE}" \
  --yes \
  -- bash -c "${JOB}"

echo "Launched ${NAME}"
