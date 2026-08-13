#!/usr/bin/env bash
# Repair the reserved marker embedding rows of a base checkpoint on Beaker, writing the fixed copy
# to weka.
#
# Why this is a gantry job rather than a laptop script: the repair has to BUILD the model to load the
# checkpoint into, and Qwen3.5 is a GDN/attention hybrid whose kernels need triton. The audit half
# needs none of that (it reads the embedding matrix straight out of the distcp with load_keys), so
# this job runs the audit before AND after and writes both reports.
#
# ⚠ Gate every training launch on the AFTER report's "audit_pass": true. An untrained <|summ|> row is
# bit-identical to its neighbours and far below a trained row's norm, and RMSNorm amplifies a
# low-norm row into a full-strength meaningless vector at every occurrence -- which flatlines
# training at CE ~0.79 for every mask including plain causal, and reads as "the mask is too
# restrictive" rather than as an embedding bug.
#
# Usage:
#   src/scripts/data/fix_marker_embeddings_gantry.sh
#
# Overridable env: CLUSTER WORKSPACE BUDGET WEKA PRIORITY GPUS NAME IMAGE BASE OUT FAMILY
#                  MODEL_SIZE MARKER_SET TOKENIZER
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2,ai2/neptune-cirrascale,ai2/saturn-cirrascale}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
# One GPU: not for compute, but so triton imports cleanly for the hybrid's GDN blocks.
GPUS="${GPUS:-1}"
NAME="${NAME:-summtoken-repair-base}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

BASE="${BASE:-/weka/${WEKA}/ai2-llm/checkpoints/q35-4b-dense-256k-fix/step2385/model_and_optim}"
OUT="${OUT:-/weka/${WEKA}/ai2-llm/checkpoints/amandab/q35-4b-dense-256k-summfix}"
FAMILY="${FAMILY:-qwen3_5}"
MODEL_SIZE="${MODEL_SIZE:-4B}"
# doc_start/doc_end are used by this layout too (the documents are box-wrapped), so all four rows.
MARKER_SET="${MARKER_SET:-doc_start,doc_end,summary,pad}"
# Prefer the weka-staged tokenizer; fall back to the Hub id if it is not there.
TOKENIZER="${TOKENIZER:-}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

read -r -d '' REMOTE <<REMOTE_EOF || true
set -uo pipefail
export PYTHONPATH="\$(pwd)/src:\${PYTHONPATH:-}"
F=src/scripts/data/fix_marker_embeddings.py

echo "=== base: ${BASE}"
ls -la "${BASE}" 2>/dev/null | head -4 || { echo "BASE missing"; exit 2; }

TOK="${TOKENIZER}"
if [ -z "\$TOK" ]; then
  for cand in /weka/${WEKA}/ai2-llm/tokenizers/Qwen3.5-0.8B \
              /weka/${WEKA}/ai2-llm/tokenizers/Qwen3.5-0.8B-Base; do
    [ -d "\$cand" ] && { TOK="\$cand"; break; }
  done
fi
[ -z "\$TOK" ] && TOK="Qwen/Qwen3.5-0.8B"
echo "=== tokenizer: \$TOK"

echo
echo "=== AUDIT BEFORE (expect cosine ~1.0 and a low norm: the untrained-row signature) ==="
python \$F --audit-only --base "${BASE}" --family "${FAMILY}" \
    --marker-set "${MARKER_SET}" --audit-json /results/audit_before.json
echo "  rc=\$?"

echo
echo "=== REPAIR ==="
python \$F --base "${BASE}" --out "${OUT}" --family "${FAMILY}" \
    --model-size "${MODEL_SIZE}" --marker-set "${MARKER_SET}" \
    --tokenizer "\$TOK" --audit-json /results/audit_after.json
rc=\$?
echo "  rc=\$rc"
[ \$rc -ne 0 ] && exit \$rc

echo
echo "=== AUDIT AFTER, re-read from the WRITTEN copy (not from memory) ==="
python \$F --audit-only --base "${OUT}/model_and_optim" --family "${FAMILY}" \
    --marker-set "${MARKER_SET}" --audit-json /results/audit_written.json
rc=\$?
python - <<'PYEOF'
import json, sys
r = json.load(open("/results/audit_written.json"))
print("cosine_gate_pass:", r["cosine_gate_pass"], " norm_gate_pass:", r["norm_gate_pass"])
if not r["audit_pass"]:
    print("!!! the WRITTEN checkpoint does not pass its own gates -- do NOT train from it")
    sys.exit(1)
print("written checkpoint passes both gates; safe to train from")
PYEOF
exit \$?
REMOTE_EOF

gantry run \
  --name "${NAME}" \
  --task-name "${NAME}" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --priority "${PRIORITY}" \
  --beaker-image "${IMAGE}" \
  --gpus "${GPUS}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --python-manager conda \
  --system-python \
  --install "pip install -e . && pip install dataclass-extensions" \
  --allow-dirty \
  --yes \
  -- bash -c "${REMOTE}"
