#!/usr/bin/env bash
# Verify + visualize the SummTokenSFT attention mask on REAL Qwen3.5-tokenized context windows,
# on one Beaker CPU node via gantry (weka-mounted).
#
# Wraps verify_summary_mask_on_real_data.py. For a handful of real document-chunked windows it runs
# the production emitter and role builder, then asserts -- at probe positions spread through the
# context -- that:
#   * a DOCUMENT token sees its own document + the instruction + earlier summary runs, and NO other
#     document's content;
#   * a SUMMARY token sees its own document and earlier summary runs (the relay);
#   * a QUERY/ANSWER token sees the instruction and every summary run, and NO raw document content --
#     while on a causal-arm example the same position sees everything.
# It also prints a picture of the mask and the realized block sparsity. Exits non-zero if any window
# fails, so this is usable as a gate before launching an arm.
#
# CPU only: nothing here needs a GPU (no model is built, and no (T,T) mask is ever materialized), so
# it also does NOT need triton -- unlike the marker repair.
#
# Usage:
#   src/scripts/train/memexpress/sft_summtoken/verify_summary_mask_gantry.sh
#   SHARDS='/weka/oe-training-default/ai2-llm/checkpoints/prasanns/docchunk_5task_fixed40k/contra_dense/token_ids_part_*.npy' \
#     MARKER_SET=qwen3 src/scripts/train/memexpress/sft_summtoken/verify_summary_mask_gantry.sh
#
# Overridable env: CLUSTER WORKSPACE BUDGET WEKA PRIORITY CPUS NAME IMAGE SHARDS MARKER_SET
#                  N_WINDOWS MIN_DOCS N_SUMMARY PAD_TO GRID TOKENIZER
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
# CLAUDE.md: every Beaker job launches at urgent. (ai2/holmes is the one documented exception.)
PRIORITY="${PRIORITY:-urgent}"
CPUS="${CPUS:-4}"
NAME="${NAME:-summtoken-mask-verify}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

# Default to the Qwen3.5 box-marker (document-chunked) shards. These are REAL Qwen3.5-tokenized
# windows with real document structure -- the summary runs are inserted here by the production
# emitter, because the summary-token shards themselves do not exist until the converter is run.
SHARDS="${SHARDS:-/weka/${WEKA}/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/shards_chunked/contradiction_train/token_ids_part_*.npy}"
MARKER_SET="${MARKER_SET:-qwen3_5}"
N_WINDOWS="${N_WINDOWS:-4}"
MIN_DOCS="${MIN_DOCS:-6}"
N_SUMMARY="${N_SUMMARY:-5}"
PAD_TO="${PAD_TO:-0}"      # 0 = round up to a multiple of 128 (exercises the pad tail)
GRID="${GRID:-48}"
# Decoding probe context is a nicety; the tokenizer is read from weka, not huggingface.co.
TOKENIZER="${TOKENIZER:-/weka/${WEKA}/ai2-llm/tokenizers/Qwen3.5-0.8B}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

read -r -d '' REMOTE <<REMOTE_EOF || true
set -uo pipefail
export PYTHONPATH="\$(pwd)/src:\${PYTHONPATH:-}"
V=src/scripts/train/memexpress/sft_summtoken/verify_summary_mask_on_real_data.py

echo "=== shards: ${SHARDS}"
_dir=\$(dirname "${SHARDS}")
[ -d "\$_dir" ] || echo "  WARNING: \$_dir does not exist"
ls -la "\$_dir" 2>/dev/null | head -6 || true
echo "--- parent, in case the glob is one level off ---"
ls -la "\$(dirname \$_dir)" 2>/dev/null | head -8 || true

python \$V \
  --shards "${SHARDS}" \
  --marker-set "${MARKER_SET}" \
  --n-windows ${N_WINDOWS} \
  --min-docs ${MIN_DOCS} \
  --num-summary-tokens ${N_SUMMARY} \
  --pad-to ${PAD_TO} \
  --grid ${GRID} \
  --tokenizer "${TOKENIZER}" \
  --report-json /results/summary_mask_report.json
rc=\$?
echo "=== verify rc=\$rc ==="

# A verifier that cannot fail proves nothing. Re-run with the relay severed
# (--summary-visible-tokens 0): the query must then be UNABLE to see the summary runs, so the checks
# are expected to FAIL here. If this run passes, the assertions are not actually binding.
echo
echo "=== negative control: --summary-visible-tokens 0 (expected to FAIL) ==="
python \$V \
  --shards "${SHARDS}" --marker-set "${MARKER_SET}" \
  --n-windows 1 --min-docs ${MIN_DOCS} --num-summary-tokens ${N_SUMMARY} \
  --summary-visible-tokens 0 --grid 16 > /tmp/negctl.log 2>&1
neg=\$?
grep -E "FAIL:|windows OK" /tmp/negctl.log | head -8
if [ \$neg -eq 0 ]; then
  echo "!!! NEGATIVE CONTROL PASSED -- the assertions are not binding. Treat the main run as void."
  exit 3
fi
# A non-zero exit is NOT enough: a crash (missing module, bad path) also exits non-zero and would
# masquerade as "the checks have teeth". Require the EXPECTED failure to actually appear.
if ! grep -q "cannot see every summary token" /tmp/negctl.log; then
  echo "!!! NEGATIVE CONTROL failed for the WRONG REASON (no summary-visibility assertion fired)."
  echo "--- its log ---"; tail -20 /tmp/negctl.log
  exit 4
fi
echo "negative control failed for the right reason: the checks have teeth."

exit \$rc
REMOTE_EOF

gantry run \
  --name "${NAME}" \
  --task-name "${NAME}" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --priority "${PRIORITY}" \
  --beaker-image "${IMAGE}" \
  --cpus "${CPUS}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --python-manager conda \
  --system-python \
  --install "pip install -e . && pip install dataclass-extensions" \
  --allow-dirty \
  --yes \
  --show-logs \
  -- bash -c "${REMOTE}"
