#!/usr/bin/env bash
# Build the SummTokenSFT shards on one Beaker CPU node, writing straight to weka.
#
# Inserts a <|summ|> run after every context document of the existing Qwen3.5 document-chunked
# (box-marker) shards, through the production emitter. No re-tokenization and no source JSONL: the
# documents stay byte-identical to the doc-chunked arms, which is what keeps the families comparable.
#
# Discovers the per-task subdirectories under SRC_ROOT rather than hardcoding them, then verifies the
# result in the same job -- a shard that does not verify is worse than no shard.
#
# Usage:
#   src/scripts/data/build_summary_token_shards_gantry.sh
#
# Overridable env: CLUSTER WORKSPACE BUDGET WEKA PRIORITY CPUS NAME IMAGE SRC_ROOT OUT_ROOT
#                  N_SUMMARY MARKER_SET MAX_LEN
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2,ai2/neptune-cirrascale,ai2/saturn-cirrascale}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
CPUS="${CPUS:-8}"
NAME="${NAME:-summtoken-build-shards}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

SRC_ROOT="${SRC_ROOT:-/weka/${WEKA}/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/shards_chunked}"
OUT_ROOT="${OUT_ROOT:-/weka/${WEKA}/ai2-llm/checkpoints/amandab/summtoken_5task_xlong}"
N_SUMMARY="${N_SUMMARY:-5}"
MARKER_SET="${MARKER_SET:-qwen3_5}"
# Drop examples that no longer fit the training window AFTER insertion -- visibly and counted here,
# rather than silently at load time.
MAX_LEN="${MAX_LEN:-262144}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

read -r -d '' REMOTE <<REMOTE_EOF || true
set -uo pipefail
export PYTHONPATH="\$(pwd)/src:\${PYTHONPATH:-}"
B=src/scripts/data/build_summary_token_shards.py
V=src/scripts/train/memexpress/sft_summtoken/verify_summary_mask_on_real_data.py

echo "=== source tree: ${SRC_ROOT}"
ls -la "${SRC_ROOT}" || { echo "SRC_ROOT missing"; exit 2; }

shopt -s nullglob
built=0
for d in "${SRC_ROOT}"/*/ ; do
  task=\$(basename "\$d")
  # Name the output the way the launcher's _task_source expects: <short>_summary.
  short=\$(echo "\$task" | sed -E 's/_train\$//; s/^contradiction\$/contra/; s/^retrieval\$/nq/')
  out="${OUT_ROOT}/\${short}_summary"
  echo
  echo "=== \$task -> \$out  \$(date '+%T') ==="
  python \$B --in-dir "\$d" --out-dir "\$out" \
      --marker-set "${MARKER_SET}" --num-summary-tokens ${N_SUMMARY} --max-len ${MAX_LEN}
  rc=\$?
  echo "  rc=\$rc"
  [ \$rc -eq 0 ] && built=\$((built+1))
done
echo
echo "=== built \$built task trees ==="
ls -la "${OUT_ROOT}" || true

echo
echo "=== VERIFY each built tree (a shard that does not verify is worse than no shard) ==="
fail=0
for d in "${OUT_ROOT}"/*/ ; do
  echo "--- \$(basename \$d) ---"
  python \$V --shards "\$d/token_ids_part_*.npy" --marker-set "${MARKER_SET}" \
      --n-windows 2 --min-docs 4 --num-summary-tokens ${N_SUMMARY} --grid 24 \
      2>&1 | grep -E "already contain|roles:|other docs|query start|answer end|analytic|==>|windows OK"
  [ \${PIPESTATUS[0]} -ne 0 ] && fail=1
done
[ \$fail -ne 0 ] && { echo "!!! at least one built tree FAILED verification"; exit 1; }
echo "=== all built trees verified ==="
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
  -- bash -c "${REMOTE}"
