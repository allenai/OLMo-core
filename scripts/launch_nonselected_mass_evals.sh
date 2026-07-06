#!/bin/bash
#
# Sweep the compressive-landmark `nonselected_mass` (alpha) for a checkpoint, holding the hard
# top-k landmark retrieval fixed at k=64, by repeatedly invoking launch_long_context_evals.sh.
#
# `nonselected_mass` is the fraction of attention mass (in [0,1)) reserved at top-k decode for the
# landmark (compression) tokens of the NON-selected blocks; the remaining (1 - mass) goes to the
# local block + the selected (top-k) blocks. It only affects FastCompressiveLandmarkAttention
# checkpoints with top-k decode enabled. The checkpoint's baked-in default is 0.1.
#
# For each mass value m we launch the full HELMET (8k-128k) + RULER (4k-128k) suite with
# OLMO_CORE_LANDMARK_TOP_K=64 and OLMO_CORE_LANDMARK_NONSELECTED_MASS=m, tagging every run name /
# output dir / dashboard row with a per-value suffix (tk64_nsm<m>) so results never collide.
#
# Requires the harness branches that plumb landmark_nonselected_mass through:
#   * RULER : oe-eval branch amandab/ruler-memproj (set as OE_EVAL_BRANCH below)
#   * HELMET: ../../ai2-helmet with the --olmo_core_landmark_nonselected_mass flag
#
# Usage:
#   ./launch_nonselected_mass_evals.sh /weka/.../q4b-base-fast-compressive-landmark-8node/step2385
#
# Overrides (env vars):
#   NONSELECTED_MASSES  space-separated mass values (default: "0.0 0.05 0.1 0.2 0.4")
#   TOP_K               fixed top-k blocks (default: 64)
#   RULER_DASHBOARD     dashboard for the RULER rows (default: memory-LC-nsm)
#   plus anything launch_long_context_evals.sh understands (SKIP_HELMET, SKIP_RULER,
#   RULER_LENGTHS_K, HELMET_MAX_LENGTH, CLUSTER, PRIORITY, ...).

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <weka_checkpoint_path>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOP_K="${TOP_K:-64}"
NONSELECTED_MASSES=( ${NONSELECTED_MASSES:-0.0 0.05 0.1 0.2 0.4} )

# RULER's olmo_core model-arg landmark_nonselected_mass is only recognized on this oe-eval branch
# (same branch that carries landmark_top_k_blocks). HELMET reads ../../ai2-helmet directly.
export OE_EVAL_BRANCH="${OE_EVAL_BRANCH:-amandab/ruler-memproj}"

for m in "${NONSELECTED_MASSES[@]}"; do
  # Filesystem/Beaker-safe tag: 0.05 -> tk64_nsm0p05. Used as the HELMET and RULER name suffix so
  # each value lands in a distinct OUTPUT_DIR / dashboard row (a distinct HELMET suffix also avoids
  # reusing a stale cached HELMET result).
  tag="tk${TOP_K}_nsm${m//./p}"

  echo "============================================================================"
  echo "==> nonselected_mass=${m} (top_k=${TOP_K})  suffix=${tag}"
  echo "============================================================================"

  OLMO_CORE_LANDMARK_TOP_K="${TOP_K}" \
  OLMO_CORE_LANDMARK_NONSELECTED_MASS="${m}" \
  HELMET_NAME_SUFFIX="${tag}" \
  RULER_NAME_SUFFIX="${tag}" \
  RULER_DASHBOARD="${RULER_DASHBOARD:-memory-LC-nsm}" \
    bash "${SCRIPT_DIR}/launch_long_context_evals.sh" "${MODEL_PATH}"
done

echo "All nonselected_mass sweep jobs submitted (masses: ${NONSELECTED_MASSES[*]}, top_k=${TOP_K})."
