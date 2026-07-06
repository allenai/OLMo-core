#!/bin/bash
# Build / publish the corpus-reasoning visualization website FROM the OLMo-core repo.
#
# The viz pipeline lives in the corpus-reasoning checkout (./corpus-reasoning/viz),
# a standalone clone (no longer an OLMo-core submodule). This thin wrapper just
# delegates to it, so you can drive the same website from either repo:
#
#   # from OLMo-core:
#   bash viz.sh                 # build corpus-reasoning/viz/outputs/index.html
#   bash viz.sh --update-demo   # also refresh the committed demo snapshot
#   bash viz.sh --publish       # build + S3-sync (set VIZ_S3_DEST first)
#
#   # from corpus-reasoning:
#   bash viz/run.sh [same flags]
#
# Experiment configs are read from THIS OLMo-core checkout (OLMO_CORE_ROOT); task
# data from CR_DATA_ROOT (default /scratch/users/prasann/corpus-reasoning/data).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB="$ROOT/corpus-reasoning"

# Make sure the corpus-reasoning checkout is present. It used to be a git submodule
# (private SSH URL, which broke gantry/Beaker launches), so it's now a plain clone.
if [[ ! -f "$SUB/viz/run.sh" ]]; then
  echo "[viz.sh] corpus-reasoning not found at '$SUB'; cloning..."
  git clone git@github.com:PrasannS/corpus-reasoning.git "$SUB"
fi

# Pin experiment-config source to this OLMo-core checkout.
export OLMO_CORE_ROOT="${OLMO_CORE_ROOT:-$ROOT}"

exec bash "$SUB/viz/run.sh" "$@"
