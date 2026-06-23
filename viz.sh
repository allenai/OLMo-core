#!/bin/bash
# Build / publish the corpus-reasoning visualization website FROM the OLMo-core repo.
#
# The viz pipeline lives in the corpus-reasoning submodule (./corpus-reasoning/viz).
# This thin wrapper just delegates to it, so you can drive the same website from
# either repo:
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

# Make sure the submodule is checked out (and current with the pinned commit).
if [[ ! -f "$SUB/viz/run.sh" ]]; then
  echo "[viz.sh] initializing corpus-reasoning submodule..."
  git -C "$ROOT" submodule update --init "$SUB"
fi

# Pin experiment-config source to this OLMo-core checkout.
export OLMO_CORE_ROOT="${OLMO_CORE_ROOT:-$ROOT}"

exec bash "$SUB/viz/run.sh" "$@"
