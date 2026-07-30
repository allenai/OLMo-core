#!/bin/bash
# Build the corpus-reasoning visualization site end-to-end.
#
# Reads task data from CR_DATA_ROOT (default /scratch/users/prasann/corpus-reasoning/data)
# and experiment configs from OLMO_CORE_ROOT (default: the OLMo-core checkout that
# owns this submodule). Writes a self-contained outputs/index.html.
#
# Usage:
#   bash viz/run.sh                 # build outputs/index.html
#   bash viz/run.sh --update-demo   # also refresh the committed demo/index.html
#   CR_DATA_ROOT=/path bash viz/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

"$PY" "$HERE/build_site.py" "$@"
