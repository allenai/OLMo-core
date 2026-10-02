#!/usr/bin/env bash
# Copy finished cells from the node-local work dirs (every cell is written there FIRST) into the repo
# results dir when the repo copy is missing or unparsable (e.g. EDQUOT on /accounts, 2026-09-21).
# Idempotent; safe to run from the login node while jobs are live. Usage: sync_results.sh [node ...]
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core; RES=$REPO/debug/devloss_grid/results
for node in "${@:-cubbins lorax sneetches horton thidwick mcfuzz}"; do
  W=/net/$node/data/prasann/devloss_grid/work; [ -d "$W" ] || continue
  for f in "$W"/*.json; do
    [ -f "$f" ] || continue; b=$(basename "$f")
    python -c "import json,sys; json.load(open(sys.argv[1]))" "$f" 2>/dev/null || { echo "skip unparsable work copy $node:$b"; continue; }
    if ! python -c "import json,sys; json.load(open(sys.argv[1]))" "$RES/$b" 2>/dev/null; then
      cp "$f" "$RES/$b.sync" && mv "$RES/$b.sync" "$RES/$b" && echo "synced $b from $node" || { echo "!! could not write $b (quota?)"; rm -f "$RES/$b.sync"; }
    fi
  done
done
