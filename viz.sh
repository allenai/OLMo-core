#!/bin/bash
# Build / publish the corpus-reasoning visualization websites — fully in-tree.
#
# TWO sites share this codebase (see src/corpus_reasoning/viz_hub/README.md):
#   * HUB site  (Cloudflare Pages project 'corpus-reasoning-hub'):  src/corpus_reasoning/viz_hub/
#     — the Results tab (central results-hub table) + experiments. Built by THIS wrapper.
#   * MIXING site (Pages project 'corpus-reasoning-viz'):           src/corpus_reasoning/viz/
#     — the mixing-tables build. Build directly: bash src/corpus_reasoning/viz/run.sh
#
# (History: this used to delegate to a standalone ./corpus-reasoning clone; that clone was
# deleted 2026-07-13 after its unique viz work was pushed to origin and vendored into viz_hub/.)
#
#   bash viz.sh                 # build src/corpus_reasoning/viz_hub/outputs/index.html
#   bash viz.sh --update-demo   # also refresh the committed demo snapshot
#   bash viz.sh --publish       # build + publish (see deploy_cloudflare.sh)
#
# Experiment configs are read from THIS OLMo-core checkout (OLMO_CORE_ROOT); task data from
# CR_DATA_ROOT (default /scratch/users/prasann/corpus-reasoning/data); the results table from
# RESULTS_HUB_DIR (default: sibling results-hub checkout).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OLMO_CORE_ROOT="${OLMO_CORE_ROOT:-$ROOT}"

exec bash "$ROOT/src/corpus_reasoning/viz_hub/run.sh" "$@"
