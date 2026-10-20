#!/bin/bash
# setA Pareto data jobs (CPU-only gantry, unallocated + urgent, minRuntime 1h). Runs the PUSHED commit.
#
#   bash research/token_routing/setA_pareto/data_beaker.sh gold          # setA (seed 42): gold sidecars + 32k merge
#   bash research/token_routing/setA_pareto/data_beaker.sh expand 43     # fresh setA rows at seed 43 (2k-32k) + merge
#
# Outputs under $WEKA/ctc_sft_sets/setA_pareto/:
#   setA32k_s<seed>/   one merged shard (11 tasks, rows <= 32,768 tokens) + gold_fingerprints.json
#   reports/           sidecar / build / overlap reports
# The work itself is _data_inner.sh. ACCEPT_DIFFER=rerank lets named tasks differ from the shipped
# shards (seed 42: the shipped rerank shards were not tokenized from the current train.jsonl; same rows).
set -euo pipefail
MODE="${1:?gold|expand}"; SEED="${2:-42}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
CPUS="${CPUS:-32}"
gantry run --name "setA-pareto-data-$MODE-s$SEED-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' --cluster ai2/phobos --cluster ai2/hammond \
  --gpus 0 --cpus "$CPUS" --memory 192GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env WEKA="$WEKA" --env MODE="$MODE" --env SEED="$SEED" --env JOBS="$CPUS" --env ACCEPT_DIFFER="${ACCEPT_DIFFER:-}" \
  --allow-dirty --timeout 0 --yes -- bash research/token_routing/setA_pareto/_data_inner.sh
