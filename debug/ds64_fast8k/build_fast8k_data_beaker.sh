#!/bin/bash
# Beaker-native data build for the FAST 8k screening loop (debug/ds64_fast8k/README.md).
#
# Same construction as debug/ds64_fast2k/build_fast2k_data_beaker.sh, one rung up: it builds NO
# pools, it slices windows out of the ds64 campaign's EXISTING 8k rung pool and tokenizes them.
# WHY 8k and not 2k: outlier's soft-token arms die at 8k+, not at 2k (cc00 scores 0.80 at 2k and
# 0.17 at 8k), so the 2k screen is blind to the thing being screened. Everything else -- unpacked
# rows, matched steps, the CE floor, the FLOP meter -- is the fast2k loop unchanged.
#
#   TASK=outlier bash debug/ds64_fast8k/build_fast8k_data_beaker.sh
#
# Outputs (weka): $WEKA/ds64/fast8k/{arms/<task>_g<slice>.jsonl, shards/<task>_g<slice>/}
# Every shard dir gets fast8k_stats.json (vs the 8k rung) and fast8k_stats_2k.json (vs the 2k rung):
# the CE chance floor, the n/k distribution, and the train-vs-eval overlap check.
set -uo pipefail
export TASK="${TASK:-outlier}"
# Nominal tokens at 8192/example -> 977 / 1953 / 2441 rows. The ds64 8k pool is ~2700 rows, so 20M
# is the largest budget that fits; compose_fast8k.py SKIPS (never truncates) a budget that does not.
export BUDGETS="${BUDGETS:-8M,16M,20M}"
# The two-phase arm's disjoint windows: 85% of the 16M budget, then the remaining 15%.
export SLICES="${SLICES:-P1:0:1660,P2:1660:293}"
# TOKENIZE-time cap only (a longer example is DROPPED, silently shrinking the budget), so keep it
# generous; the TRAIN-time --seq-len comes from the shard's reported max_example_len.
export SEQ_LEN="${SEQ_LEN:-24576}"
export POOL_ROWS="${POOL_ROWS:-2700}"    # only used if the ds64 8k pool is missing
export WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
export TOKENIZER=$WEKA/hf_tokenizers/Qwen3.5-0.8B-Base
export EVAL_RUNG="${EVAL_RUNG:-$WEKA/outlier_lengthmix/eval_rungs/$TASK/rung_8192.jsonl}"
export EVAL_RUNG_2K="${EVAL_RUNG_2K:-$WEKA/outlier_lengthmix/eval_rungs/$TASK/rung_2048.jsonl}"
export CTC_BRANCH="${CTC_BRANCH:-prasann/ctc_public}"
case "$TASK" in
  nq) export CONV_TASK=retrieval CHUNK_BY=document ;;
  oolong) export CONV_TASK=oolong CHUNK_BY=line ;;
  *) export CONV_TASK=$TASK CHUNK_BY=document ;;
esac

# This is a 0-GPU tokenization job, so it belongs on the CPU-only DEV clusters FIRST: phobos and
# hammond have `storage:weka` and no GPUs at all, while saturn/neptune/ceres schedule `eager` and
# were measured at 0/216, 0/96 and 0/88 free slots -- an urgent CPU job queued behind them for
# ~50 min with no placement (2026-09-15). jupiter is kept as the strict-priority backfill.
CLUSTERS="${CLUSTERS:-ai2/phobos ai2/hammond ai2/jupiter* ai2/holmes* ai2/titan* ai2/prometheus* ai2/ceres* ai2/saturn* ai2/neptune*}"
CLUSTER_ARGS=""; for c in $CLUSTERS; do CLUSTER_ARGS="$CLUSTER_ARGS --cluster $c"; done

gantry run --name "fast8k-data-$TASK-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  $CLUSTER_ARGS \
  --gpus 0 --cpus ${BUILD_CPUS:-4} --memory ${BUILD_MEM:-32GiB} --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env TASK="$TASK" --env BUDGETS="$BUDGETS" --env SLICES="$SLICES" --env SEQ_LEN="$SEQ_LEN" \
  --env WEKA="$WEKA" --env POOL_ROWS="$POOL_ROWS" \
  --env TOKENIZER="$TOKENIZER" --env EVAL_RUNG="$EVAL_RUNG" --env EVAL_RUNG_2K="$EVAL_RUNG_2K" \
  --env CONV_TASK="$CONV_TASK" --env CHUNK_BY="$CHUNK_BY" --env CTC_BRANCH="$CTC_BRANCH" \
  --env BUILD_CPUS="${BUILD_CPUS:-4}" \
  --allow-dirty --timeout 0 --yes -- bash debug/ds64_fast8k/_build_fast8k_inner.sh
