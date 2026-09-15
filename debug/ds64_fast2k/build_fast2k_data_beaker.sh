#!/bin/bash
# Beaker-native data build for the FAST 2k screening loop (debug/ds64_fast2k/README.md).
#
# Unlike debug/ds64/build_ds64_data_beaker.sh this builds NO pools: it slices nested prefixes out of
# the ds64 campaign's EXISTING 2k rung pool (same task generator, same seed pool, the same rows the
# 2k slice of the short-heavy mix is drawn from) and tokenizes them at a SHORT seq-len. So the job
# is tokenization only -- minutes, no GPU, no ctc-data build -- and the fast2k rows come from
# exactly the length bucket the 2k eval rung scores.
#
#   TASK=outlier bash debug/ds64_fast2k/build_fast2k_data_beaker.sh
#
# Outputs (weka): $WEKA/ds64/fast2k/{arms/<task>_f<B>.jsonl, shards/<task>_f<B>/}
# Every shard dir also gets fast2k_stats.json -- the CE chance floor, the n/k distribution, and the
# train-vs-eval overlap check (fast2k_stats.py). Read them off the job log.
set -uo pipefail
export TASK="${TASK:-outlier}"
export BUDGETS="${BUDGETS:-2M,4M,8M}"    # nominal tokens at 2048/example -> 1024 / 2048 / 4096 rows
# TOKENIZE-time cap only (a longer example is DROPPED, silently shrinking the budget), so keep it
# generous; the TRAIN-time --seq-len is picked from the shard's reported max_example_len.
export SEQ_LEN="${SEQ_LEN:-8192}"
export WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
export TOKENIZER=$WEKA/hf_tokenizers/Qwen3.5-0.8B-Base
export EVAL_RUNG="${EVAL_RUNG:-$WEKA/outlier_lengthmix/eval_rungs/$TASK/rung_2048.jsonl}"
export CTC_BRANCH="${CTC_BRANCH:-prasann/ctc_public}"
case "$TASK" in
  nq) export CONV_TASK=retrieval CHUNK_BY=document ;;
  oolong) export CONV_TASK=oolong CHUNK_BY=line ;;
  *) export CONV_TASK=$TASK CHUNK_BY=document ;;
esac

gantry run --name "fast2k-data-$TASK-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/neptune*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' \
  --gpus 0 --cpus 16 --memory 120GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env TASK="$TASK" --env BUDGETS="$BUDGETS" --env SEQ_LEN="$SEQ_LEN" --env WEKA="$WEKA" \
  --env TOKENIZER="$TOKENIZER" --env EVAL_RUNG="$EVAL_RUNG" --env CONV_TASK="$CONV_TASK" \
  --env CHUNK_BY="$CHUNK_BY" --env CTC_BRANCH="$CTC_BRANCH" \
  --allow-dirty --timeout 0 --yes -- bash debug/ds64_fast2k/_build_fast2k_inner.sh
