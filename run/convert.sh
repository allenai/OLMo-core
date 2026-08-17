#!/usr/bin/env bash
# Tokenize task JSONL into olmo-core SFT shards.  All arguments pass through to convert_to_shards.
#
#   run/convert.sh --input-jsonl DIR/contradiction/train.jsonl --task contradiction \
#       --out-dir SHARDS/contradiction --tokenizer Qwen/Qwen3.5-0.8B-Base \
#       --marker-set qwen3_5 --emit landmark --query-position after
#
# This is the step that writes the shard's format fingerprint -- the record ctc-eval later checks
# a checkpoint against. The flags that must match at eval time are listed in
# src/scripts/ctc/README.md ("Things that must match between convert and eval").
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
exec "$CTC_PYTHON" "$CTC_REPO/src/scripts/ctc/convert_to_shards.py" "$@"
