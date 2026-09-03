#!/bin/bash
# Relaunch the three summary-token SFT arms on the v3 ladder, serving the CAUSAL arm of the mask
# mixture. Supersedes the 2026-08-15 sweep, of which almost nothing was usable:
#
#   * every job served the FULLY RESTRICTED mask, because the mixture coin is drawn under
#     `self.training` and inference left `causal_example` None. causal trained at
#     standard_mix_prob=1.0 and decay ends at mix_end_p=1.0, so both were scored under a mask they
#     never saw. Fixed in 6e3a4e309 (`--summary-mask-mode`, default causal).
#   * 60 of 87 exit-0 jobs wrote EMPTY json ("ladder keys present: []"): the bundle has no 1M/2M
#     files at all (every yarn4/yarn8 job) and no xlong rungs for rerank/oolong. Those passes are
#     dropped here rather than relaunched.
#   * 15 jobs aborted on the [maxlen] guard -- the <|summ|> run lengthens the prompt past the
#     MAX_LENGTH table's dense margin. The runner now scales the cap for VARIANT=summary.
#   * 8 jobs died in the block-sparse mask at seq_len ~33-36k. Serving the causal arm skips that
#     path entirely.
#
# Rungs kept are the ones whose files actually exist: base 2k-32k for all five tasks, 64k/128k and
# 256k for contra/nq/outlier only. 512k is dropped too -- the 2026-08-15 yarn2 jobs asked for
# '256k,512k' and only 256k came back.
#
# Usage:
#   MODE=smoke bash .../relaunch_2026-08-16_summtoken_v3_causal.sh   # 3 cheap gate jobs, run FIRST
#   MODE=full  bash .../relaunch_2026-08-16_summtoken_v3_causal.sh   # the 33-job sweep
#   DRY=1 MODE=full bash ...                                         # print, don't submit
set -uo pipefail

MODE="${MODE:-smoke}"
DRY="${DRY:-0}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"          # CLAUDE.md: every beaker job is urgent
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"        # ai2/oe-training is deprecated; this is what the sweep used
BRANCH="${BRANCH:-amandab/hils-eval}"
REF="${REF:-6e3a4e309347430bc0b222ea256f5c2acbe90a7e}"   # the summary-mask-mode commit
STEP_DIR="${STEP_DIR:-step1772}"
# The locally-installed `gantry` (v3.2.0) predates --min-runtime; pin the newer CLI via uvx.
GANTRY_BIN="${GANTRY_BIN:-uvx --from beaker-gantry@3.7.0 gantry}"
# CLAUDE.md/standing preference: default to no minRuntime (unallocated, always-preemptible).
# Only set MIN_RUNTIME (e.g. `MIN_RUNTIME=10m`) when explicitly asked for a specific launch --
# do not change this default.
MIN_RUNTIME="${MIN_RUNTIME:-}"
CKPT_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/amandab
LEDGER_DIR="${LEDGER_DIR:-records/eval_launches}"

ARMS="${ARMS:-causal p50 decay}"
LADDER_VERSION=v3
SUMMARY_MASK_MODE="${SUMMARY_MASK_MODE:-causal}"

submit() {                              # submit <name> <env-prefix string>
  local name="$1" envs="$2"
  local cmd="$envs bash src/scripts/train/memexpress/singletask_ladder/run_beaker_multirung_eval.sh"
  if [ "$DRY" = "1" ]; then
    echo "DRY  $name"
    echo "     $cmd"
    return 0
  fi
  local min_runtime_args=()
  [ -n "$MIN_RUNTIME" ] && min_runtime_args=(--min-runtime "$MIN_RUNTIME")
  $GANTRY_BIN run \
    --name "$name" --task-name eval \
    --workspace "$WORKSPACE" --cluster "$CLUSTER" --priority "$PRIORITY" \
    "${min_runtime_args[@]}" \
    --beaker-image "$IMAGE" --gpus "$NGPU" --shared-memory 10GiB \
    --weka oe-training-default:/weka/oe-training-default \
    --budget "$BUDGET" \
    --branch "$BRANCH" --ref "$REF" \
    --env OMP_NUM_THREADS=8 --env NCCL_DEBUG=WARN \
    --install "pip install -e '.[all]'" \
    --yes --allow-dirty --no-logs \
    -- bash -lc "$cmd"
  local rc=$?
  # Count only what Beaker actually accepted. A submission can fail (deprecated budget, bad image)
  # while the loop keeps going, and reporting those as launched is how a sweep gets believed to be
  # running when it is not.
  if [ $rc -ne 0 ]; then
    echo "!! SUBMIT FAILED (rc=$rc): $name"
    failed=$((failed+1))
  else
    ok=$((ok+1))
  fi
  return $rc
}

echo "=== summtoken v3 relaunch: MODE=$MODE mask=$SUMMARY_MASK_MODE ladder=$LADDER_VERSION ref=${REF:0:9} ==="
n=0; ok=0; failed=0

for arm in $ARMS; do
  RUN="q35-4b-summ-${arm}-5task-packed"
  CKPT="$CKPT_ROOT/$RUN/$STEP_DIR"

  if [ "$MODE" = "smoke" ]; then
    # Gate: 8 examples at 2k and 16k. 16k is the point -- it clears the flex threshold (8192), so it
    # proves the causal short-circuit keeps a long prefill off the block-sparse path.
    NGPU=1
    n=$((n+1))
    submit "ev-smoke-${arm}-v3c-$(date +%H%M%S)" \
      "RUN=$RUN TASK=contra VARIANT=summary CKPT='$CKPT' PROMPT_FORMAT='chat' QUERY_POSITION='both' \
MAX_TEST=8 BATCH_SIZE=1 NGPU=1 LADDER_XLONG=0 XLONG_ONLY=0 COT_MODE='none' EVAL_TAG='v3c-smoke' \
NUM_SUMMARY_TOKENS=5 RUNGS_OVERRIDE='2k,16k' LADDER_VERSION=$LADDER_VERSION \
SUMMARY_MASK_MODE=$SUMMARY_MASK_MODE TOKENIZER=Qwen/Qwen3.5-0.8B \
WEKA_LLM=/weka/oe-training-default/ai2-llm"
    continue
  fi

  NGPU=2
  # ---- base ladder, 2k-32k, all five v3 tasks ----
  for task in contra nq outlier rerank oolong; do
    n=$((n+1))
    submit "ev-${task}-${RUN}-v3c-base" \
      "RUN=$RUN TASK=$task VARIANT=summary CKPT='$CKPT' PROMPT_FORMAT='chat' QUERY_POSITION='both' \
MAX_TEST=600 BATCH_SIZE=1 NGPU=2 LADDER_XLONG=0 XLONG_ONLY=0 COT_MODE='none' EVAL_TAG='v3c-base' \
NUM_SUMMARY_TOKENS=5 RUNGS_OVERRIDE='' LADDER_VERSION=$LADDER_VERSION \
SUMMARY_MASK_MODE=$SUMMARY_MASK_MODE TOKENIZER=Qwen/Qwen3.5-0.8B \
WEKA_LLM=/weka/oe-training-default/ai2-llm"
  done

  # ---- xlong native, 64k/128k, only the tasks with xlong files ----
  for task in contra nq outlier; do
    n=$((n+1))
    submit "ev-${task}-${RUN}-v3c-xlong-native" \
      "RUN=$RUN TASK=$task VARIANT=summary CKPT='$CKPT' PROMPT_FORMAT='chat' QUERY_POSITION='both' \
MAX_TEST=600 BATCH_SIZE=1 NGPU=2 LADDER_XLONG=1 XLONG_ONLY=1 XLONG_RUNGS='64k,128k' COT_MODE='none' \
EVAL_TAG='v3c-xlong-native' NUM_SUMMARY_TOKENS=5 RUNGS_OVERRIDE='' LADDER_VERSION=$LADDER_VERSION \
SUMMARY_MASK_MODE=$SUMMARY_MASK_MODE TOKENIZER=Qwen/Qwen3.5-0.8B \
WEKA_LLM=/weka/oe-training-default/ai2-llm"
  done

  # ---- xlong yarn2, 256k only (512k has no files) ----
  for task in contra nq outlier; do
    n=$((n+1))
    submit "ev-${task}-${RUN}-v3c-xlong-yarn2" \
      "RUN=$RUN TASK=$task VARIANT=summary CKPT='${CKPT}_yarn2' PROMPT_FORMAT='chat' QUERY_POSITION='both' \
MAX_TEST=600 BATCH_SIZE=1 NGPU=2 LADDER_XLONG=1 XLONG_ONLY=1 XLONG_RUNGS='256k' COT_MODE='none' \
EVAL_TAG='v3c-xlong-yarn2' NUM_SUMMARY_TOKENS=5 RUNGS_OVERRIDE='' LADDER_VERSION=$LADDER_VERSION \
SUMMARY_MASK_MODE=$SUMMARY_MASK_MODE TOKENIZER=Qwen/Qwen3.5-0.8B \
WEKA_LLM=/weka/oe-training-default/ai2-llm"
  done
done

echo "=== attempted $n, ACCEPTED $ok, failed $failed (MODE=$MODE dry=$DRY) ==="
[ "$failed" -gt 0 ] && echo "!! $failed submission(s) did NOT reach Beaker -- do not treat this sweep as launched."
echo "Ledger: $LEDGER_DIR/2026-08-16_q35-4b-summ-*-v3causal.yaml"
