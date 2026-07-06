#!/bin/bash
#
# Launch landmark top-k retrieval evals for an olmo-core landmark checkpoint.
#
# For each requested context length, runs HELMET and/or RULER with hard top-k
# landmark block retrieval at decode time (the landmark paper's inference
# procedure, https://arxiv.org/abs/2305.16300 section 3.2). One job per
# (suite, length, top_k) because the top_k must match the length.
#
# top_k is set as a fraction of the context window, measured in landmark blocks
# (block_size = mem_freq + 1 = 64 for this model, so a length of L tokens has
# L/64 blocks):
#     top_k = round(fraction * L / 64) = len_k * {4, 8, 12}  for {1/4, 1/2, 3/4}
#
# Requires the patched harness branches:
#   * ai2-helmet  @ amandab/qwen-landmark-eval  (--olmo_core_landmark_top_k, MIN_LENGTH)
#   * oe-eval     @ amandab/ruler-memproj       (landmark_top_k_blocks model-arg)
# and an olmo-core commit with GenerationConfig.landmark_top_k_blocks (pushed).
#
# Usage:
#   ./launch_topk_landmark_evals.sh <weka_checkpoint_path>
#
# Env overrides:
#   LENGTHS_K     space-separated lengths in K (default "8 16 32 64 128"; RULER also 4)
#   FRACTIONS     space-separated as num/den (default "1/4 1/2 3/4")
#   SUFFIX_TAG    extra tag appended to every run suffix (e.g. "fix"); gives HELMET a fresh
#                 OUTPUT_DIR and RULER a fresh run name so a rerun never reuses cached/stale
#                 outputs from a prior buggy run (the harness skips already-completed outputs)
#   SKIP_HELMET, SKIP_RULER
#   RULER_LENGTHS_K  override RULER lengths (default: LENGTHS_K plus 4)
#   PRIORITY, CLUSTER, WORKSPACE, RULER_DASHBOARD, COOKBOOK_BIN
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <weka_checkpoint_path>}"

# ---- shared knobs (mirror launch_long_context_evals.sh) ----
PRIORITY="${PRIORITY:-urgent}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="ai2/oe-other"
NUM_GPUS_RULER="${NUM_GPUS_RULER:-2}"
NUM_GPUS_HELMET="${NUM_GPUS_HELMET:-8}"
OLMO_CORE_TOKENIZER="${OLMO_CORE_TOKENIZER:-Qwen/Qwen3-4B}"
OLMO_CORE_BATCH_SIZE="${OLMO_CORE_BATCH_SIZE:-1}"   # landmark attention -> bs=1
BLOCK_SIZE="${BLOCK_SIZE:-64}"                       # mem_freq(63) + 1

OLMO_CORE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT:-$(git -C "${OLMO_CORE_REPO}" rev-parse HEAD)}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"
HELMET_DIR="${HELMET_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../ai2-helmet" && pwd)}"
COOKBOOK_BIN="${COOKBOOK_BIN:-olmo-cookbook-eval}"

# oe-eval branch carrying the landmark_top_k_blocks model-arg (REQUIRED for RULER top-k).
OE_EVAL_BRANCH="${OE_EVAL_BRANCH:-amandab/ruler-memproj}"
# Distinct dashboard so top-k runs never collide with prior dense memory-LC runs.
RULER_DASHBOARD="${RULER_DASHBOARD:-memory-LC-topk}"

LENGTHS_K=( ${LENGTHS_K:-8 16 32 64 128} )
RULER_LENGTHS_K=( ${RULER_LENGTHS_K:-4 ${LENGTHS_K[@]}} )
FRACTIONS=( ${FRACTIONS:-1/4 1/2 3/4} )

# top_k (in blocks) for fraction num/den of a len_k context.
topk_for() {
  local len_k="$1" num="$2" den="$3"
  echo $(( len_k * 1024 / BLOCK_SIZE * num / den ))
}

frac_tag() { echo "$1" | tr '/' '_'; }   # 1/4 -> 1_4, for run-name suffixes

# DRY_RUN=1 prints the launch commands instead of executing them.
run() { if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY: $*"; else "$@"; fi; }

# olmo-cookbook-eval transiently fails when its per-call temp clone of the oe-eval branch hiccups
# (it then runs oe_eval/launch.py from CWD and can't find it). Retry a few times, and never let one
# failed submission abort the whole sweep -- record it and move on.
FAILED_SUBMITS=()
run_retry() {
  local what="$1"; shift
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY: $*"; return 0; fi
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "  !! ${what}: submit attempt ${attempt} failed; retrying..." >&2
    sleep 5
  done
  echo "  XX ${what}: giving up after 3 attempts" >&2
  FAILED_SUBMITS+=("${what}")
  return 0
}

echo "Model:        ${MODEL_PATH}"
echo "olmo_core:    ${OLMO_CORE_COMMIT}"
echo "HELMET:       ${SKIP_HELMET:+SKIP}${SKIP_HELMET:-lengths ${LENGTHS_K[*]}k}"
echo "RULER:        ${SKIP_RULER:+SKIP}${SKIP_RULER:-lengths ${RULER_LENGTHS_K[*]}k}"
echo "fractions:    ${FRACTIONS[*]}"
echo

# ============================================================================
# HELMET: one job per (length, fraction); MIN_LENGTH=MAX_LENGTH isolates the length.
# ============================================================================
if [ "${SKIP_HELMET:-0}" != "1" ]; then
  for len_k in "${LENGTHS_K[@]}"; do
    for frac in "${FRACTIONS[@]}"; do
      num="${frac%/*}"; den="${frac#*/}"
      topk="$(topk_for "$len_k" "$num" "$den")"
      max_len=$(( len_k * 1024 ))
      suffix="${len_k}k_tk${topk}${SUFFIX_TAG:+_${SUFFIX_TAG}}"
      echo "==> HELMET ${len_k}k  top_k=${topk} (${frac})  suffix=${suffix}"
      if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY: HELMET MAX=MIN=${max_len} TOP_K=${topk} SUFFIX=${suffix} -> gantry_eval.sh"
        continue
      fi
      ( cd "${HELMET_DIR}" && \
        MODEL_NAME_OR_PATH="${MODEL_PATH}" \
        MAX_LENGTH="${max_len}" \
        MIN_LENGTH="${max_len}" \
        OLMO_CORE_LANDMARK_TOP_K="${topk}" \
        EVAL_NAME_SUFFIX="${suffix}" \
        CLUSTER="${CLUSTER}" \
        WORKSPACE="${WORKSPACE}" \
        BUDGET="${BUDGET}" \
        NUM_GPUS="${NUM_GPUS_HELMET}" \
        BACKEND=olmo_core \
        PRIORITY="${PRIORITY}" \
        OLMO_CORE_TOKENIZER="${OLMO_CORE_TOKENIZER}" \
        OLMO_CORE_BATCH_SIZE="${OLMO_CORE_BATCH_SIZE}" \
        OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT}" \
        TIMEOUT=0 \
        bash ./gantry_eval.sh )
    done
  done
fi

# ============================================================================
# RULER: one job per (length, fraction) via olmo-cookbook-eval -> oe-eval.
# ============================================================================
if [ "${SKIP_RULER:-0}" != "1" ]; then
  for len_k in "${RULER_LENGTHS_K[@]}"; do
    for frac in "${FRACTIONS[@]}"; do
      num="${frac%/*}"; den="${frac#*/}"
      topk="$(topk_for "$len_k" "$num" "$den")"
      max_len=$(( len_k * 1024 ))
      suffix="${len_k}k_tk${topk}${SUFFIX_TAG:+_${SUFFIX_TAG}}"
      model_args="trust_remote_code=true,max_length=${max_len},tokenizer=${OLMO_CORE_TOKENIZER},landmark_top_k_blocks=${topk}"
      echo "==> RULER ruler:${len_k}k  top_k=${topk} (${frac})  suffix=${suffix}"
      run_retry "RULER ${len_k}k tk${topk}" "${COOKBOOK_BIN}" evaluate \
        "${MODEL_PATH}" \
        --model-backend olmo_core \
        -y "${PRIORITY}" \
        -c "${CLUSTER}" \
        -b "${BUDGET}" \
        -d "${RULER_DASHBOARD}" \
        -w "${WORKSPACE}" \
        -t "ruler:${len_k}k" \
        -n "${NUM_GPUS_RULER}" \
        -z "${OLMO_CORE_BATCH_SIZE}" \
        -g \
        --oe-eval-branch "${OE_EVAL_BRANCH}" \
        --name-suffix "${suffix}" \
        -l "install=uv sync --python 3.11 && uv pip install --no-deps ${FLASH_ATTN_WHEEL} && uv pip install --no-deps git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_COMMIT}" \
        --model-args "${model_args}"
    done
  done
fi

echo
if [ "${#FAILED_SUBMITS[@]}" -gt 0 ]; then
  echo "FAILED submissions (${#FAILED_SUBMITS[@]}): ${FAILED_SUBMITS[*]}"
  echo "Re-run with the appropriate LENGTHS_K / SKIP_* to retry just those."
else
  echo "All submissions succeeded."
fi
echo "Done submitting."
