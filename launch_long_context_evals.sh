#!/bin/bash
#
# Launch long-context evals for an olmo-core checkpoint on weka:
#   * HELMET  @ 8k-64k   -> via ../ai2-helmet/gantry_eval.sh
#   * RULER   @ 4k-64k   -> via olmo-cookbook-eval (oe-eval), like ../olmo-cookbook/run_ruler.sh
#
# HELMET's configs/single_task/* only exist at 8k/16k/32k/64k, so a single
# gantry_eval.sh run with MAX_LENGTH=65536 IS exactly the 8k-64k suite.
#
# RULER 4k is NOT available in ai2-helmet (configs-dist only has 8k-64k), which
# is why RULER goes through oe-eval instead: ruler:4k..ruler:64k are registered
# task groups there (each = the full 13-subtask suite at that length).
#
# Defaults assume a landmark-attention Qwen3-4B olmo-core checkpoint:
#   olmo_core backend, Qwen/Qwen3-4B tokenizer, generation batch size 1
#   (landmark attention requires bs=1), urgent priority.
#
# Usage:
#   ./launch_long_context_evals.sh /weka/oe-training-default/ai2-llm/checkpoints/<user>/<run>/stepXXXX
#
# All jobs run in workspace ai2/flex2; RULER results report to the memory-LC
# dashboard (HELMET writes outputs to weka, it has no dashboard).
#
# Common overrides (env vars):
#   PRIORITY, NUM_GPUS, OLMO_CORE_TOKENIZER, OLMO_CORE_BATCH_SIZE,
#   CLUSTER, WORKSPACE, RULER_DASHBOARD, SKIP_HELMET=1, SKIP_RULER=1,
#   HELMET_MAX_LENGTH (cap HELMET suite, e.g. 32768 for an un-extended model)

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <weka_checkpoint_path>}"

# ---- shared knobs ----
PRIORITY="${PRIORITY:-urgent}"                       # all jobs run at urgent priority
CLUSTER="${CLUSTER:-ai2/jupiter}"                    # all jobs run on jupiter
BUDGET="ai2/oe-other"                                # ai2/oe-base is deprecated
NUM_GPUS_RULER="${NUM_GPUS_RULER:-2}"
NUM_GPUS_HELMET="${NUM_GPUS_HELMET:-8}"
OLMO_CORE_TOKENIZER="${OLMO_CORE_TOKENIZER:-Qwen/Qwen3-4B}"
OLMO_CORE_BATCH_SIZE="${OLMO_CORE_BATCH_SIZE:-1}"    # landmark attention -> bs=1

# These checkpoints were written by a branch of OLMo-core whose configs use
# fields not present in the released ai2-olmo-core on PyPI (e.g. RoPEConfig's
# no_global_rope). Both eval images install olmo_core from PyPI, so we reinstall
# it from the checkpoint's training commit or the model config won't deserialize.
# Defaults to the current HEAD of this OLMo-core checkout (override if needed).
OLMO_CORE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT:-$(git -C "${OLMO_CORE_REPO}" rev-parse HEAD)}"

# The RULER (oe-eval, gantry) path builds a fresh uv venv that lacks flash-attn,
# which the olmo_core model requires. Install the prebuilt wheel matching the
# gantry env (torch 2.8 + cu12 + cp311, cxx11abiTRUE — the only variant shipped).
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"

# Resolve ../ai2-helmet relative to this script's location.
HELMET_DIR="${HELMET_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../ai2-helmet" && pwd)}"

# Optional chat-template HELMET run. Set HELMET_USE_CHAT_TEMPLATE=True to evaluate with the chat
# template; HELMET's --thinking flag is left off, so hybrid models (e.g. Qwen3) run in non-thinking
# mode (enable_thinking=False). A marker (default "nothink") is appended to the output dir and job
# description so it never collides with the default completion-mode run. Override with
# HELMET_NAME_SUFFIX.
HELMET_USE_CHAT_TEMPLATE="${HELMET_USE_CHAT_TEMPLATE:-False}"
HELMET_NAME_SUFFIX="${HELMET_NAME_SUFFIX:-}"
if [ -z "${HELMET_NAME_SUFFIX}" ] && [ "${HELMET_USE_CHAT_TEMPLATE}" = "True" ]; then
  HELMET_NAME_SUFFIX="nothink"
fi

# ============================================================================
# 1) HELMET @ 8k-64k  (ai2-helmet gantry_eval.sh, olmo_core backend)
# ============================================================================
if [ "${SKIP_HELMET:-0}" != "1" ]; then
  echo "==> Launching HELMET 8k-64k for ${MODEL_PATH}"
  ( cd "${HELMET_DIR}" && \
    MODEL_NAME_OR_PATH="${MODEL_PATH}" \
    MAX_LENGTH="${HELMET_MAX_LENGTH:-65536}" \
    CLUSTER="${CLUSTER}" \
    WORKSPACE="${WORKSPACE:-ai2/flex2}" \
    BUDGET="${BUDGET}" \
    NUM_GPUS="${NUM_GPUS_HELMET}" \
    BACKEND=olmo_core \
    PRIORITY="${PRIORITY}" \
    OLMO_CORE_TOKENIZER="${OLMO_CORE_TOKENIZER}" \
    OLMO_CORE_BATCH_SIZE="${OLMO_CORE_BATCH_SIZE}" \
    OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT}" \
    USE_CHAT_TEMPLATE="${HELMET_USE_CHAT_TEMPLATE}" \
    EVAL_NAME_SUFFIX="${HELMET_NAME_SUFFIX}" \
    bash ./gantry_eval.sh )
fi

# ============================================================================
# 2) RULER @ 4k-64k  (olmo-cookbook-eval -> oe-eval, olmo_core backend)
#    One job per length; each ruler:Nk group runs the full 13-subtask suite.
#    olmo_core backend reads the tokenizer from the checkpoint config (matches
#    ../olmo-cookbook/run_ruler.sh, which passes no tokenizer for olmo_core).
# ============================================================================
# Lengths in K; max_length = K * 1024. Kept as a plain list (no associative
# arrays) so this runs under macOS's stock bash 3.2.
if [ "${SKIP_RULER:-0}" != "1" ]; then
  # Override with e.g. RULER_LENGTHS_K="4" to smoke-test a single length.
  RULER_LENGTHS_K=( ${RULER_LENGTHS_K:-4 8 16 32 64} )

  # Optional non-thinking (ChatML) RULER variant. Set RULER_CHAT_TEMPLATE (e.g. qwen3_nothink) to
  # run RULER through a chat template, and OE_EVAL_BRANCH to the oe-eval branch that carries that
  # template (e.g. amandab/ruler-memproj). ALWAYS pair the non-thinking run with a distinct
  # RULER_DASHBOARD: the cookbook keys the remote output path on the dashboard
  # (s3://.../evaluation/<dashboard>/<model>/<task-hash>), so a separate dashboard is what keeps the
  # thinking and non-thinking partial-results files from ever colliding.
  OE_EVAL_BRANCH_FLAG=""
  if [ -n "${OE_EVAL_BRANCH:-}" ]; then
    OE_EVAL_BRANCH_FLAG="--oe-eval-branch ${OE_EVAL_BRANCH}"
  fi

  # Put a marker (default "nothink") in the run name for chat-template variants. --name-suffix
  # appends it to the model name, so it shows up in the beaker display name, the dashboard row, and
  # the S3 output dir -- keeping the variant clearly distinct from the default RULER run. Override
  # the marker with RULER_NAME_SUFFIX.
  NAME_SUFFIX_FLAG=""
  ruler_name_suffix="${RULER_NAME_SUFFIX:-}"
  if [ -z "${ruler_name_suffix}" ] && [ -n "${RULER_CHAT_TEMPLATE:-}" ]; then
    ruler_name_suffix="nothink"
  fi
  if [ -n "${ruler_name_suffix}" ]; then
    NAME_SUFFIX_FLAG="--name-suffix ${ruler_name_suffix}"
  fi

  for k in "${RULER_LENGTHS_K[@]}"; do
    task="ruler:${k}k"
    max_length=$(( k * 1024 ))
    model_args="trust_remote_code=true,max_length=${max_length},tokenizer=${OLMO_CORE_TOKENIZER}"
    if [ -n "${RULER_CHAT_TEMPLATE:-}" ]; then
      model_args="${model_args},chat_model=true,chat_template=${RULER_CHAT_TEMPLATE}"
    fi
    echo "==> Launching RULER ${task} (max_length=${max_length}) for ${MODEL_PATH}"
    olmo-cookbook-eval evaluate \
      "${MODEL_PATH}" \
      --model-backend olmo_core \
      -y "${PRIORITY}" \
      -c "${CLUSTER}" \
      -b "${BUDGET}" \
      -d "${RULER_DASHBOARD:-memory-LC}" \
      -w "${WORKSPACE:-ai2/flex2}" \
      -t "${task}" \
      -n "${NUM_GPUS_RULER}" \
      -z "${OLMO_CORE_BATCH_SIZE}" \
      -g \
      ${OE_EVAL_BRANCH_FLAG} \
      ${NAME_SUFFIX_FLAG} \
      -l "install=uv sync --python 3.11 && uv pip install --no-deps ${FLASH_ATTN_WHEEL} && uv pip install --no-deps git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_COMMIT}" \
      --model-args "${model_args}"
  done
fi

echo "All long-context eval jobs submitted."
