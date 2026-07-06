#!/bin/bash
#
# Launch corpus-reasoning SUITE evals (cr_* tasks, native olmo_core generation) for an olmo-core
# checkpoint on weka. Uses the oe-eval branch prasann/longctx-eval (corpus_reasoning_suite.py),
# which reads eval JSONL from weka cr_suite_data by default (the cr_* task default;
# override with CR_SUITE_DATA_DIR).
#
# Task names are cr_<eval-file-stem> (see the bundled suite_eval_manifest.tsv); modes are the
# response-style prefills direct / cot / cr.
#
# Usage:
#   ./launch_cr_suite_evals.sh /weka/oe-training-default/ai2-llm/checkpoints/prasanns/<run>/stepXXXX
#
# Overrides (env):
#   TASKS="cr_oolong_test_synth_ctx8192_spliteval cr_contradiction_eval_pubmed_both_n100_k3 ..."
#   MODES="direct"            # direct | cot | cr (space-separated)
#   PRIORITY CLUSTER NUM_GPUS BATCH_SIZE OLMO_CORE_TOKENIZER WORKSPACE DASHBOARD
#   OE_EVAL_BRANCH (default prasann/longctx-eval)  OLMO_CORE_COMMIT (default HEAD)
#   CR_SUITE_DATA_DIR (override the weka eval-data dir)  NAME_SUFFIX
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <weka_checkpoint_path>}"

PRIORITY="${PRIORITY:-urgent}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
BUDGET="ai2/oe-other"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"                       # landmark attention -> bs=1
OLMO_CORE_TOKENIZER="${OLMO_CORE_TOKENIZER:-Qwen/Qwen3-4B}"
DASHBOARD="${DASHBOARD:-memory-cr-suite}"
OE_EVAL_BRANCH="${OE_EVAL_BRANCH:-prasann/longctx-eval}"
MODES=( ${MODES:-direct} )
# Default smoke set: one representative task per a few families (small, fast). Override via TASKS.
TASKS=( ${TASKS:-cr_oolong_test_synth_ctx8192_spliteval cr_contradiction_eval_pubmed_both_n100_k3} )
NAME_SUFFIX="${NAME_SUFFIX:-}"

OLMO_CORE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT:-$(git -C "${OLMO_CORE_REPO}" rev-parse HEAD)}"

FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"

# olmo-cookbook-eval parses -l/--gantry-args as comma-split key=value pairs, NOT JSON.
INSTALL_CMD="uv sync --python 3.11 && uv pip install --no-deps ${FLASH_ATTN_WHEEL} && uv pip install --no-deps git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_COMMIT}"
GANTRY_ARGS="install=${INSTALL_CMD}"
# Optional: point the cr_* tasks at a non-default eval-data dir on weka.
if [ -n "${CR_SUITE_DATA_DIR:-}" ]; then
  GANTRY_ARGS="${GANTRY_ARGS} -l env=CR_SUITE_DATA_DIR=${CR_SUITE_DATA_DIR}"
fi

for mode in "${MODES[@]}"; do
  TASK_FLAGS=()
  for t in "${TASKS[@]}"; do
    TASK_FLAGS+=( -t "${t}::${mode}" )
  done
  suffix="cr-${mode}${NAME_SUFFIX:+-${NAME_SUFFIX}}"
  echo "==> Launching ${#TASKS[@]} cr_* task(s), mode=${mode} for ${MODEL_PATH}"
  olmo-cookbook-eval evaluate \
    "${MODEL_PATH}" \
    --model-backend olmo_core \
    -y "${PRIORITY}" \
    -c "${CLUSTER}" \
    -b "${BUDGET}" \
    -d "${DASHBOARD}" \
    -w "${WORKSPACE}" \
    "${TASK_FLAGS[@]}" \
    -n "${NUM_GPUS}" \
    -z "${BATCH_SIZE}" \
    -g \
    --oe-eval-branch "${OE_EVAL_BRANCH}" \
    --name-suffix "${suffix}" \
    -l "${GANTRY_ARGS}" \
    --model-args "trust_remote_code=true,max_length=65536,tokenizer=${OLMO_CORE_TOKENIZER}"
done

echo "All cr_* suite eval jobs submitted."
