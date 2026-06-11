#!/bin/bash
#
# Launch oolong + contradiction evals for an olmo-core checkpoint on weka, in the same
# format as launch_long_context_evals.sh (RULER section): olmo-cookbook-eval -> oe-eval,
# native olmo_core generation backend, one gantry job per (task-set x mode).
#
# Tasks come from the oe-eval branch prasann/longctx-eval (oe_eval_tasks/longctx_corpus_reasoning.py):
#   longctx_oolong_synth_ctx{1024..65536}           7 tasks, oolong spliteval ladder
#   longctx_contradiction_pubmed_n{100,250,500,1000} 4 tasks, contradiction eval ladder
#
# Modes (config variants) select the response style of cotmix-SFT'd models by prefilling
# the assistant turn:
#   direct -> prefill "Answer:" / "Contradicting pairs:"  (plain answers)
#   cot    -> prefill "Reasoning:" (larger generation budget)
#   cr     -> no prefill (model's choice; use for plain-SFT or base checkpoints)
#
# Eval JSONL is read from weka (checkpoints/prasanns/longctx_sft_qwen/eval_jsonl), which the
# eval job already mounts for the checkpoint -- no extra dataset plumbing needed.
#
# Defaults assume a (landmark-attention) Qwen3-4B olmo-core checkpoint: olmo_core backend,
# Qwen/Qwen3-4B tokenizer, generation batch size 1 (landmark attention requires bs=1).
#
# Usage:
#   ./launch_longctx_task_evals.sh /weka/oe-training-default/ai2-llm/checkpoints/prasanns/<run>/stepXXXX
#
# Common overrides (env vars):
#   MODES="direct cot"        # which modes to run (default; use "cr" for base/plain models)
#   TASK_SETS="oolong contra" # which task families
#   PRIORITY, CLUSTER, NUM_GPUS, BATCH_SIZE, OLMO_CORE_TOKENIZER, WORKSPACE,
#   DASHBOARD (default memory-longctx-tasks), OE_EVAL_BRANCH (default prasann/longctx-eval),
#   OLMO_CORE_COMMIT (default: HEAD of this checkout), NAME_SUFFIX

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <weka_checkpoint_path>}"

PRIORITY="${PRIORITY:-urgent}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
BUDGET="ai2/oe-other"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"                       # landmark attention -> bs=1
OLMO_CORE_TOKENIZER="${OLMO_CORE_TOKENIZER:-Qwen/Qwen3-4B}"
DASHBOARD="${DASHBOARD:-memory-longctx-tasks}"
OE_EVAL_BRANCH="${OE_EVAL_BRANCH:-prasann/longctx-eval}"
MODES=( ${MODES:-direct cot} )
TASK_SETS=( ${TASK_SETS:-oolong contra} )
NAME_SUFFIX="${NAME_SUFFIX:-}"

# The checkpoints were written by OLMo-core@prasann/landmark (landmark attention configs +
# the landmark-generation max_length fix); eval jobs must install that exact code.
OLMO_CORE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT:-$(git -C "${OLMO_CORE_REPO}" rev-parse HEAD)}"

# The oe-eval gantry venv lacks flash-attn, which the olmo_core model requires (see
# launch_long_context_evals.sh for the wheel-variant rationale).
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"

INSTALL_CMD="uv sync --python 3.11 && uv pip install --no-deps ${FLASH_ATTN_WHEEL} && uv pip install --no-deps git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_COMMIT}"
GANTRY_ARGS_JSON="{\"install\": \"${INSTALL_CMD}\"}"

OOLONG_TASKS=()
for ctx in 1024 2048 4096 8192 16384 32768 65536; do
  OOLONG_TASKS+=( "longctx_oolong_synth_ctx${ctx}" )
done
CONTRA_TASKS=()
for n in 100 250 500 1000; do
  CONTRA_TASKS+=( "longctx_contradiction_pubmed_n${n}" )
done

for mode in "${MODES[@]}"; do
  for task_set in "${TASK_SETS[@]}"; do
    if [ "${task_set}" = "oolong" ]; then
      tasks=( "${OOLONG_TASKS[@]}" )
    else
      tasks=( "${CONTRA_TASKS[@]}" )
    fi
    TASK_FLAGS=()
    for t in "${tasks[@]}"; do
      TASK_FLAGS+=( -t "${t}::${mode}" )
    done

    suffix="${task_set}-${mode}${NAME_SUFFIX:+-${NAME_SUFFIX}}"
    echo "==> Launching ${task_set} (${#tasks[@]} tasks, mode=${mode}) for ${MODEL_PATH}"
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
      -l "${GANTRY_ARGS_JSON}" \
      --model-args "trust_remote_code=true,max_length=65536,tokenizer=${OLMO_CORE_TOKENIZER}"
  done
done

echo "All longctx task eval jobs submitted."
