#!/usr/bin/env bash
#
# Launch the `emo_1b14b_1t` run (OLMoE-1B-7B with EMO routing) on Beaker.
#
# Submits `src/scripts/train/OLMoE-1B-7B-emo.py` via `python -m olmo_core.launch.beaker`, which
# clones this repo at the current commit into the container and runs it under torchrun. Make sure
# the EMO code is committed (and pushed) before launching.
#
# Cluster sizing / paths can be overridden with environment variables, e.g.:
#   NODES=4 PRIORITY=high ./src/scripts/train/launch_emo_1b14b_1t.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Run name + paths (on the shared weka filesystem)
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-emo_1b14b_1t}"
OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-training-default/ryanwang/TestEMO}"
SAVE_FOLDER="${SAVE_FOLDER:-${OUTPUT_DIR}/${RUN_NAME}}"
WORK_DIR="${WORK_DIR:-/weka/oe-training-default/ryanwang/dataset-cache}"
DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"

# ---------------------------------------------------------------------------
# Beaker cluster configuration
# ---------------------------------------------------------------------------
NODES="${NODES:-2}"
GPUS="${GPUS:-8}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
WEKA="${WEKA:-oe-training-default}"

# Environment variables sourced from Beaker secrets (NAME=SECRET_NAME). Adjust the secret names on
# the right to match your Beaker workspace if they differ.
ENV_SECRETS=(
    "GITHUB_TOKEN=RYAN_GITHUB_TOKEN"
    "WANDB_API_KEY=RYAN_WANDB_API_KEY"
    "BEAKER_TOKEN=RYAN_BEAKER_TOKEN"
    "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY"
    "HF_TOKEN=RYAN_HF_TOKEN"
)

# ---------------------------------------------------------------------------
# EMO routing hyperparameters (emo_1b14b_1t)
# ---------------------------------------------------------------------------
LR="${LR:-4e-3}"
MIN_POOL="${MIN_POOL:-8}"
MAX_POOL="${MAX_POOL:-128}"
EVAL_POOL="${EVAL_POOL:-32}"
NUM_SHARED="${NUM_SHARED:-1}"

# Weights & Biases logging (set WANDB_ENABLED=false to disable).
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_ENTITY="${WANDB_ENTITY:-ryanyxw}"
WANDB_PROJECT="${WANDB_PROJECT:-olmoe-modular}"

SCRIPT="src/scripts/train/OLMoE-1B-7B-emo.py"

# Set BEAKER_DRY_RUN=1 to print the resolved Beaker launch config without submitting.
BEAKER_DRY_RUN_FLAG=()
if [[ "${BEAKER_DRY_RUN:-0}" != "0" ]]; then
    BEAKER_DRY_RUN_FLAG=(--dry-run)
fi

# ---------------------------------------------------------------------------
# Submit to Beaker.
#   Flags before `--` configure the Beaker launch; everything after `--` is the training command
#   (Beaker wraps it with torchrun automatically for multi-GPU jobs).
# ---------------------------------------------------------------------------
python -m olmo_core.launch.beaker \
    "${BEAKER_DRY_RUN_FLAG[@]}" \
    --name "${RUN_NAME}" \
    --nodes "${NODES}" \
    --gpus "${GPUS}" \
    --workspace "${WORKSPACE}" \
    --cluster "${CLUSTER}" \
    --priority "${PRIORITY}" \
    --weka "${WEKA}" \
    --shared-filesystem \
    --env-secret "${ENV_SECRETS[@]}" \
    -- \
    "${SCRIPT}" "${RUN_NAME}" \
    --model-type=emo \
    --save-folder="${SAVE_FOLDER}" \
    --work-dir="${WORK_DIR}" \
    --data-root="${DATA_ROOT}" \
    --lr="${LR}" \
    --min-document-expert-pool="${MIN_POOL}" \
    --max-document-expert-pool="${MAX_POOL}" \
    --eval-document-expert-pool="${EVAL_POOL}" \
    --num-shared-experts="${NUM_SHARED}" \
    --global-load-balancing \
    --model.block.feed_forward_moe.num_experts=128 \
    --model.block.name=moe \
    --model.block.sequence_mixer.qk_norm=null \
    --model.block.sequence_mixer.backend=flash_2 \
    --model.block.feed_forward_moe.lb_loss_weight=1e-1 \
    --dataset.generate_doc_lengths=true \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --trainer.max_duration='{value: 1_000_000_000_000, unit: tokens}' \
    --trainer.callbacks.wandb.enabled="${WANDB_ENABLED}" \
    --trainer.callbacks.wandb.entity="${WANDB_ENTITY}" \
    --trainer.callbacks.wandb.project="${WANDB_PROJECT}" \
    --trainer.callbacks.wandb.name="${RUN_NAME}"
