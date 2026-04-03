#!/bin/bash
#
# Launch Qwen3 4B SFT on the 100k tokenized dataset (4 nodes).
#
# Prerequisites:
#   1. Convert checkpoint from HF (if not already done):
#      uv run gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
#          --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
#          --weka=oe-adapt-default:/weka/oe-adapt-default \
#          --weka=oe-training-default:/weka/oe-training-default \
#          --priority urgent --gpus 1 \
#          -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
#              -i Qwen/Qwen3-4B-Base \
#              -o /weka/oe-adapt-default/jacobm/repos/cse-579/checkpoints/Qwen3-4B-Base-oc \
#              -c src/scripts/train/sft/qwen3-4b-config.json \
#              --skip-validation
#
#   2. Ensure the 100k dataset is tokenized at:
#      /weka/oe-adapt-default/jacobm/repos/cse-579/datasets/Dolci-Think-SFT-32B-qwen3-olmo-thinker-100k
#
# Usage:
#   bash src/scripts/train/sft/launch-qwen3-4b-sft.sh
#
set -euo pipefail

# --- Configuration ---
RUN_NAME=qwen3-4b-sft-100k
BASE_CKPT=/weka/oe-adapt-default/jacobm/repos/cse-579/checkpoints/Qwen3-4B-Base-oc/model_and_optim
CLUSTER=ai2/saturn
DATASET_PATH=/weka/oe-adapt-default/jacobm/repos/cse-579/datasets/Dolci-Think-SFT-32B-qwen3-olmo-thinker-100k

python src/scripts/train/sft/Qwen3-4B-SFT.py launch \
    ${RUN_NAME} \
    ${BASE_CKPT} \
    ${CLUSTER} \
    --dataset_path=${DATASET_PATH} \
    --budget=ai2/oe-adapt \
    --workspace=ai2/flex2 \
    --num_nodes=4 \
    --launch.priority=urgent \
    --launch.preemptible=true
