#!/bin/bash

# Requires gantry to be installed: uv tool install beaker-gantry

gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --system --extra fla --extra transformers" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    --weka=oe-training-default:/weka/oe-training-default \
    --priority urgent \
    --gpus 1 \
    -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
        -i /weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/HYBRID_SFT_SMOKETEST/step4 \
        -o /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_SFT_SMOKETEST/step4-hf \
        --max-sequence-length 32768 \
        --skip-validation
