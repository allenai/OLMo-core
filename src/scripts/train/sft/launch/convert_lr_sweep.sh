#!/bin/bash

# Converts checkpoints from the LR sweep experiments to HuggingFace format.
# Outputs go to oe-adapt-default to avoid overwriting originals in oe-training-default.

# LR 1e-4
gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    --weka=oe-training-default:/weka/oe-training-default \
    --priority urgent \
    --gpus 1 \
    -- /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
        -i /weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/TEST_HYBRIC_SFT_LARGER_LR1e-4/step46412 \
        -o /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR1e-4/step46412-hf \
        --max-sequence-length 32768 \
        --skip-validation

# LR 5e-5
gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    --weka=oe-training-default:/weka/oe-training-default \
    --priority urgent \
    --gpus 1 \
    -- /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
        -i /weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/TEST_HYBRIC_SFT_LARGER_LR5e-5/step46412 \
        -o /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR5e-5/step46412-hf \
        --max-sequence-length 32768 \
        --skip-validation

# LR 2.5e-5
gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    --weka=oe-training-default:/weka/oe-training-default \
    --priority urgent \
    --gpus 1 \
    -- /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
        -i /weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412 \
        -o /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf \
        --max-sequence-length 32768 \
        --skip-validation
