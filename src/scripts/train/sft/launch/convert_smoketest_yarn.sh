#!/bin/bash

# Convert YaRN smoketest checkpoint to HuggingFace format.

STEP=step4
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
BEAKER_IMAGE=tylerr/olmo-core-tch291cu128-2025-11-25

RUN_NAME="HYBRID_SFT_YARN_SMOKETEST"
INPUT="/weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/${RUN_NAME}/${STEP}"
OUTPUT="/weka/oe-adapt-default/nathanl/checkpoints/${RUN_NAME}/${STEP}-hf"

gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --beaker-image "${BEAKER_IMAGE}" \
    --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    --weka=oe-training-default:/weka/oe-training-default \
    --priority urgent \
    --gpus 1 \
    -- /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
        -i "${INPUT}" \
        -o "${OUTPUT}" \
        -t "${TOKENIZER}" \
        --max-sequence-length 32768 \
        --skip-validation
