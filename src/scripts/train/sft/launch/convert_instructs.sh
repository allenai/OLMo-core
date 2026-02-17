#!/bin/bash

# Converts 0217 hybrid instruct SFT checkpoints to HuggingFace format
# and downloads the tokenizer into each output directory.

STEP=step3256
TOKENIZER_REPO=allenai/olmo-3-tokenizer-instruct-dev
BEAKER_IMAGE=tylerr/olmo-core-tch291cu128-2025-11-25

# LRS=(8e-5 5e-5 2.5e-5 1e-4 6e-5 3e-5 1.5e-5)
LRS=(5e-5 2.5e-5)

for LR in "${LRS[@]}"; do
    RUN_NAME="HYBRID_INSTRUCT_SFT_0217_${LR}"
    INPUT="/weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/${RUN_NAME}/${STEP}"
    OUTPUT="/weka/oe-adapt-default/nathanl/checkpoints/${RUN_NAME}/${STEP}-hf"

    gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
        --beaker-image "${BEAKER_IMAGE}" \
        --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --gpus 1 \
        -- bash -c "\
            /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
                -i ${INPUT} \
                -o ${OUTPUT} \
                --max-sequence-length 32768 \
                --skip-validation \
            && /opt/conda/bin/python -c \"from huggingface_hub import snapshot_download; snapshot_download('${TOKENIZER_REPO}', local_dir='${OUTPUT}')\" \
        "
done
