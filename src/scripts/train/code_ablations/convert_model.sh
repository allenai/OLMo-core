#!/usr/bin/env bash
set -eo pipefail

# Convert an OLMo-core checkpoint to HuggingFace format using Beaker/Gantry.
#
# Usage:
#   ./convert_model.sh s3://bucket/path/to/checkpoint
#
# The converted model will be uploaded to the same S3 path with "-hf" appended.

if [ -z "$1" ]; then
    echo "Usage: $0 <s3-checkpoint-path>"
    echo "Example: $0 s3://ai2-llm/checkpoints/user/model/step1000"
    exit 1
fi

MODEL_PATH="$1"

# Remove trailing slash if present
MODEL_PATH="${MODEL_PATH%/}"

# Compute output path by appending -hf
OUTPUT_PATH="${MODEL_PATH}-hf"

echo "Converting checkpoint: ${MODEL_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"

uv run gantry run \
    --workspace ai2/olmo4 \
    --budget ai2/oe-base \
    --priority high \
    --gpus 0 \
    --yes \
    --timeout -1 \
    --show-logs \
    -- bash -c "
        set -eo pipefail
        TMP_INPUT=\$(mktemp -d)
        TMP_OUTPUT=\$(mktemp -d)

        echo 'Downloading checkpoint from ${MODEL_PATH} to \${TMP_INPUT}...'
        aws s3 cp --recursive '${MODEL_PATH}' \"\${TMP_INPUT}\"

        echo 'Converting checkpoint...'
        uv run src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i \"\${TMP_INPUT}\" \
            -o \"\${TMP_OUTPUT}\" \
            -s 8192 \
            -t allenai/dolma2-tokenizer \
            --skip-validation

        echo 'Uploading converted model to ${OUTPUT_PATH}...'
        aws s3 cp --recursive \"\${TMP_OUTPUT}\" '${OUTPUT_PATH}'

        echo 'Cleaning up...'
        rm -rf \"\${TMP_INPUT}\" \"\${TMP_OUTPUT}\"

        echo 'Done!'
    "
