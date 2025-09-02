#!/bin/bash

echo "Input: gs://ai2-llm/checkpoints/OLMo25-from476838/step500680"
echo "Output: gs://ai2-llm/checkpoints/OLMo25-from476838/step500680-unsharded"

python unshard_checkpoint.py \
    --input gs://ai2-llm/checkpoints/OLMo25-from476838/step500680 \
    --output gs://ai2-llm/checkpoints/OLMo25-from476838/step500680-unsharded \
    --overwrite

if [ $? -eq 0 ]; then
    echo "Checkpoint unsharding completed successfully!"
else
    exit 1
fi