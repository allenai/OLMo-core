#!/bin/bash

LOCAL_OUTPUT="/data/input/amanr/OLMo-core/checkpoints/step500680-unsharded"
GCS_OUTPUT="gs://ai2-llm/checkpoints/OLMo25-from476838/step500680-unsharded"
mkdir -p $(dirname $LOCAL_OUTPUT)

python unshard_checkpoint.py \
    --input gs://ai2-llm/checkpoints/OLMo25-from476838/step500680/model_and_optim \
    --output $LOCAL_OUTPUT \
    --overwrite

if [ $? -eq 0 ]; then
    echo "Checkpoint unsharding completed successfully!"
    gsutil -m cp -r $LOCAL_OUTPUT $GCS_OUTPUT
    if [ $? -eq 0 ]; then
    else
        exit 1
    fi
else
    exit 1
fi