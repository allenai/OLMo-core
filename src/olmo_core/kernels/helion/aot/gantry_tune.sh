#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <file_to_run>"
    exit 1
fi


FILE_TO_RUN="$1"
FILE_DIR=$(dirname "$FILE_TO_RUN")
CONFIG_DIR="$FILE_DIR/aot/configs"


gantry run \
    -w ai2/OLMo-core \
    -b ai2/oe-base \
    --show-logs \
    --gpu-type=h100 \
    --gpus=1 \
    --beaker-image=tylerr/olmo-core-tch291cu128-2025-11-25 \
    --priority=normal \
    --preemptible \
    --system-python \
    --uv-all-extras \
    -- \
    bash -c "python $FILE_TO_RUN || cp -r $CONFIG_DIR/* /results/"