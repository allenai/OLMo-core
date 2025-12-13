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
    --not-preemptible \
    --system-python \
    --uv-all-extras \
    -- \
    bash -c "HELION_AOT_AUTOTUNE=create HELION_AUTOTUNE_PROGRESS_BAR=false python $FILE_TO_RUN || cp -r $CONFIG_DIR/* /results/"

# To extract the best config from gantry logs, use:
# gantry logs 01KCB8KTG47EAYN8ND1FWN0B58 | grep -A 4 "One can hardcode the best config and skip autotuning with:"