#!/bin/bash

size=${1:-190M}

srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    src/scripts/train/ladder/2026Q1/more_norms_ladder.py run \
    --name=olmo3-peri-ln \
    --cluster=lambda \
    --max-gpus=256 \
    --chinchilla-multiple=8.0 \
    --norm-style=peri \
    --embedding-norm \
    --size="${size}"
