#!/bin/bash

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and runs the command you give it with 'torchrun'.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    src/scripts/train/ladder/2026Q1/more_norms_ladder.py run \
    --name=olmo3-peri-norm \
    --cluster=lambda \
    --max-gpus=64 \
    --chinchilla-multiple=8.0 \
    --norm-style=peri \
    --embedding-norm \
    --size=190M
