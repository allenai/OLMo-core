#!/bin/bash

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your (Beaker) username.
# * All other args are passed to torchrun.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    src/scripts/train/ladder/2026Q1/gatted_attn_ladder.py run \
    --name=olmo3-gated-attn \
    --cluster=lambda \
    --max-gpus=64 \
    --chinchilla-multiple=8.0 \
    --size=7B
# 7B requires 64 GPUs but can run on 128 or 256 GPUs