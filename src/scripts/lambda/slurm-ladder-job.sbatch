#!/bin/bash

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your (Beaker) username.
# * All other args are passed to torchrun.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    ./src/scripts/train/ladder/olmo3_ladder.py run \
    --name=olmo3-baseline-ladder-test \
    --cluster=lambda \
    --max-gpus=64 \
    --chinchilla_multiple=8.0 \
    --size=370M