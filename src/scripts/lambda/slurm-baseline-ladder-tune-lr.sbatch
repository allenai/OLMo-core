#!/bin/bash

lr_multiplier=${1:-1.0}

size=760M
# rank_mbz=8  # set based on rank mbz for H100s
    # --rank-mbz="${rank_mbz}" \

srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    src/scripts/train/ladder/olmo3_ladder.py run \
    --name="olmo3-${lr_multiplier}xLR" \
    --project=olmo3-ladder-tune-lr \
    --cluster=lambda \
    --max-gpus=64 \
    --lr-multiplier="${lr_multiplier}" \
    --chinchilla-multiple=1.0 \
    --size="${size}"
