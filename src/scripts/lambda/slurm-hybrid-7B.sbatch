#!/bin/bash

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your (Beaker) username.
# * All other args are passed to torchrun.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    ./src/scripts/train/linear-rnns/OLMo3.1-7B-hybrid.py train OLMo3.1-7B-6T-30h lambda \
    --dataset.mix_base_dir="gs://ai2-llm/" \
    --train_module.dp_config.name=hsdp \
    --train_module.dp_config.shard_degree=128 \
    --model.block.attention.backend=flash_2

# Launch on 64 nodes
# Ideally we would use flash_3
