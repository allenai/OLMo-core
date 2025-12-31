#!/bin/bash

# Launch instructions:
# ssh into one of the login nodes.
# activate tmux session: linear-rnns-shared
# ./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-hybrid-7B.sbatch hybrid-OLMo3.1-7B-6T-30h 64

# midtraining-deux = 4M tokens
GLOBAL_BATCH_SIZE=$((1024 * 1024 * 4))

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your (Beaker) username.
# * All other args are passed to torchrun.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    ./src/scripts/train/linear-rnns/OLMo3.1-7B-hybrid-midtraining.py train OLMo3.1-7B-6T-30h-midtrain-deux lambda \
    --launch=null \
    --dataset.mix_base_dir="gs://ai2-llm/" \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir="gs://ai2-llm/" \
    --train_module.dp_config.name=hsdp \
    --train_module.dp_config.shard_degree=8 \
    --model.block.attention.backend=flash_2 \
    --trainer.callbacks.slack_notifier.enabled=true \
    --dataset.source_mixture_config.global_batch_size=$GLOBAL_BATCH_SIZE \
    --data_loader.global_batch_size=$GLOBAL_BATCH_SIZE

# Launch on 16 nodes, shard degree 8
# Ideally we would use flash_3
