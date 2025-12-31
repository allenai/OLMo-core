#!/bin/bash

# Launch instructions:
# ssh into one of the login nodes.
# activate tmux session: linear-rnns-shared
# ./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-hybrid-7B-long-context.sbatch hybrid-OLMo3.1-7B-6T-30h-long-context 16

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your (Beaker) username.
# * All other args are passed to torchrun.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh \
    ./src/scripts/train/linear-rnns/OLMo3.1-7B-hybrid-long-context.py train OLMo3.1-7B-6T-30h-long-context lambda \
    --launch=null \
    --dataset.mix_base_dir="gs://ai2-llm/" \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir="gs://ai2-llm/" \
    --model.block.attention.backend=flash_2 \
    --trainer.callbacks.slack_notifier.enabled=true

# Launch on 16 nodes, shard degree 8
# Ideally we would use flash_3
