#!/bin/bash
#SBATCH --job-name=ai2-slurm-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=/data/ai2/logs/%j/task_%t.log

USERNAME=petew
WORKSPACE=ai2/linear-rnns

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your Beaker username.
# * A Beaker workspace to pull env variables from.
# * All other args are passed to torchrun.
./src/scripts/lambda/entrypoint.sh $USERNAME $WORKSPACE \
    ./src/scripts/train/OLMo2/OLMo2-1B.py train slurm-test lambda \
    --dataset.mix_base_dir=/data/caia-mltrain/data/ \
    --dataset.mix=OLMo-mix-0925-official \
    --model.block.attention.backend=torch \
    --train_module.rank_microbatch_size=$((16 * 4096)) \
    --trainer.callbacks.wandb.enabled=false \
    --trainer.callbacks.comet.enabled=false \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --trainer.callbacks.downstream_evaluator.enabled=false \
    --trainer.no_checkpoints \
    --trainer.no_evals \
    --trainer.hard_stop='{unit: steps, value: 100}'
