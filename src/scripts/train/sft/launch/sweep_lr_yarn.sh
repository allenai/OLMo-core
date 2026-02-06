#!/bin/bash

BASE_CKPT="/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context/step23842/model_and_optim"

# # 5e-5 final model  - second base model
uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
    HYBRID_SFT_YARN_LR5e-5 $BASE_CKPT \
    ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=5e-5 \
    --launch.priority=urgent \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --dataset_path /weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/olmo-hybrid-sft-triple-tools

# # 2.5e-5 (2x smaller) - second base model
uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
    HYBRID_SFT_YARN_LR2.5e-5 $BASE_CKPT \
    ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=2.5e-5 \
    --launch.priority=urgent \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --dataset_path /weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/olmo-hybrid-sft-triple-tools
