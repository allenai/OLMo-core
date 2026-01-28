#!/bin/bash

BASE_CKPT="/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842/model_and_optim"

uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
    HYBRID_SFT_SMOKETEST $BASE_CKPT \
    ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=5e-5 \
    --launch.priority=urgent \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --dataset_path /weka/oe-adapt-default/nathanl/dataset/sft-tokenization-test
