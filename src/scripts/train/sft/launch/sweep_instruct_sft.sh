#!/bin/bash

BASE_CKPT="/weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/TEST_HYBRIC_SFT_LARGER_LR5e-5/step46412"

# 8e-5 (best instruct sft)
uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
    HYBRID_INSTRUCT_SFT_8e-5 $BASE_CKPT \
    ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=8e-5 \
    --launch.priority=urgent \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

# # 5e-5 (another good instruct sft)
uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
    HYBRID_INSTRUCT_SFT_5e-5 $BASE_CKPT \
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
    --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

