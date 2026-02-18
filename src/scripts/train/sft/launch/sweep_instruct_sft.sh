#!/bin/bash

BASE_CKPT="/weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412"

# 8e-5 (best instruct sft)
# uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
#     HYBRID_INSTRUCT_SFT_0217_8e-5 $BASE_CKPT \
#     ai2/jupiter \
#     --trainer.callbacks.wandb.enabled=True \
#     --trainer.max_duration.value=2 \
#     --train_module.optim.lr=8e-5 \
#     --launch.priority=urgent \
#     --seq_len=32768 \
#     --init_seed=42 \
#     --launch.num_gpus=8 \
#     --num_nodes=8 \
#     --budget ai2/oe-adapt \
#     --workspace ai2/olmo-instruct \
#     --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

# # 5e-5 (another good instruct sft)
# uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
#     HYBRID_INSTRUCT_SFT_0217_5e-5 $BASE_CKPT \
#     ai2/jupiter \
#     --trainer.callbacks.wandb.enabled=True \
#     --trainer.max_duration.value=2 \
#     --train_module.optim.lr=5e-5 \
#     --launch.priority=urgent \
#     --seq_len=32768 \
#     --init_seed=42 \
#     --launch.num_gpus=8 \
#     --num_nodes=8 \
#     --budget ai2/oe-adapt \
#     --workspace ai2/olmo-instruct \
#     --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

# 2.5e-5
# uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
#     HYBRID_INSTRUCT_SFT_0217_2.5e-5 $BASE_CKPT \
#     ai2/jupiter \
#     --trainer.callbacks.wandb.enabled=True \
#     --trainer.max_duration.value=2 \
#     --train_module.optim.lr=2.5e-5 \
#     --launch.priority=urgent \
#     --seq_len=32768 \
#     --init_seed=42 \
#     --launch.num_gpus=8 \
#     --num_nodes=8 \
#     --budget ai2/oe-adapt \
#     --workspace ai2/olmo-instruct \
#     --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

# uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
#     HYBRID_INSTRUCT_SFT_0217_1e-4 $BASE_CKPT \
#     ai2/jupiter \
#     --trainer.callbacks.wandb.enabled=True \
#     --trainer.max_duration.value=2 \
#     --train_module.optim.lr=1e-4 \
#     --launch.priority=urgent \
#     --seq_len=32768 \
#     --init_seed=42 \
#     --launch.num_gpus=8 \
#     --num_nodes=8 \
#     --budget ai2/oe-adapt \
#     --workspace ai2/olmo-instruct \
#     --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

# 9e-5
uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
    HYBRID_INSTRUCT_SFT_0217_9e-5 $BASE_CKPT \
    ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=9e-5 \
    --launch.priority=urgent \
    --seq_len=32768 \
    --init_seed=42 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114


# uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
#     HYBRID_INSTRUCT_SFT_0217_6e-5 $BASE_CKPT \
#     ai2/jupiter \
#     --trainer.callbacks.wandb.enabled=True \
#     --trainer.max_duration.value=2 \
#     --train_module.optim.lr=6e-5 \
#     --launch.priority=urgent \
#     --seq_len=32768 \
#     --init_seed=42 \
#     --launch.num_gpus=8 \
#     --num_nodes=8 \
#     --budget ai2/oe-adapt \
#     --workspace ai2/olmo-instruct \
#     --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

# uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
#     HYBRID_INSTRUCT_SFT_0217_3e-5 $BASE_CKPT \
#     ai2/jupiter \
#     --trainer.callbacks.wandb.enabled=True \
#     --trainer.max_duration.value=2 \
#     --train_module.optim.lr=3e-5 \
#     --launch.priority=urgent \
#     --seq_len=32768 \
#     --init_seed=42 \
#     --launch.num_gpus=8 \
#     --num_nodes=8 \
#     --budget ai2/oe-adapt \
#     --workspace ai2/olmo-instruct \
#     --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

# uv run python src/scripts/train/sft/Olmo-3-Hybrid-7B-SFT.py launch \
#     HYBRID_INSTRUCT_SFT_0217_1.5e-5 $BASE_CKPT \
#     ai2/jupiter \
#     --trainer.callbacks.wandb.enabled=True \
#     --trainer.max_duration.value=2 \
#     --train_module.optim.lr=1.5e-5 \
#     --launch.priority=urgent \
#     --seq_len=32768 \
#     --init_seed=42 \
#     --launch.num_gpus=8 \
#     --num_nodes=8 \
#     --budget ai2/oe-adapt \
#     --workspace ai2/olmo-instruct \
#     --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114
