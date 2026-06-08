#!/usr/bin/env bash
# Launch SFT for the OLMo hybrid (GatedDeltaNet) 7B "think" model.
#
# Positional args to the python script:  <cmd> <run_name> <pretrain_checkpoint> <cluster>
# Everything after that is either a recognized flag (--seq_len/--num_nodes/--budget/
# --workspace/--dataset_path/--global_batch_size) or a dotted config override
# (--train_module.optim.lr=..., --launch.priority=..., --trainer.callbacks.wandb.enabled=...).
set -euo pipefail

BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842
DATASET_PATH=/weka/oe-adapt-default/saumyam/Dolci-Think-SFT-32B-75pct-olmo-tokenizer
CLUSTER=ai2/jupiter
LEARNING_RATE=1e-4
SEQ_LEN=32768
NUM_NODES=4
RUN_NAME=olmo-7b-hybrid-75pct-mix-lr-${LEARNING_RATE}

python src/scripts/train/OLMo_hybrid/OLMo-hybrid-7B-sft-think-train.py launch \
    "${RUN_NAME}" \
    "${BASE_CKPT}" \
    "${CLUSTER}" \
    --dataset_path="${DATASET_PATH}" \
    --seq_len="${SEQ_LEN}" \
    --num_nodes="${NUM_NODES}" \
    --budget=ai2/oe-other \
    --workspace=ai2/olmo-instruct \
    --launch.num_gpus=8 \
    --launch.priority=urgent \
    --launch.follow=false \
    --train_module.optim.lr="${LEARNING_RATE}" \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2
