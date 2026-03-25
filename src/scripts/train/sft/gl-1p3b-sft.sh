#!/bin/bash
# SFT for the GemmaLike 1.3B ladder model (gl-1p3b-dolma-2xc-v2).
#
# Usage:
#   bash src/scripts/train/sft/gl-1p3b-sft.sh
#
# Set DATASET_PATH before running.

CHECKPOINT=/weka/oe-training-default/ai2-llm/checkpoints/kylel/olm4_mixing_calibration/gl-1p3b-dolma-2xc-v2/step42905/model_and_optim
DATASET_PATH=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-only
LR=8e-5

python src/scripts/train/sft/GemmaLike-1p3B-SFT.py launch gl-1p3b-instruct-SFT-${LR} \
    $CHECKPOINT \
    ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=8192 \
    --num_nodes=2 \
    --global_batch_size=524288 \
    --budget=ai2/oe-adapt \
    --workspace=ai2/oe-data \
    --dataset_path=$DATASET_PATH \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --launch.priority=urgent
