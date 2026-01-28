#!/bin/bash

MODEL_SIZES=(
#   "260m"
#   "709m"
#   "1.3b"
#   "2b"
  "4b"
#   "8b"
)
CLUSTER="ai2/jupiter"

for size in "${MODEL_SIZES[@]}"; do
  echo "Launching model size: $size"
  uv run python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch \
    ri-olmo-v1-${size}-4xC \
    ${CLUSTER} \
    --trainer.callbacks.wandb.enabled=true \
    --launch.priority="urgent" \
    --launch.follow=false \
    --chinchilla-multiple=4.0
done