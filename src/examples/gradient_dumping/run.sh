#!/bin/bash
# Simple example: train with gradient dumping enabled

torchrun --nproc-per-node=4 src/examples/llm/train.py \
  gradient_dumping_example \
  --save-folder=/tmp/gradient_dumping_example \
  --work-dir=/tmp/dataset-cache \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.hard_stop='{value: 25, unit: steps}' \
  --trainer.callbacks.grad_dump.enabled=true \
  --trainer.callbacks.grad_dump.start_step=0 \
  --trainer.callbacks.grad_dump.step_interval=2 \
  --trainer.callbacks.grad_dump.end_step=10 \
  --train_module.dp_config.name="fsdp"

# python -m olmo_core.launch.beaker \
#   --gpus=4 \
#   --weka=oe-training-default \
#   --shared-filesystem \
#   --allow-dirty \
#   --workspace=ai2/caia \
#   --priority=high \
#   -- src/examples/llm/train.py \
#   gradient_dumping_example \
#   --save-folder=/weka/oe-training-default/kf/gradient_dumping_example \
#   --work-dir=/weka/oe-training-default/kf/dataset-cache \
#   --trainer.callbacks.lm_evaluator.enabled=false \
#   --trainer.callbacks.downstream_evaluator.enabled=false \
#   --trainer.no_checkpoints \
#   --trainer.hard_stop='{value: 25, unit: steps}' \
#   --trainer.callbacks.grad_dump.enabled=true \
#   --trainer.callbacks.grad_dump.start_step=0 \
#   --trainer.callbacks.grad_dump.step_interval=2 \
#   --trainer.callbacks.grad_dump.end_step=10 \
#   --train_module.dp_config.name="fsdp"