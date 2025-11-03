#!/bin/bash
# Simple example: train with gradient dumping enabled

torchrun --nproc-per-node=2 src/examples/llm/train.py \
  gradient_dumping_example \
  --save-folder=gs://allennlp-kevinf/ \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.hard_stop='{value: 15, unit: steps}' \
  --train_module.dp_config.name="fsdp" \
  --trainer.callbacks.grad_dump.enabled=true \
  --trainer.callbacks.grad_dump.start_step=0 \
  --trainer.callbacks.grad_dump.step_interval=2 \
  --trainer.callbacks.grad_dump.end_step=10 \
  --trainer.callbacks.grad_dump.save_first_n=1000 \
  # --work-dir=gs://allennlp-kevinf/ \
  # --model-factory "olmo2_7B"


# python -m olmo_core.launch.beaker \
#   --name="grad_dumper_olmo2_7B_nodes_2_gpus_4_with_overrides" \
#   --gpus=4 \
#   --nodes=2 \
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
#   --train_module.dp_config.name="fsdp" \
#   --trainer.callbacks.grad_dump.enabled=true \
#   --trainer.callbacks.grad_dump.start_step=0 \
#   --trainer.callbacks.grad_dump.step_interval=2 \
#   --trainer.callbacks.grad_dump.end_step=10 \
#   --model-factory "olmo2_7B"
