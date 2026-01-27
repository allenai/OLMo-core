#!/usr/bin/env bash
# This script launches a single-node experiment for OLMo2-1B training using the specified cluster and run name.
# Usage:
#  ./src/scripts/test/single_node_experiment.sh
run_name=test-run-wandb-saving
cluster=ai2/jupiter
uv run python src/scripts/train/OLMo2/OLMo2-1B.py launch "$run_name" "$cluster" \
  --launch.num_nodes=1 \
  --launch.priority=high \
  --launch.follow=false \
  --launch.launch_timeout=$((24 * 60 * 60)) \
  --launch.workspace=ai2/OLMo-pretraining-stability \
  --trainer.callbacks.wandb.enabled=true \
  --trainer.callbacks.comet.enabled=true \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.no_evals \
  --trainer.hard_stop='{unit: steps, value: 20}'
