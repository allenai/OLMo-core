#!/bin/bash

run_name=1B-test
srun --kill-on-bad-exit=1 ./src/scripts/prime-intellect/entrypoint.sh \
    ./src/scripts/train/OLMo2/OLMo2-1B.py train $run_name prime-intellect \
    --trainer.save_folder="/data/checkpoints/$USERNAME/$run_name" \
    --trainer.callbacks.wandb.enabled=false \
    --trainer.callbacks.comet.enabled=false \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --trainer.callbacks.downstream_evaluator.enabled=false \
    --trainer.no_checkpoints \
    --trainer.no_evals \
    --trainer.hard_stop='{unit: steps, value: 20}' "$@"
