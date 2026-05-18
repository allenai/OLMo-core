#!/bin/bash

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and runs the command you give it with 'torchrun'.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh ./src/scripts/lambda/distributed_test.py
