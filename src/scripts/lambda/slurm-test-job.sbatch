#!/bin/bash

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your (Beaker) username.
# * All other args are passed to torchrun.
srun --kill-on-bad-exit=1 ./src/scripts/lambda/entrypoint.sh ./src/scripts/lambda/distributed_test.py
