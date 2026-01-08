#!/usr/bin/env bash
set -eo pipefail

uv run pytest src/test/nn/hf/checkpoint_test.py -xvs "$@"
