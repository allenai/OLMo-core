#!/usr/bin/env bash
set -euo pipefail
bash src/examples/olmo_ddp/olmoe3_hero_decay_setup.sh
uv pip install --python "$(command -v python)" --no-deps \
  'olmo-checkpoint-uploader @ git+https://github.com/jacob-morrison/olmo-checkpoint-uploader.git@50069318bd7b6bcfed655a8a01d2892e56b7abff'
