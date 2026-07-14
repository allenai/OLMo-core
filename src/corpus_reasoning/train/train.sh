#!/bin/bash
# Thin wrapper around scripts/train/train.py kept for backwards compatibility
# with older sbatch scripts. New callers should use train.py directly:
#   python scripts/train/train.py <config.yml>
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$SCRIPT_DIR/../lib/common.sh"
setup_env
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

CONFIG="${1:?Usage: bash scripts/train/train.sh <config.yml>}"
python scripts/train/train.py "$CONFIG"
