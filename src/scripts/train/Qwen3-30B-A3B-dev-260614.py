"""
Qwen/Qwen3-30B-A3B random-init or converted-checkpoint pretraining recipe.

Examples:
  python src/scripts/train/Qwen3-30B-A3B-dev-260614.py dry_run qwen3-30b-debug ai2/jupiter
  QWEN_MODEL_SCALE=full QWEN_MAX_LAYERS=2 python src/scripts/train/Qwen3-30B-A3B-dev-260614.py dry_run qwen3-30b-2l ai2/jupiter
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ.setdefault("QWEN_MODEL_ID", "Qwen/Qwen3-30B-A3B")
os.environ.setdefault("QWEN_TOKENIZER_ID", "Qwen/Qwen3-30B-A3B")
os.environ.setdefault(
    "QWEN_DATA_PATHS",
    "/workspace/tasks/june12/scratch/qwen3_30b_loss/"
    "qwen_retokenized_olmo_mix_0925_education_jobs_first1m.uint32.npy",
)

runpy.run_path(
    str(Path(__file__).with_name("Qwen3-moe-dev-260612.py")),
    run_name="__main__",
)
