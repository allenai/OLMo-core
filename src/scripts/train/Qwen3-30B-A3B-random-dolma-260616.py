"""
Qwen3-30B-A3B random-init training on Dolma data with the Dolma tokenizer.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ.pop("QWEN_LOAD_PATH", None)
os.environ.pop("QWEN_DATA_PATHS", None)
os.environ.pop("QWEN_DEFAULT_DATA_PATH", None)
os.environ["QWEN_MODEL_ID"] = "Qwen/Qwen3-30B-A3B"
os.environ["QWEN_TOKENIZER_ID"] = "dolma2"
os.environ.setdefault("QWEN_MODEL_SCALE", "debug")
os.environ.setdefault("QWEN_ATTENTION_BACKEND", "torch")
os.environ.setdefault("QWEN_SEQUENCE_LENGTH", "512")
os.environ.setdefault("QWEN_GLOBAL_BATCH_SEQS", "8")
os.environ.setdefault("QWEN_MICRO_BATCH_SEQS", "1")
os.environ.setdefault("QWEN_MAX_STEPS", "10")
os.environ.setdefault("QWEN_DATA_NUM_WORKERS", "0")
os.environ.setdefault("QWEN_CHECKPOINTER", "false")

runpy.run_path(
    str(Path(__file__).with_name("Qwen3-moe-dev-260612.py")),
    run_name="__main__",
)
