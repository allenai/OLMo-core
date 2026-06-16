"""
GPT-OSS-20B random-init training on Dolma data with the Dolma tokenizer.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ.pop("GPT_OSS_LOAD_PATH", None)
os.environ.pop("GPT_OSS_DATA_PATHS", None)
os.environ.pop("GPT_OSS_DEFAULT_DATA_PATH", None)
os.environ["GPT_OSS_MODEL_ID"] = "openai/gpt-oss-20b"
os.environ["GPT_OSS_TOKENIZER_ID"] = "dolma2"
os.environ.setdefault("GPT_OSS_MODEL_SCALE", "debug")
os.environ.setdefault("GPT_OSS_ATTENTION_BACKEND", "torch")
os.environ.setdefault("GPT_OSS_SEQUENCE_LENGTH", "512")
os.environ.setdefault("GPT_OSS_GLOBAL_BATCH_SEQS", "8")
os.environ.setdefault("GPT_OSS_MICRO_BATCH_SEQS", "1")
os.environ.setdefault("GPT_OSS_MAX_STEPS", "10")
os.environ.setdefault("GPT_OSS_DATA_NUM_WORKERS", "0")
os.environ.setdefault("GPT_OSS_CHECKPOINTER", "false")

runpy.run_path(
    str(Path(__file__).with_name("GPT-OSS-20B-dev-260614.py")),
    run_name="__main__",
)
