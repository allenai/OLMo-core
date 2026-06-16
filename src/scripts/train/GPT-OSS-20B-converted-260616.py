"""
GPT-OSS-20B converted-checkpoint training on the GPT-OSS-retokenized shard.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["GPT_OSS_MODEL_ID"] = "openai/gpt-oss-20b"
os.environ["GPT_OSS_TOKENIZER_ID"] = "openai/gpt-oss-20b"
os.environ["GPT_OSS_MODEL_SCALE"] = "full"
os.environ.setdefault("GPT_OSS_LOAD_PATH", "/workspace/checkpoint/gpt-oss-20b-olmo")
os.environ["GPT_OSS_DEFAULT_DATA_PATH"] = (
    "/workspace/tasks/june12/scratch/gpt_oss_20b_loss/"
    "gpt_oss_retokenized_olmo_mix_0925_education_jobs_first1m.uint32.npy"
)
os.environ.setdefault("GPT_OSS_EP_DIM", "8")
os.environ.setdefault("GPT_OSS_USE_EP_NO_SYNC", "true")
os.environ.setdefault("GPT_OSS_USE_ROWWISE_A2A", "true")
os.environ.setdefault("GPT_OSS_EP_NO_SYNC_CAPACITY_FACTOR", "8.0")
os.environ.setdefault("GPT_OSS_PER_LAYER_RECOMPUTE", "true")
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
