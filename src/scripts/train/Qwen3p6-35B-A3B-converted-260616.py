"""
Qwen3.6-35B-A3B converted-checkpoint training on the Qwen3.6-retokenized shard.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["QWEN_MODEL_ID"] = "Qwen/Qwen3.6-35B-A3B"
os.environ["QWEN_TOKENIZER_ID"] = "Qwen/Qwen3.6-35B-A3B"
os.environ["QWEN_MODEL_SCALE"] = "full"
os.environ.setdefault("QWEN_LOAD_PATH", "/workspace/checkpoint/qwen3.6-35b-a3b-olmo")
os.environ["QWEN_DEFAULT_DATA_PATH"] = (
    "/workspace/tasks/june12/scratch/qwen3_6_loss/"
    "qwen36_retokenized_olmo_mix_0925_education_jobs_first1m.uint32.npy"
)
os.environ.setdefault("QWEN_EP_DIM", "8")
os.environ.setdefault("QWEN_EP_PATH", "rowwise_nvshmem")
os.environ.setdefault("QWEN_EP_CAPACITY_FACTOR", "8.0")
os.environ.setdefault("QWEN_PER_LAYER_RECOMPUTE", "true")
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
