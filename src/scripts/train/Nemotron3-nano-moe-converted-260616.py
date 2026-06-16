"""
Nemotron3 Nano MoE converted-checkpoint training on the Nemotron-retokenized shard.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["NEMOTRON_MODEL_ID"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
os.environ["NEMOTRON_TOKENIZER_ID"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
os.environ["NEMOTRON_MODEL_SCALE"] = "full"
os.environ.setdefault("NEMOTRON_LOAD_PATH", "/workspace/checkpoint/nemotron3-nano-30b-a3b-olmo")
os.environ["NEMOTRON_DEFAULT_DATA_PATH"] = (
    "/workspace/tasks/june12/scratch/nemotron3_nano_loss/"
    "nemotron_retokenized_olmo_mix_0925_education_jobs_first1m.uint32.npy"
)
os.environ.setdefault("NEMOTRON_DATA_PARALLEL", "fsdp")
os.environ.setdefault("NEMOTRON_ATTENTION_BACKEND", "torch")
os.environ.setdefault("NEMOTRON_SEQUENCE_LENGTH", "512")
os.environ.setdefault("NEMOTRON_GLOBAL_BATCH_SEQS", "8")
os.environ.setdefault("NEMOTRON_MICRO_BATCH_SEQS", "1")
os.environ.setdefault("NEMOTRON_MAX_STEPS", "10")
os.environ.setdefault("NEMOTRON_DATA_NUM_WORKERS", "0")
os.environ.setdefault("NEMOTRON_CHECKPOINTER", "false")

runpy.run_path(
    str(Path(__file__).with_name("Nemotron3-nano-moe-dev-260613.py")),
    run_name="__main__",
)
