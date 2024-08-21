"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/train.py
"""

from glob import glob
from typing import Tuple

import torch

from olmo_core.data import DataCollator, IterableDataset, MemMapDataset
from olmo_core.nn.rope import RoPEType
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.optim import CosWithWarmup
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    SchedulerCallback,
    SpeedMonitorCallback,
)
from olmo_core.utils import get_default_device, has_flash_attn

# Tokenizer settings.
VOCAB_SIZE = 50304
EOS_TOKEN_ID = 50256
PAD_TOKEN_ID = 50256

# Model settings.
COMPILE = False
FUSED_OPS = not COMPILE and has_flash_attn()
ROPE_TYPE = RoPEType.default if COMPILE else None

# Trainer settings.
SAVE_FOLDER = "/tmp/run01"
DATA_FILES = "/net/nfs/allennlp/llm-data/c4/en/c4-train.*.npy"
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 128
SEED = 3423


def build_model() -> Tuple[Transformer, int]:
    model_config = TransformerConfig.llama2_271M(
        VOCAB_SIZE,
        fused_ops=FUSED_OPS,
        use_flash=not COMPILE,
        rope_type=ROPE_TYPE,
    )

    flops_per_token = model_config.num_flops_per_token(SEQUENCE_LENGTH)
    model = model_config.build(init_device="meta")

    # Activation checkpointing:
    #  model.apply_activation_checkpointing("full")

    # Maybe compile.
    if COMPILE:
        model.apply_compile()

    # FSDP or DDP.
    model.apply_fsdp2(param_dtype=torch.bfloat16)
    # OR
    #  model.apply_ddp2(compile_enabled=COMPILE)

    # Materialize and init parameters.
    model.to_empty(device=get_default_device())
    model.init_weights()

    return model, flops_per_token


def build_dataset() -> Tuple[IterableDataset, DataCollator]:
    paths = sorted(glob(DATA_FILES))
    assert paths
    collator = DataCollator(pad_token_id=PAD_TOKEN_ID)
    dataset = MemMapDataset(
        *paths,
        sequence_length=SEQUENCE_LENGTH,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )
    return IterableDataset(dataset, seed=SEED, drop_last=True), collator


def main():
    model, model_flops_per_token = build_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset, collator = build_dataset()

    trainer = (
        TrainerConfig(
            work_dir=SAVE_FOLDER,
            save_folder=SAVE_FOLDER,
            train_sequence_length=SEQUENCE_LENGTH,
            global_batch_size=BATCH_SIZE,
            microbatch_size=16,
            fused_loss=FUSED_OPS,
            autocast_precision=torch.bfloat16,
            save_overwrite=True,
            data_loader_workers=4,
            metrics_log_interval=5,
        )
        .with_callback(SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(GPUMemoryMonitorCallback())
        .with_callback(GradClipperCallback(max_grad_norm=1.0))
        .with_callback(CheckpointerCallback(save_interval=10_000, ephemeral_save_interval=250))
        .with_callback(SpeedMonitorCallback(num_flops_per_token=model_flops_per_token))
    ).build(model, optim, dataset, collator)

    trainer.fit()


if __name__ == "__main__":
    prepare_training_environment(seed=SEED, backend="nccl")
    try:
        main()
    finally:
        teardown_training_environment()
