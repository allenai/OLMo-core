"""
Example of how to train a transformer language model.
"""

from glob import glob
from typing import Tuple

import torch

from olmo_core.data import DataCollator, IterableDataset, MemMapDataset
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.optim import ConstantScheduler
from olmo_core.train import TrainerConfig, prepare_training_environment
from olmo_core.train.callbacks import (
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    SchedulerCallback,
)
from olmo_core.utils import get_default_device

# Tokenizer settings.
VOCAB_SIZE = 50304
EOS_TOKEN_ID = 50256
PAD_TOKEN_ID = 50256

# Model settings.
COMPILE = False
FUSED_OPS = False

# Trainer settings.
SAVE_FOLDER = "/tmp/run01"
DATA_FILES = "/net/nfs/allennlp/llm-data/c4/en/c4-train.*.npy"
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 128
SEED = 3423


def build_model() -> Transformer:
    model = TransformerConfig.llama2_271M(VOCAB_SIZE, fused_ops=FUSED_OPS and not COMPILE).build(
        init_device="meta"
    )

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

    return model


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
    model = build_model()
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
        )
        .with_callback(SchedulerCallback(scheduler=ConstantScheduler()))
        .with_callback(GPUMemoryMonitorCallback())
        .with_callback(GradClipperCallback(max_grad_norm=1.0))
    ).build(model, optim, dataset, collator)

    trainer.fit()


if __name__ == "__main__":
    prepare_training_environment(seed=SEED, backend="nccl")
    main()
