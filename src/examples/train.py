"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/train.py
"""

from glob import glob

import torch

from olmo_core.data import MemMapDataset
from olmo_core.nn.rope import RoPEType
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
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
    WandBCallback,
)
from olmo_core.utils import get_default_device, has_flash_attn

LOAD_PATH = None  # path to a checkpoint folder
WANDB_RUN = None  # name of W&B run

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
BATCH_SIZE = 256
DEVICE_MICRO_BATCH_SIZE = 16
SEED = 3423


def build_model(model_config: TransformerConfig) -> Transformer:
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

    return model


def build_dataset() -> MemMapDataset:
    paths = sorted(glob(DATA_FILES))
    assert paths
    dataset = MemMapDataset(
        *paths,
        sequence_length=SEQUENCE_LENGTH,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )
    return dataset


def main():
    model_config = TransformerConfig.llama2_271M(
        VOCAB_SIZE,
        fused_ops=FUSED_OPS,
        use_flash=not COMPILE,
        rope_type=ROPE_TYPE,
    )

    optim_config = AdamWConfig(lr=1e-3)

    trainer_config = (
        TrainerConfig(
            work_dir=SAVE_FOLDER,
            save_folder=SAVE_FOLDER,
            global_batch_size=BATCH_SIZE,
            microbatch_size=DEVICE_MICRO_BATCH_SIZE,
            fused_loss=FUSED_OPS,
            autocast_precision=torch.bfloat16,
            save_overwrite=True,
            data_seed=SEED,
            data_loader_workers=4,
            metrics_log_interval=5,
        )
        .with_callback(SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(GPUMemoryMonitorCallback())
        .with_callback(GradClipperCallback(max_grad_norm=1.0))
        .with_callback(CheckpointerCallback(save_interval=10_000, ephemeral_save_interval=250))
        .with_callback(
            SpeedMonitorCallback(
                num_flops_per_token=model_config.num_flops_per_token(SEQUENCE_LENGTH)
            )
        )
    )

    if WANDB_RUN is not None:
        trainer_config.with_callback(
            WandBCallback(
                name=WANDB_RUN,
                config=dict(
                    model=model_config.as_config_dict(),
                    optim=optim_config.as_config_dict(),
                    trainer=trainer_config.as_config_dict(),
                ),
            )
        )

    model = build_model(model_config)
    optim = optim_config.build(model)
    dataset = build_dataset()
    trainer = trainer_config.build(model, optim, dataset)

    if LOAD_PATH is not None:
        trainer.load_checkpoint(LOAD_PATH)

    trainer.fit()


if __name__ == "__main__":
    prepare_training_environment(seed=SEED)
    try:
        main()
    finally:
        teardown_training_environment()
