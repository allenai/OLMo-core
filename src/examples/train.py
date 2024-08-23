"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/train.py
"""

import json
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
SAVE_FOLDER = "/tmp/run01"
DATA_FILES = "/net/nfs/allennlp/llm-data/c4/en/c4-train.*.npy"  # a glob
SEQUENCE_LENGTH = 1024
SEED = 3423
COMPILE = False

MODEL_CONFIG = TransformerConfig.llama2_271M(
    vocab_size=50304,
    fused_ops=not COMPILE and has_flash_attn(),
    use_flash=not COMPILE and has_flash_attn(),
    rope_type=RoPEType.default if COMPILE else None,
)

OPTIM_CONFIG = AdamWConfig(lr=1e-3)

TRAINER_CONFIG = (
    TrainerConfig(
        work_dir=SAVE_FOLDER,
        save_folder=SAVE_FOLDER,
        global_batch_size=256,
        microbatch_size=16,
        fused_loss=has_flash_attn(),
        autocast_precision=torch.bfloat16,
        save_overwrite=True,
        data_seed=SEED,
        data_loader_workers=4,
        metrics_log_interval=5,
    )
    .with_callback(SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
    .with_callback(GPUMemoryMonitorCallback())
    .with_callback(GradClipperCallback(max_grad_norm=1.0))
    .with_callback(
        CheckpointerCallback(
            save_interval=10_000,
            ephemeral_save_interval=250,
            save_async=True,
            pre_train_checkpoint=LOAD_PATH is None,
        )
    )
    .with_callback(
        SpeedMonitorCallback(num_flops_per_token=MODEL_CONFIG.num_flops_per_token(SEQUENCE_LENGTH))
    )
)


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
        eos_token_id=50256,
        pad_token_id=50256,
    )
    return dataset


def main():
    config_dict = dict(
        model=MODEL_CONFIG.as_config_dict(),
        optim=OPTIM_CONFIG.as_config_dict(),
        trainer=TRAINER_CONFIG.as_config_dict(),
        load_path=LOAD_PATH,
    )

    # Maybe add W&B callback.
    if WANDB_RUN is not None:
        TRAINER_CONFIG.with_callback(
            WandBCallback(
                name=WANDB_RUN,
                config=config_dict,
            )
        )

    # Build components.
    model = build_model(MODEL_CONFIG)
    optim = OPTIM_CONFIG.build(model)
    dataset = build_dataset()
    trainer = TRAINER_CONFIG.build(model, optim, dataset)

    # Save config to file.
    trainer.checkpointer.write_file(SAVE_FOLDER, "config.json", json.dumps(config_dict, indent=2))

    # Maybe load a checkpoint.
    if LOAD_PATH is not None:
        trainer.load_checkpoint(LOAD_PATH)

    # Train.
    trainer.fit()


if __name__ == "__main__":
    prepare_training_environment(seed=SEED)
    try:
        main()
    finally:
        teardown_training_environment()
