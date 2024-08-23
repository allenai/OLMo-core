"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/train.py
"""

import json

import torch

from olmo_core.data import MemMapDatasetConfig
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.nn.transformer import TransformerConfig
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
DATA_FILES = ["/net/nfs/allennlp/llm-data/c4/en/c4-train.*.npy"]  # can be globs
SEED = 3423

MODEL_CONFIG = TransformerConfig.llama2_271M(
    vocab_size=50304,  # a little big than actual vocab size to make it a multiple of 128
    compile=False,
    dp_config=DataParallelConfig(
        name=DataParallelType.fsdp, param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    ),
)

OPTIM_CONFIG = AdamWConfig(lr=1e-3)

DATASET_CONFIG = MemMapDatasetConfig.glob(
    *DATA_FILES,
    sequence_length=1024,
    eos_token_id=50256,
    pad_token_id=50256,
)

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
        SpeedMonitorCallback(
            num_flops_per_token=MODEL_CONFIG.num_flops_per_token(DATASET_CONFIG.sequence_length)
        )
    )
)


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
    model = MODEL_CONFIG.build(init_device="meta", device=get_default_device())
    optim = OPTIM_CONFIG.build(model)
    dataset = DATASET_CONFIG.build()
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
