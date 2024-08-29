"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/train.py
"""

import json

from olmo_core.config import DType
from olmo_core.data import MemMapDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.utils import get_rank, init_hybrid_shard_mesh
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
from olmo_core.utils import get_default_device

LOAD_PATH = None  # path to a checkpoint folder
WANDB_RUN = None  # name of W&B run
SAVE_FOLDER = "/tmp/run01"
DATA_FILES = ["/net/nfs/allennlp/llm-data/c4/en/c4-train.*.npy"]  # can be globs
SEED = 3423

TOKENIZER_CONFIG = TokenizerConfig.gpt2()

MODEL_CONFIG = TransformerConfig.llama2_271M(
    vocab_size=TOKENIZER_CONFIG.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    compile=False,
    dp_config=DataParallelConfig(
        name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
    ),
)

OPTIM_CONFIG = AdamWConfig(lr=1e-3)

DATASET_CONFIG = MemMapDatasetConfig.glob(
    *DATA_FILES,
    sequence_length=1024,
    tokenizer=TOKENIZER_CONFIG,
)

TRAINER_CONFIG = (
    TrainerConfig(
        work_dir=SAVE_FOLDER,
        save_folder=SAVE_FOLDER,
        global_batch_size=256,
        microbatch_size=16,
        autocast_precision=DType.bfloat16,
        save_overwrite=True,
        data_seed=SEED,
        data_loader_workers=4,
        metrics_collect_interval=5,
    )
    .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
    .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
    .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
    .with_callback(
        "checkpointer",
        CheckpointerCallback(
            save_interval=1000,
            ephemeral_save_interval=50,
            save_async=True,
            pre_train_checkpoint=LOAD_PATH is None,
        ),
    )
    .with_callback(
        "speed_monitor",
        SpeedMonitorCallback(
            num_flops_per_token=MODEL_CONFIG.num_flops_per_token(DATASET_CONFIG.sequence_length)
        ),
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
            "wandb",
            WandBCallback(
                name=WANDB_RUN,
                config=config_dict,
            ),
        )

    # Build components.
    model = MODEL_CONFIG.build(
        init_device="meta",
        device=get_default_device(),
        dp_mesh=init_hybrid_shard_mesh(num_replicas=2),
    )
    optim = OPTIM_CONFIG.build(model)
    dataset = DATASET_CONFIG.build()
    trainer = TRAINER_CONFIG.build(model, optim, dataset)

    # Save config to file.
    if get_rank() == 0:
        trainer.checkpointer.write_file(
            SAVE_FOLDER, "config.json", json.dumps(config_dict, indent=2)
        )

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
