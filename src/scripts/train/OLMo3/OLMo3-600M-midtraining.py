"""
Midtraining (continued pretraining) script for OLMo3-600M.

Applies the OLMo 3 midtraining recipe to the 600M ladder model checkpoint,
training on the dolmino midtraining data mix with a linear LR decay to zero.

Set MAX_TOKENS below to control the schedule:
  - 100_000_000_000  (100B tokens)
  - 10_000_000_000   (10B tokens)

Or override on the command line:
  --trainer.max_duration='tokens(10000000000)'

Examples:
    Dry run (print config and exit):
        python src/scripts/train/OLMo3/OLMo3-600M-midtraining.py \
            --save-folder /tmp/test \
            --data-root /weka/oe-training-default/ai2-llm \
            --dry-run

    Single-node training:
        torchrun --nproc-per-node=8 src/scripts/train/OLMo3/OLMo3-600M-midtraining.py \
            --save-folder /path/to/save \
            --data-root /weka/oe-training-default/ai2-llm

    100B schedule:
        torchrun --nproc-per-node=8 src/scripts/train/OLMo3/OLMo3-600M-midtraining.py \
            --save-folder /path/to/save/600M-midtrain-100B \
            --data-root /weka/oe-training-default/ai2-llm

    10B schedule (override MAX_TOKENS via CLI):
        torchrun --nproc-per-node=8 src/scripts/train/OLMo3/OLMo3-600M-midtraining.py \
            --save-folder /path/to/save/600M-midtrain-10B \
            --data-root /weka/oe-training-default/ai2-llm \
            --trainer.max_duration='tokens(10000000000)'
"""

import argparse
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**20  # ~1M tokens (scaled from 7B's ~2M to match 600M ladder target)
MAX_TOKENS = 100_000_000_000  # 100B  (change to 10_000_000_000 for 10B schedule)
SEED = 1337

LOAD_PATH = "/oe-eval-default/ai2-llm/checkpoints/model-ladders/olmo3-baseline-ladder/600M/step89187"

# Peak LR from the ladder's WSD schedule for 600M.
# Computed as: 0.0047 * (num_params / 108_000_000)^(-1/3) / 2.0
# For a proper midtrain the starting LR should match the checkpoint's current LR.
# You can read the exact value from the checkpoint with:
#   from olmo_core.distributed.checkpoint import load_state_dict
#   state = {"optim.param_groups.embeddings.weight.lr": None}
#   load_state_dict("<checkpoint>/model_and_optim", state)
#   print(state)
# The value below matches what the 600M long-context script uses for this checkpoint.
LR = 0.00020712352850360292


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo3_600M(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        mix=DataMix.OLMo_midtraining_mix_0625_100B,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 8192,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            load_path=LOAD_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            max_duration=Duration.tokens(MAX_TOKENS),
            work_dir=opts.work_dir,
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=SEED,
    ).merge(overrides)


if __name__ == "__main__":
    main(build_config)
