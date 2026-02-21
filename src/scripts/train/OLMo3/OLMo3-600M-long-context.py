"""
Long-context extension training script for OLMo3-600M.

Extends context from 8192 -> 65536 using YaRN RoPE scaling on the 600M ladder model.
Based on the OLMo3-7B-long-context recipe, scaled down for 600M.

Examples:
    Dry run (print config and exit):
        python src/scripts/train/OLMo3/OLMo3-600M-long-context.py \
            --save-folder /tmp/test \
            --data-root /weka/oe-training-default/ai2-llm \
            --dry-run

    Train on a single node (for debugging):
        torchrun --nproc-per-node=8 src/scripts/train/OLMo3/OLMo3-600M-long-context.py \
            --save-folder /path/to/save \
            --data-root /weka/oe-training-default/ai2-llm

    Multi-node training:
        torchrun --nproc-per-node=8 --nnodes=N --node-rank=RANK ... \
            src/scripts/train/OLMo3/OLMo3-600M-long-context.py \
            --save-folder /path/to/save \
            --data-root /weka/oe-training-default/ai2-llm
"""

import argparse
import math
from typing import List, Optional

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
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
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 65536
OLD_CONTEXT_LEN = 8192
YARN_FACTOR = DEFAULT_SEQUENCE_LENGTH // OLD_CONTEXT_LEN  # 8

GLOBAL_BATCH_SIZE = 65536 * 16  # ~1M tokens (scaled down from 7B's ~4M)
MAX_TOKENS = 10_000_000_000  # 10B tokens (scaled down from 7B's 50B)
LR = 0.00020712352850360292

MAX_TOKENS_PER_RANK = 32_768  # 600M fits more tokens per rank than 7B

LOAD_PATH = "/oe-eval-default/ai2-llm/checkpoints/model-ladders/olmo3-baseline-ladder/600M/step89187"


def _compute_cp_degree(sequence_length: int, world_size: int) -> Optional[int]:
    """Auto-compute context parallelism degree based on sequence length and world size."""
    if sequence_length <= MAX_TOKENS_PER_RANK:
        return None

    min_cp = 2
    while (sequence_length // min_cp) > MAX_TOKENS_PER_RANK:
        min_cp *= 2

    cp_degree = min(min_cp, world_size)
    if world_size % cp_degree != 0:
        cp_degree = int(2 ** math.floor(math.log2(world_size)))

    return cp_degree if cp_degree > 1 else None


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo3_600M(
        vocab_size=tokenizer_config.padded_vocab_size(),
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(
            factor=YARN_FACTOR,
            beta_fast=32,
            beta_slow=1,
            old_context_len=OLD_CONTEXT_LEN,
        )
    )

    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        f"{opts.data_root}/preprocessed/tylerr/lc-reshard-final/v0.6/allenai/dolma2-tokenizer/*.npy",
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
        sequence_length=sequence_length,
        generate_doc_lengths=True,
        source_group_size=8,
        source_permutation_seed=123,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    import torch
    world_size = max(torch.cuda.device_count(), 1)
    cp_degree = _compute_cp_degree(sequence_length, world_size)

    cp_config: Optional[TransformerContextParallelConfig] = None
    if cp_degree is not None:
        cp_config = TransformerContextParallelConfig.llama3(degree=cp_degree)

    dp_world_size = world_size // (cp_degree or 1)
    if dp_world_size <= 1:
        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        )
    else:
        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=min(dp_world_size, 8),
        )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        ),
        scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
        compile_model=True,
        dp_config=dp_config,
        cp_config=cp_config,
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
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
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
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
        .with_callback("config_saver", ConfigSaverCallback())
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


if __name__ == "__main__":
    main(build_config)
