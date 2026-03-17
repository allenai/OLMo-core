"""
Official long-context extension training script for OLMo-hybrid-7B.

This extends the midtraining checkpoint to 65k sequence length by dropping RoPE
(DroPE) and using context parallelism with Ulysses (degree=2). Fused linear loss
is enabled to reduce memory usage at long sequence lengths.
"""

import argparse
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, StepSkipRange, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.00020712352850360292

REMOVE_HEADS = 2


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    # Build model config.
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert isinstance(model_config.block.sequence_mixer, AttentionConfig)

    # Remove heads (and scale down d_model) to compensate for extra params.
    model_config.d_model -= REMOVE_HEADS * 128
    num_heads = model_config.block.sequence_mixer.n_heads - REMOVE_HEADS
    model_config.block.sequence_mixer.n_heads = num_heads
    assert model_config.d_model / num_heads == 128

    attn_block = model_config.block

    # Drop RoPE (DroPE) from all layers at the start of long context training.
    attn_block = attn_block.replace(
        sequence_mixer=attn_block.sequence_mixer.replace(rope=None),
    )

    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=num_heads,
            head_dim=int(0.75 * model_config.d_model / num_heads),
            allow_neg_eigval=True,
        ),
    )

    # 3 GDN layers followed by 1 attention layer, repeating.
    model_config.block = {"gdn": gdn_block, "attn": attn_block}
    model_config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
    assert model_config.n_layers % len(model_config.block_pattern) == 0

    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        "gs://ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
        sequence_length=sequence_length,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=16,
        prefetch_factor=8,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        # Note: because we have 30 heads and are using Ulysses, we can only set degree to 2.
        # This means that LC training needs to be performed on B200s rather than H100s in order
        # to have enough HBM to fit the model + activations.
        cp_config=TransformerContextParallelConfig.ulysses(degree=2),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            # TODO: Update this path to a publicly accessible checkpoint URL.
            load_path="gs://ai2-llm/checkpoints/lambda/willm/linear-rnns/OLMo3.1-7B-6T-30h-midtrain-deux-soup/step23842/",
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            metrics_collect_interval=50,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
            # We began LC training with intra-document masking enabled, but ran into data-triggered
            # errors with varlen FLA operations. We skipped a few steps to continue training before
            # eventually disabling intra-document masking around step 1500.
            steps_to_skip=[StepSkipRange(start=961, stop=976)],
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
                fixed_steps=[960],
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
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
