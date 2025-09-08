"""
Train a 7B OLMo model on long contexts. Run this script without any arguments to see usage info.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride
from olmo_core.optim.adamw import SkipStepAdamWConfig
from olmo_core.train import LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.callbacks.profiler import ProfilerCallback
from olmo_core.train.common import Duration
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import TransformerTensorParallelConfig

log = logging.getLogger(__name__)


CONTEXT_LENGTH = 4 * 16_384
INTRA_DOCUMENT_MASKING = False
# 64K length, 32 GPUs, FP8, no intra-doc masking -> 2,750 TPS
# 64K length, 32 GPUs, no FP8, intra-doc masking -> 3,250 TPS
# 64K length, 32 GPUs, FP8, intra-doc masking    -> 3,500 TPS


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )
    # config.block.attention.sliding_window = SlidingWindowAttentionConfig(
    #     force_full_attention_on_first_layer=False,
    #     force_full_attention_on_last_layer=True,
    #     # NOTE: 4097 instead of 4096 to reproduce with the off-by-one bug.
    #     pattern=[4096, 4096, 4096, -1],
    # )
    config.block.attention.use_flex = True
    config.block.attention.use_sinks = True
    config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=1 * CONTEXT_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=1e-5,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            foreach=True,
        ),
        compile_model=True,
        z_loss_multiplier=1e-5,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=4,
        ),
        float8_config=Float8Config(enabled=False),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
        tp_config=TransformerTensorParallelConfig(
            degree=8,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.5,
        ),
        state_dict_load_opts={"strict": False},
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            max_duration=Duration.steps(25),
            load_path='gs://ai2-llm/checkpoints/OLMo25-from476838/step500680-unsharded',
            load_strategy=LoadStrategy.always,

        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                enabled=False,
                save_interval=10_000,
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-7B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-7B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    #     .with_callback(
    #         "profiler",
    #         ProfilerCallback(
    #             enabled=True,
    #             wait=15,
    #             warmup=2,
    #             active=2,
    #             repeat=1,
    #             skip_first=3,
    #         ),
    #     )
    )


if __name__ == "__main__":
    main(
        sequence_length=CONTEXT_LENGTH,
        global_batch_size=64 * CONTEXT_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        intra_document_masking=INTRA_DOCUMENT_MASKING,
    )