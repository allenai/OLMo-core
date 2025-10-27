"""
Train a 1B OLMo model. Run this script without any arguments to see usage info.
"""

import os

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.parametrization import MupScalingStrategy, ParametrizationConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks.comet import CometCallback
from olmo_core.train.callbacks.parametrization_coord_data import (
    ParametrizationCoordDataCallback,
)
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.common import Duration
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 64
COLLECTION_STEP = 100
D_MODEL_MULTIPLIER = int(os.environ.get("D_MODEL_MULTIPLIER", "1"))


def build_model_config(
    common: CommonComponents,
) -> TransformerConfig:
    # return TransformerConfig.olmo2_190M(
    #     vocab_size=common.tokenizer.padded_vocab_size(),
    # )
    return TransformerConfig.llama_like(
        d_model=4 * D_MODEL_MULTIPLIER,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=2,
        n_heads=2,
        hidden_size_multiplier=1.5,
        hidden_size_multiple_of=2,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        fused_ops=False,
        use_flash=False,
        parametrization=ParametrizationConfig(
            scaling_strategy=MupScalingStrategy.constant_inputs,
            width_scalings={
                WidthHyperParam.d_model: D_MODEL_MULTIPLIER,
                WidthHyperParam.hidden_size: D_MODEL_MULTIPLIER,
                WidthHyperParam.head_dim: D_MODEL_MULTIPLIER,
            },
        ),
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=1 * SEQUENCE_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=5e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=False,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=1),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=f"../data/checkpoints/{common.run_name}_{D_MODEL_MULTIPLIER}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            no_checkpoints=True,
            no_evals=True,
            hard_stop=Duration.steps(COLLECTION_STEP + 1),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-1B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-1B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "parametrization_coord_data",
            ParametrizationCoordDataCallback(
                enabled=True,
                collection_step=COLLECTION_STEP,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=16 * SEQUENCE_LENGTH,
        sequence_length=SEQUENCE_LENGTH,
        include_default_evals=False,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
    )
