"""
Smoke test for hybrid GDN + attention SFT training with think data.

Verifies that the hybrid architecture (3 GDN layers + 1 attention layer, repeating)
trains correctly in an SFT setting with packed FSL data and label masking.

Based on: src/scripts/train/OLMo_hybrid/OLMo-hybrid-7B-sft-think.py
Model: 190M hybrid (olmo3_190M base with GDN blocks swapped in)
Duration: 20 steps, 1 node

Dry run:
    python src/scripts/train/smoketests/hybrid-sft-think-test.py \
        dry_run test-hybrid-sft ai2/jupiter

Launch:
    python src/scripts/train/smoketests/hybrid-sft-think-test.py \
        launch test-hybrid-sft ai2/jupiter \
        --launch.priority=low \
        --launch.follow=false
"""

from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_beaker_username
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerConfig,
)
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 16 * SEQUENCE_LENGTH
DATASET_PATH = (
    "/weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/olmo-hybrid-sft-triple-tools"
)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo3_190M(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, AttentionConfig)

    num_heads = config.block.sequence_mixer.n_heads  # 12

    attn_block = config.block

    # Disable RoPE on attention blocks (matching the 7B hybrid script).
    attn_block = attn_block.replace(
        sequence_mixer=attn_block.sequence_mixer.replace(rope=None),
    )

    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=num_heads,
            head_dim=int(0.75 * config.d_model / num_heads),
            allow_neg_eigval=True,
        ),
    )

    # 3 GDN layers followed by 1 attention layer, repeating.
    config.block = {"gdn": gdn_block, "attn": attn_block}
    config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
    assert config.n_layers % len(config.block_pattern) == 0

    config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=2.5e-5,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            compile=False,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(
            warmup_fraction=0.03,
            alpha_f=0.0,
        ),
    )


def build_data_components(
    common: CommonComponents,
    dataset_path: str,
) -> DataComponents:
    clean_path = dataset_path.rstrip("/")
    dataset_config = NumpyPackedFSLDatasetConfig(
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        paths=[f"{clean_path}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=common.max_sequence_length,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=4
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            load_strategy=LoadStrategy.if_available,
            save_folder=f"{common.save_folder}/",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(20),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project=f"{get_beaker_username()}-hybrid-sft-smoketest",
                enabled=False,
                cancel_check_interval=5,
            ),
        )
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        beaker_workspace="ai2/oe-t-ladder",
        num_nodes=1,
        num_execution_units=1,
        dataset_path=DATASET_PATH,
    )
    main(config_builder=config_builder)
