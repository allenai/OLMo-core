from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
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
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import (
    LinearWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, LoadStrategy, StepSkipRange, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.00020712352850360292

REMOVE_HEADS = 2
INSTANCE_FILTER = True


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo3_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, AttentionConfig)

    # Remove heads (and scale down d_model) to compensate for extra params.
    config.d_model -= REMOVE_HEADS * 128
    num_heads = config.block.sequence_mixer.n_heads - REMOVE_HEADS
    config.block.sequence_mixer.n_heads = num_heads
    assert config.d_model / num_heads == 128

    attn_block = config.block

    # Drop RoPE (DroPE) from all layers at the start of long context training.
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
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        # Note: because we have 30 heads and are using Ulysses, we can only set degree to 2.
        # this means that LC training needs to be performed on B200s rather than H100s in order
        # to have enough HBM to fit the model + activations.
        cp_config=TransformerContextParallelConfig.ulysses(degree=2),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        "gs://ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        generate_doc_lengths=intra_document_masking,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=16,
        prefetch_factor=8,
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            load_path="gs://ai2-llm/checkpoints/lambda/willm/linear-rnns/OLMo3.1-7B-6T-30h-midtrain-deux-soup/step23842/",
            save_folder=f"{common.root_dir}/checkpoints/willm/linear-rnns/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_TOKENS),
            # We began LC training with intra-document masking enabled, but ran into data-triggered
            # errors with varlen FLA operations. We skipped a few steps to continue training before
            # eventually disabling intra-document masking around step 1500.
            steps_to_skip=[StepSkipRange(start=961, stop=976)],
        )
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
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project="linear-rnns",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
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
        include_instance_filter=INSTANCE_FILTER,
        beaker_workspace="ai2/linear-rnns",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
