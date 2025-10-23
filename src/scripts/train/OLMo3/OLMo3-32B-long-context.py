from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.io import join_path
from olmo_core.nn.attention import AttentionBackendName, SlidingWindowAttentionConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    MonkeyPatcherCallback,
    ProfilerCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 64 * 1024  # 64k seq len
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024  # ~8M tokens
MAX_TOKENS = 50_000_000_000  # 50B


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_32B(vocab_size=common.tokenizer.padded_vocab_size())
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[4096, 4096, 4096, -1],
    )
    config.block.attention.backend = AttentionBackendName.flash_2  # much faster for CP
    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = SEQUENCE_LENGTH
    if common.launch is not None:
        gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
        if all("B200" in g for g in gpus):
            rank_microbatch_size *= 2

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=6e-4,  # todo: set to appropriate value for the 32B
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=8,  # 64
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.3,
        ),
        # When CP is used, the CP mesh gets folded into the DP_shard mesh.
        cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4),  # 8
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%z')}"

    return (
        TrainerConfig(
            save_folder=f"gs://ai2-llm/checkpoints/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_TOKENS),
            hard_stop=Duration.steps(16),
        )
        .with_callback(
            "profiler",
            ProfilerCallback(enabled=False, skip_first=3, wait=10, warmup=2, active=3, repeat=1),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                enabled=False,  # todo turn back on for real thing
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project="olmo3",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(
                name=run_name,
                enabled=False,  # todo turn back on for real thing
            ),
        )
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    """
    Default dataset and data loader configurations. Constructs a simple FSL dataset and data loader
    configuration with default settings.
    """

    # dataset_config = NumpyPackedFSLDatasetConfig.glob(
    #     str(
    #         join_path(
    #             common.root_dir,
    #             "preprocessed/tylerr/lc-reshard-final/v0.6/allenai/dolma2-tokenizer/*.npy",
    #         )
    #     ),
    #     tokenizer=common.tokenizer,
    #     work_dir=common.work_dir,
    #     sequence_length=common.max_sequence_length,
    #     generate_doc_lengths=True,  # enables intra-document masking
    #     source_group_size=8,
    #     source_permutation_seed=123,
    #     instance_filter_config=None
    #     if not include_instance_filter
    #     else InstanceFilterConfig(
    #         repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
    #     ),
    # )

    dataset_config = NumpyFSLDatasetConfig.glob(
        str(
            join_path(
                common.root_dir,
                "preprocessed/tylerr/lc-reshard-final/v0.6/allenai/dolma2-tokenizer/*.npy",
            )
        ),
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        generate_doc_lengths=True,  # enables intra-document masking
        # source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        beaker_image="petew/olmo-core-tch280cu128-2025-09-19",
        include_instance_filter=True,
        include_default_evals=False,
    )

    main(config_builder=config_builder)
