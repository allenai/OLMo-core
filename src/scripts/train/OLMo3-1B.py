"""
Train a 1B OLMo model. Run this script without any arguments to see usage info.
"""
from datetime import datetime

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    BatchSizeSchedulerCallback,
    CheckpointerCallback,
    CometCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import TransformerTensorParallelConfig

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = (
    1024 * 8192
)  # batch size at step 0, let's keep this independent of the sequence length in case we change it.
MAX_DURATION = int(
    10e12
)  # Setting this higher than 6T (expected run time), in case we get to run longer since 1) we're using WSD and 2) our anneal will use different data
ANNEAL_TOKENS = int(100e9)
LR = (
    4.4e-5 * 2
)  # Based on 6T tokens with 100B anneal, don't forget to adjust when max duration or anneal length changes.
SAVE_INTERVAL = 10000
EVAL_INTERVAL = 1000


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_1B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=16,
        hidden_size_multiple_of=1024,
    )
    #  config.block.name = TransformerBlockType.default
    #  config.block.attention.qk_norm = None
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        # NOTE: 4097 instead of 4096 to reproduce with the off-by-one bug.
        pattern=[4097, 4097, 4097, -1],
    )
    config.block.attention.use_flex = True
    config.block.attention.use_sinks = False
    config.block.attention.use_head_qk_norm = True

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = 4 * SEQUENCE_LENGTH
    if common.launch is not None:
        gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
        if all("B200" in g for g in gpus):
            rank_microbatch_size *= 2

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=False,
            step_increment_bugfix=False,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=4,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=WSD(
            units=SchedulerUnits.steps,
            warmup=2000,
            decay=(
                int(ANNEAL_TOKENS / GLOBAL_BATCH_SIZE)
            ),  # TODO: This isn't right because it doesn't take batchwup into account.
            decay_fraction=None,
        ),
        tp_config=TransformerTensorParallelConfig(
            degree=4,
        )
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 50

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    config = (
        TrainerConfig(
            save_folder=f"gs://ai2-llm/checkpoints/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
            load_strategy=LoadStrategy.never,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                workspace="ai2",
                project="olmo3",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project="olmo3",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_recommended_evals(
            common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        )
    )

    # batch size warmup
    config.callbacks["batchwup"] = BatchSizeSchedulerCallback(
        batch_sizes=[GLOBAL_BATCH_SIZE, GLOBAL_BATCH_SIZE * 2, GLOBAL_BATCH_SIZE * 4],
        schedule=[
            Duration.tokens(0),
            Duration.tokens(167_772_160_000),
            Duration.tokens(503_316_480_000),
        ],
    )

    return config


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
        intra_document_masking=True,
    )
