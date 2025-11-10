from datetime import datetime

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config, AOFloat8LinearConfig
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import CommonComponents, main, build_launch_config
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig, SchedulerUnits
from olmo_core.train import Duration, TrainerConfig, LoadStrategy
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback, BatchSizeSchedulerCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

from beaker import Priority

SEQUENCE_LENGTH = 8 * 1024
INITIAL_GLOBAL_BATCH_SIZE = 4 * 1024 * 1024


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.llama3_8B_qknorm(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )
    config.block.attention.use_flash = True
    return config

def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = SEQUENCE_LENGTH 
    if common.launch is not None:
        gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
        if all("B200" in g for g in gpus):
            rank_microbatch_size *= 2
        
        
    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
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
        ),
        float8_config=Float8Config(
            enabled=False
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(
            units=SchedulerUnits.steps,    # mandatory with batch size warmup
            warmup_steps=2000
        ),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    common.launch.workspace='ai2/long-contexts'
    common.launch.budget='ai2/oe-base'
    common.launch.priority=Priority.urgent

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    config = (
        TrainerConfig(
            load_path="gs://ai2-llm/checkpoints/amandab/LlamaClone-8B/step0",
            load_strategy=LoadStrategy.always,
            save_folder=f"gs://ai2-llm/checkpoints/amandab/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(int(5e12)),
            hard_stop=Duration.tokens(int(150e9)), # stop at 10B tokens for this run 
            keys_to_ignore=[r'q_norm', r'k_norm'], # indicates that we're re-init-ing these keys-- tho actually they weren't in the prior init at all 

        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=500,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                workspace="ai2",
                project="long-contexts",
                enabled=False,
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
            common.tokenizer,
            SEQUENCE_LENGTH,
            cluster,
            task_set="fast"
        )
    )

    # batch size warmup
    config.callbacks["batchwup"] = BatchSizeSchedulerCallback(
        batch_sizes=[
            INITIAL_GLOBAL_BATCH_SIZE,
            INITIAL_GLOBAL_BATCH_SIZE * 2,
            INITIAL_GLOBAL_BATCH_SIZE * 4,
        ],
        schedule=[
            Duration.tokens(0),
            Duration.tokens(167_772_160_000),
            Duration.tokens(503_316_480_000),
        ]
    )

    return config

if __name__ == "__main__":
    main(
        global_batch_size=INITIAL_GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
        beaker_workspace="ai2/long-contexts"
    )
