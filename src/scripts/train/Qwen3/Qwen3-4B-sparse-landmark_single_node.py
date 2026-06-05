from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    LandmarkInstanceSourceConfig,
    NumpyDocumentSourceMixConfig,
)
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerTrainModuleConfig,
)

# Landmark attention block structure (identical to the original landmark script).
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 1024  # must be divisible by BLOCK_SIZE
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ

LANDMARK_TOKEN_ID = 151860  # reserved Qwen3 token used as the landmark token

GLOBAL_BATCH_SIZE = 1024
MAX_TOKENS = 10_000_000_000  # 10B
LR = 3.2e-4


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=1,
    )

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-4B with the FAST landmark attention sequence mixer (AttentionType.fast_landmark):
    # identical numerics to landmark+kernel, but the optimized FlashAttention-2-style backward
    # (atomic-free dk/dv + dq kernels) and retuned forward make it ~17-20x faster fwd+bwd.
    # Qwen3-4B with the SPARSE landmark attention mixer (AttentionType.sparse_landmark): a query
    # attends fully within its own chunk but sees past chunks only through their landmark tokens
    # (the last `num_landmarks` of each chunk). Sub-quadratic; num_landmarks trades capacity vs speed.
    # NOTE: num_landmarks=1 matches LandmarkInstanceSource's 1-landmark-per-chunk data (block_size
    # = mem_freq + num_landmarks = 64). num_landmarks>1 needs a data source inserting that many
    # landmark tokens per chunk.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        sparse_landmark=True,
        mem_freq=MEM_FREQ,
        num_landmarks=1,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=4, alpha_f=0.0),
        compile_model=True,
        # NOTE: context parallelism is not yet supported by SparseLandmarkAttention.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    instance_source_config = LandmarkInstanceSourceConfig(
        source=ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer_config,
                    mix=DataMix.longmino_qwen,
                    mix_base_dir="s3://ai2-llm",
                    source_group_size=8,
                )
            ],
            sequence_length=CONTENT_SEQUENCE_LENGTH,
        ),
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path="/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3-4B/model_and_optim",
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000, ephemeral_save_interval=None, max_checkpoints=3, save_async=True
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback("slack_notifier", SlackNotifierCallback(name=run_name_with_ts, enabled=False))
        .with_callback("config_saver", ConfigSaverCallback())
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=[instance_source_config],
        data_loader=data_loader_config,
    )
    return experiment_config.merge(cli_context.overrides)


if __name__ == "__main__":
    """
    Qwen3-4B with the SPARSE landmark attention mixer (sparse landmark-only-across-chunks
    attention). A query attends fully within its own chunk and to past chunks only via their
    landmark tokens, so it is sub-quadratic and faster than full/dense-landmark attention at long
    context. Same data pipeline as Qwen3-4B-landmark_single_node.py (num_landmarks=1).

    Examples:
        python src/scripts/train/Qwen3/Qwen3-4B-sparse-landmark_single_node.py dry_run my-run ai2/jupiter-cirrascale-2
        python src/scripts/train/Qwen3/Qwen3-4B-sparse-landmark_single_node.py launch my-run ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
