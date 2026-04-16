"""
Smoke test: idealized overlap pretraining (custom RoPE + tree-structured attention).

Trains a 190M model for 20 steps on idealized overlap data (read from S3 directly)
to verify the end-to-end pipeline works:
    - PreChunkedInstanceSource loads parallel (tokens, pos_ids, vis_limit) .npy files
    - Collator forwards pos_ids and vis_limit
    - Custom RoPE positions computed from pos_ids
    - Dense attention mask constructed from vis_limit
    - torch SDPA backend consumes the mask (Flash backends don't support arbitrary masks)

Success: 20 steps complete with finite, decreasing loss.

Examples:
    Dry run:
        python src/scripts/train/smoketests/idealized-overlap-test.py \\
            dry_run test-idealized-overlap ai2/jupiter

    Launch:
        python src/scripts/train/smoketests/idealized-overlap-test.py \\
            launch test-idealized-overlap ai2/jupiter \\
            --launch.priority=high \\
            --launch.follow=false
"""

from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    InstanceSourceConfig,
    PreChunkedInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import (
    build_launch_config,
    get_root_dir,
    get_work_dir,
)
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

SEQ_LENGTH = 8192

# Read data directly from S3 while the Weka transfer is still in progress.
IDEALIZED_OVERLAP_S3_BASE = (
    "s3://ai2-llm/suffix-arrays/preprocessed/"
    "dolma2-0625-v01/idealized-overlap-max-suffix-8192-wo-replacement/"
    "allenai/dolma2-tokenizer"
)


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    # get_gpu_type is unused here — we force torch backend below.
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace="ai2/oe-data",
        num_nodes=1,
        nccl_debug=False,
    )

    tokenizer_config = TokenizerConfig.dolma2()

    # Force torch backend and full attention: tree-structured masks are not
    # supported by the Flash backends.
    model_config = TransformerConfig.olmo3_190M(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.torch,
        sliding_window=None,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LENGTH * 2,
        max_sequence_length=SEQ_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        # torch.compile + dense attn_mask interact poorly; disable for verification.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )

    instance_sources: list[InstanceSourceConfig] = [
        PreChunkedInstanceSourceConfig(
            token_paths=[f"{IDEALIZED_OVERLAP_S3_BASE}/*-tokens.npy"],
            pos_ids_paths=[f"{IDEALIZED_OVERLAP_S3_BASE}/*-pos_ids.npy"],
            vis_limit_paths=[f"{IDEALIZED_OVERLAP_S3_BASE}/*-vis_limit.npy"],
            sequence_length=SEQ_LENGTH,
        ),
    ]

    data_loader_config = ComposableDataLoaderConfig(
        global_batch_size=SEQ_LENGTH * 16,
        seed=34521,
        num_workers=4,
        instance_filter_config=None,
        work_dir=work_dir,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(20),
        )
        .with_callbacks(configure_required_callbacks(run_name_with_ts))
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="olmo3",
                enabled=False,
            ),
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=instance_sources,
        data_loader=data_loader_config,
        init_seed=1337,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
