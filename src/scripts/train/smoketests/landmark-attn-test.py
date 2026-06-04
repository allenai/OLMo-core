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
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# Landmark block structure: block_size=64, seq must be divisible by block_size.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQ_LENGTH = 4096  # 4k context (4096 / 64 = 64 blocks)
CONTENT_SEQ_LENGTH = SEQ_LENGTH // BLOCK_SIZE * MEM_FREQ  # 4032

# Qwen3 reserved token used as the landmark token (same as production script).
LANDMARK_TOKEN_ID = 151860

GLOBAL_BATCH_SIZE = SEQ_LENGTH * 16  # ~65k tokens


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
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
    )

    # Use Qwen3 tokenizer to match the longmino_qwen data mix.
    tokenizer_config = TokenizerConfig.qwen3()

    # OLMo2-190M architecture with landmark attention (block_size=64, fused Triton kernel).
    # olmo2_190M is used instead of olmo3_190M because sliding window is incompatible
    # with landmark attention.
    model_config = TransformerConfig.olmo2_190M(
        vocab_size=tokenizer_config.padded_vocab_size(),
        landmark=True,
        mem_freq=MEM_FREQ,
        landmark_use_kernel=True,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LENGTH,
        max_sequence_length=SEQ_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=2),
        compile_model=False,  # torch.compile incompatible with landmark boolean mask shapes
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
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
            sequence_length=CONTENT_SEQ_LENGTH,
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
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(20),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=False,
                cancel_check_interval=5,
            ),
        )
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
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Smoke test: verifies that landmark attention trains end-to-end on GPUs with the
    full composable data pipeline (LandmarkInstanceSource → ConcatAndChunk → longmino/qwen).

    Uses an OLMo2-190M backbone with landmark attention (mem_freq=63, block_size=64,
    fused Triton kernel) at a 4k context length. Runs 20 steps — enough to confirm
    the kernel, data pipeline, and FSDP all work together without OOM or errors.

    olmo2_190M is used instead of olmo3_190M because sliding window attention is
    incompatible with landmark attention.

    Examples:
        Dry run:
            python src/scripts/train/smoketests/landmark-attn-test.py \\
                dry_run test-landmark ai2/jupiter-cirrascale-2

        Launch (async):
            python src/scripts/train/smoketests/landmark-attn-test.py \\
                launch test-landmark ai2/jupiter-cirrascale-2 \\
                --launch.priority=normal \\
                --launch.follow=false
    """
    main(config_builder=build_experiment_config)
