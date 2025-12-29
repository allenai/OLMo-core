from datetime import datetime
from typing import Optional

import torchvision  # noqa: F401

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_gpu_type, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import WSD, DionConfig
from olmo_core.train import Duration
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 8 * 1024
GLOBAL_BATCH_SIZE = 8 * 1024 * 64  # 524288
SEED = 34521


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}/"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace="ai2/OLMo-core",
        beaker_image=OLMoCoreBeakerImage.stable,
        num_nodes=1,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo3_190M(
        vocab_size=tokenizer_config.padded_vocab_size(), attn_backend=AttentionBackendName.torch
    )
    num_params = model_config.num_active_non_embedding_params
    d_model = model_config.d_model

    # Determine rank_microbatch_size based on GPU type
    rank_microbatch_size = SEQUENCE_LENGTH
    gpu_type = get_gpu_type(cli_context.cluster)
    if "B200" in gpu_type:
        rank_microbatch_size *= 2

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=DionConfig(
            lr=0.00194,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=WSD(warmup_steps=360),
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        tokenizer=tokenizer_config,
        mix_base_dir="gs://ai2-llm/",
        work_dir=work_dir,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(SEQUENCE_LENGTH, 8192),
        generate_doc_lengths=False,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,
        num_workers=8,
    )

    trainer_config = cookbook.configure_trainer(
        max_duration=Duration.chinchilla_tokens(1.0, model_params=num_params),
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    ).with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts,
            wandb_group_name=cli_context.run_name,
            wandb_project="olmo3-dion",
            checkpoint_save_interval=5000,
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/scripts/train/OLMo3/OLMo3-190M.py dry_run debug_run ai2/augusta

        To launch a training run on Augusta:
        python src/scripts/train/OLMo3/OLMo3-190M.py launch my_run ai2/augusta \
            --launch.num_nodes=1 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
