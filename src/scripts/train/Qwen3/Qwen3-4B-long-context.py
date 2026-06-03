from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, LoadStrategy
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
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

SEQUENCE_LENGTH = 65536  # 64k context
GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
MAX_TOKENS = 10_000_000_000  # 10B
# StepFun optimal LR (Li et al. 2025): 1.79 * n^-0.713 * d^0.307
# n ≈ 3.65B non-embedding params, d = 10B tokens → ~3.2e-4
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
        num_nodes=4,  # 4 nodes × 8 GPUs = 32 GPUs; cp_degree=8 → 4 DP replicas
        # override with --launch.num_nodes=N
    )

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-4B with YaRN context extension: native 32k → 64k
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # 1 instance per rank with CP
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=400, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        # Qwen3-4B: n_heads=32, n_kv_heads=8 → head_stride=4
        cp_config=TransformerContextParallelConfig.zig_zag(degree=8, head_stride=4),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    dataset_config = NumpyPackedFSLDatasetConfig.from_data_mix(
        DataMix.longmino_qwen,
        tokenizer=tokenizer_config,
        mix_base_dir="s3://ai2-llm",
        work_dir=work_dir,
        sequence_length=SEQUENCE_LENGTH,
        generate_doc_lengths=False,
        source_group_size=8,
        source_permutation_seed=123,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            # Set load_path to start from a pre-trained checkpoint, e.g.:
            load_path="/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3-4B/model_and_optim",
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            # load_optim_state=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=False,  # enable via --trainer.callbacks.wandb.enabled=true
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(name=run_name_with_ts, enabled=False),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Qwen3-4B long-context extension to 64k on the longmino dataset.
    Uses YaRN (factor=2) to extend the native 32k context to 64k.
    Data uses the dolma2/dolma3 tokenizer matching the longmino mix.

    Examples:
        Render the config and exit:
            python src/scripts/train/Qwen3/Qwen3-4B-long-context.py dry_run my-run ai2/jupiter-cirrascale-2

        Launch on Beaker with 4 nodes:
            python src/scripts/train/Qwen3/Qwen3-4B-long-context.py launch my-run ai2/jupiter-cirrascale-2

        Override node count or LR:
            python src/scripts/train/Qwen3/Qwen3-4B-long-context.py launch my-run ai2/jupiter-cirrascale-2 \\
                --launch.num_nodes=8 \\
                --train_module.optim.lr=1e-4

        Load from a checkpoint:
            python src/scripts/train/Qwen3/Qwen3-4B-long-context.py launch my-run ai2/jupiter-cirrascale-2 \\
                --trainer.load_path=gs://ai2-llm/checkpoints/.../stepXXXXX \\
                --trainer.load_strategy=if_available \\
                --trainer.load_trainer_state=false \\
                --trainer.load_optim_state=true
    """
    main(config_builder=build_experiment_config)
