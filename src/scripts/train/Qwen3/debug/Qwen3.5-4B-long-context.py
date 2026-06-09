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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# Qwen3.5-4B is a hybrid Gated DeltaNet + full-attention model. Unlike Qwen3-4B-long-context.py,
# we do NOT apply YaRN: `with_rope_scaling` is rejected for hybrid models with named blocks, and
# the linear-attention (GDN) layers carry long-range state natively. We simply train at a long
# sequence length on the longmino mix.
SEQUENCE_LENGTH = 65536  # 64k context
GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
MAX_TOKENS = 10_000_000_000  # 10B
LR = 3.2e-4

# TODO: set to your Qwen3.5-4B olmo-core checkpoint (the `model_and_optim` directory).
# Override at launch with --trainer.load_path=...
CHECKPOINT_PATH = "FILL_ME"


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
        num_nodes=4,  # override with --launch.num_nodes=N
    )

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3.5-4B: hybrid Gated DeltaNet (linear attention) + full attention, 3:1 ratio.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # 1 sequence per rank per micro-step
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
        # GatedDeltaNet layers use custom kernels; keep compile off until verified.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        # NOTE: no Ulysses context parallelism here. Ulysses splits the sequence across ranks,
        # which is incompatible with the GatedDeltaNet recurrence; each rank processes a full
        # 64k sequence. Only 1/4 of layers are full-attention, so memory is dominated by the
        # GDN layers + activation checkpointing rather than quadratic attention.
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
            load_path=CHECKPOINT_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
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
                enabled=True,
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
    Qwen3.5-4B (hybrid Gated DeltaNet + full attention) long-context training on longmino at 64k.

    The 3.5 analogue of Qwen3-4B-long-context.py. Unlike that script, it does NOT use YaRN — the
    hybrid model rejects `with_rope_scaling`, and the GatedDeltaNet layers handle long range
    natively — and it omits Ulysses context parallelism (incompatible with the GDN recurrence),
    relying on FSDP/HSDP + activation checkpointing for a full 64k sequence per rank.

    Set CHECKPOINT_PATH (or --trainer.load_path) to a Qwen3.5-4B olmo-core checkpoint before
    launching.

    Examples:
        Render the config and exit:
            python src/scripts/train/Qwen3/Qwen3.5-4B-long-context.py dry_run my-run ai2/jupiter

        Launch on Beaker with 4 nodes, pointing at a checkpoint:
            python src/scripts/train/Qwen3/Qwen3.5-4B-long-context.py launch my-run ai2/jupiter \\
                --trainer.load_path=/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3.5-4B/model_and_optim

        Override node count or LR:
            python src/scripts/train/Qwen3/Qwen3.5-4B-long-context.py launch my-run ai2/jupiter \\
                --launch.num_nodes=8 --train_module.optim.lr=1e-4
    """
    main(config_builder=build_experiment_config)
