from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration

SEQ_LENGTH = 8192
MAX_TOKENS = 10_000_000_000  # 10B
SEED = 1337


def estimate_lr(num_params: int, chinchilla_multiple: float = 4.0) -> float:
    """Estimate LR based on stable in WSD"""
    return 0.0047 * (num_params / 108_000_000) ** (-1 / 3) / chinchilla_multiple**0.5


def estimate_gbs(num_params: int) -> int:
    """Estimate global batch size, rounded down to a multiple of SEQ_LENGTH."""
    raw = round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))
    return (raw // SEQ_LENGTH) * SEQ_LENGTH


@dataclass
class LadderSize:
    model_config_fn: Callable[..., TransformerConfig]
    lr: float
    global_batch_size: int


SIZES: dict[str, LadderSize] = {
    "60M": LadderSize(
        model_config_fn=TransformerConfig.olmo3_60M,
        lr=estimate_lr(60_000_000),
        global_batch_size=estimate_gbs(60_000_000),
    ),
    "100M": LadderSize(
        model_config_fn=TransformerConfig.olmo3_100M,
        lr=estimate_lr(100_000_000),
        global_batch_size=estimate_gbs(100_000_000),
    ),
    "600M": LadderSize(
        model_config_fn=TransformerConfig.olmo3_600M,
        lr=estimate_lr(600_000_000),
        global_batch_size=estimate_gbs(600_000_000),
    ),
    "760M": LadderSize(
        model_config_fn=TransformerConfig.olmo3_760M,
        lr=estimate_lr(760_000_000),
        global_batch_size=estimate_gbs(760_000_000),
    ),
    "1B": LadderSize(
        model_config_fn=TransformerConfig.olmo3_1B,
        lr=estimate_lr(1_000_000_000),
        global_batch_size=estimate_gbs(1_000_000_000),
    ),
    "3B": LadderSize(
        model_config_fn=TransformerConfig.olmo3_3B,
        lr=estimate_lr(3_000_000_000),
        global_batch_size=estimate_gbs(3_000_000_000),
    ),
}


def _pop_overrides(overrides: list[str]) -> tuple[str, Optional[str], list[str]]:
    """Extract ``--size`` and ``--load-path`` from the overrides list."""
    size: Optional[str] = None
    load_path: Optional[str] = None
    remaining: list[str] = []
    for arg in overrides:
        if arg.startswith("--size="):
            size = arg.split("=", 1)[1]
        elif arg.startswith("--load-path="):
            load_path = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
    if size is None:
        raise ValueError(f"--size is required. Choices: {list(SIZES.keys())}")
    if size not in SIZES:
        raise ValueError(f"Invalid size '{size}'. Choices: {list(SIZES.keys())}")
    return size, load_path, remaining


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    size, load_path, overrides = _pop_overrides(cli_context.overrides)
    cfg = SIZES[size]

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
        workspace="ai2/OLMo_3",
        num_nodes=1,
        nccl_debug=False,
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = cfg.model_config_fn(vocab_size=tokenizer_config.padded_vocab_size())

    train_module_config = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH,
        learning_rate=cfg.lr,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=0, alpha_f=0.0),
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        mix=DataMix.OLMo_midtraining_mix_0625_100B,
        tokenizer=tokenizer_config,
        mix_base_dir="s3://ai2-llm",
        sequence_length=SEQ_LENGTH,
        max_target_sequence_length=max(8192, SEQ_LENGTH),
        work_dir=work_dir,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=cfg.global_batch_size, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path=load_path,
        load_trainer_state=False,
        load_optim_state=True,
        max_duration=Duration.tokens(MAX_TOKENS),
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    ).with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts, wandb_group_name=cli_context.run_name
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
    experiment_config = experiment_config.merge(overrides)
    return experiment_config


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
