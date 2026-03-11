from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration

SEQ_LENGTH = 65536
MAX_TOKENS = 10_000_000_000  # 10B
SEED = 1337


def estimate_lr(num_params: int, chinchilla_multiple: float = 4.0) -> float:
    """Estimate LR based on stable in WSD"""
    return 0.0047 * (num_params / 108_000_000) ** (-1 / 3) / chinchilla_multiple**0.5


def estimate_gbs(num_params: int) -> int:
    """Estimate global batch size"""
    return round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))


@dataclass
class LadderSize:
    model_config_fn: Callable[..., TransformerConfig]
    lr: float
    global_batch_size: int
    cp_degree: Optional[int] = None


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
    "370M": LadderSize(
        model_config_fn=TransformerConfig.olmo3_370M,
        lr=estimate_lr(370_000_000),
        global_batch_size=estimate_gbs(370_000_000),
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
        cp_degree=2,
    ),
}


def _pop_overrides(overrides: list[str]) -> tuple[str, Optional[str], int, list[str]]:
    """Extract ``--size``, ``--load-path``, and ``--launch.num_nodes`` from the overrides list."""
    size: Optional[str] = None
    load_path: Optional[str] = None
    num_nodes: Optional[int] = None
    remaining: list[str] = []
    for arg in overrides:
        if arg.startswith("--size="):
            size = arg.split("=", 1)[1]
        elif arg.startswith("--load-path="):
            load_path = arg.split("=", 1)[1]
        elif arg.startswith("--launch.num_nodes="):
            num_nodes = int(arg.split("=", 1)[1])
        else:
            remaining.append(arg)
    if size is None:
        raise ValueError(f"--size is required. Choices: {list(SIZES.keys())}")
    if size not in SIZES:
        raise ValueError(f"Invalid size '{size}'. Choices: {list(SIZES.keys())}")
    return size, load_path, num_nodes or 1, remaining


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    size, load_path, num_nodes, overrides = _pop_overrides(cli_context.overrides)
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
        num_nodes=num_nodes,
        nccl_debug=False,
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = cfg.model_config_fn(
        vocab_size=tokenizer_config.padded_vocab_size(),
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )

    train_module_config = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH,
        learning_rate=cfg.lr,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
        activation_memory_budget=0.5,
        cp_degree=cfg.cp_degree,
    )

    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        "/weka/oe-training-default/ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
        generate_doc_lengths=True,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    # round gbz to num gpus
    num_gpus = beaker_launch_config.num_gpus if beaker_launch_config else 8
    world_size = num_gpus * (beaker_launch_config.num_nodes if beaker_launch_config else 1)
    dp_world_size = world_size // (cfg.cp_degree or 1)
    rank_unit = SEQ_LENGTH * dp_world_size
    gbs = max((cfg.global_batch_size // rank_unit) * rank_unit, rank_unit)

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=gbs, seed=SEED, num_workers=4
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
