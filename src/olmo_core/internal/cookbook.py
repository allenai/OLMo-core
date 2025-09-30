import logging
import math
from datetime import datetime
from typing import Dict, Optional

import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.common import get_beaker_username, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train import Duration, DurationUnit, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    MonkeyPatcherCallback,
    ProfilerCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)


def setup_basics(cli_context: CliContext):
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{run_name_with_ts}"
    return run_name_with_ts, root_dir, work_dir, save_dir


def configure_train_module(
    max_sequence_length: int,
    rank_microbatch_size: int,
    learning_rate: float,
    scheduler: Scheduler,
    float8_enabled: bool = False,
    activation_checkpointing_enabled: bool = True,
    cp_degree: Optional[int] = None,
) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=scheduler,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=32,
        ),
        cp_config=(
            TransformerContextParallelConfig.llama3(degree=cp_degree, head_stride=4)
            if cp_degree
            else None
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=[f"blocks.{i}.feed_forward" for i in range(0, 64, 4)],
        )
        if activation_checkpointing_enabled
        else None,
        float8_config=Float8Config(
            enabled=float8_enabled,
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
        ),
    )


def configure_trainer(
    max_duration: Duration,
    checkpoint_dir: str,
    work_dir: str,
    load_path: Optional[str] = None,
    hard_stop: Optional[Duration] = None,
) -> TrainerConfig:
    # TODO: ensure load_path exists (even on remote) before proceeding
    load_strategy = LoadStrategy.always if load_path else LoadStrategy.if_available
    trainer_config = TrainerConfig(
        max_duration=max_duration,
        load_path=load_path,
        save_folder=checkpoint_dir,
        work_dir=work_dir,
        hard_stop=hard_stop,
        load_strategy=load_strategy,
        save_overwrite=True,
    )
    return trainer_config


def configure_default_callbacks(
    run_name: str,
    wandb_group_name: str,
    wandb_project: str = "olmo-cookbook",
    checkpoint_save_interval: int = 1000,
) -> Dict[str, Callback]:
    callbacks = {
        "checkpointer": CheckpointerCallback(
            save_interval=checkpoint_save_interval,
            ephemeral_save_interval=100,
            save_async=False,  # TODO: enable async saving when augusta stops being silly
        ),
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "slack_notifier": SlackNotifierCallback(name=run_name, enabled=False),
        "monkey_patcher": MonkeyPatcherCallback(),
        "wandb": WandBCallback(
            name=run_name,
            group=wandb_group_name,
            project=wandb_project,
            entity="ai2-llm",
            cancel_check_interval=20,
            enabled=True,
        ),
    }

    beaker_user = get_beaker_username()
    if beaker_user is not None:
        callbacks["beaker"] = BeakerCallback()
    if torch.cuda.is_available():
        callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()

    return callbacks


def estimate_critical_batch_size(
    sequence_length: int, duration: Duration, _factor: float = 8
) -> int:
    """
    Estimate the instant critical batch size as `bs = factor * sqrt(total_steps)`.

    Ported from https://github.com/allenai/olmo-cookbook/blob/ac657a8cc905b5363f3f97c21393ee82f1df5bc8/src/cookbook/cli/core.py#L46
    """
    if duration.unit == DurationUnit.steps:
        critical_batch_size = _factor * (duration.value ** (1 / 2))
    elif duration.unit == DurationUnit.tokens:
        # why is this different from the previous formula and the docstring??
        critical_batch_size = ((_factor**2) * (duration.value / sequence_length)) ** (1 / 3)
    else:
        raise ValueError(
            f"Duration unit {duration.unit} not supported for critical batch size estimation."
        )

    safe_batch_size = int(2 ** math.floor(math.log2(critical_batch_size)))
    max_batch_size = 2**24 // sequence_length  # 16M tokens from llama 3 405B
    result = min(safe_batch_size, max_batch_size)

    if result < safe_batch_size:
        log.warning(f"Critical batch size {safe_batch_size} capped to {result}")

    return result


def extract_hparams_from_checkpoint(checkpoint_path: str) -> Dict[str, str]:
    """
    Extract hyperparameters from a checkpoint path.
    """
    raise NotImplementedError("TODO")
