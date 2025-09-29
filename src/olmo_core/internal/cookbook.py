from typing import Dict, Optional

import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)


def configure_train_module(
    max_sequence_length: int,
    global_batch_size: int,
    instances_per_rank_microbatch: int,
    scheduler: Scheduler,
    optim: OptimConfig,
    float8_enabled: bool = False,
) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=max_sequence_length * instances_per_rank_microbatch,
        max_sequence_length=max_sequence_length,
        optim=optim,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,  # TODO
        scheduler=scheduler,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=32,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=[f"blocks.{i}.feed_forward" for i in range(0, 64, 4)],
        ),
        float8_config=Float8Config(
            enabled=float8_enabled,
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
        ),
    )


def configure_default_callbacks(
    run_name: str, launch_config: Optional[BeakerLaunchConfig] = None, wandb_project: str = "olmo3"
) -> Dict[str, Callback]:
    callbacks = {
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "slack_notifier": SlackNotifierCallback(name=run_name, enabled=False),
        "monkey_patcher": MonkeyPatcherCallback(),
        "wandb": WandBCallback(
            name=run_name,  # with suffix!
            group=run_name,
            project=wandb_project,
            entity="ai2-llm",
            enabled=True,
            cancel_check_interval=50,
        ),
    }

    if launch_config is not None:
        callbacks["beaker"] = BeakerCallback()
    if torch.cuda.is_available():
        callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()

    return callbacks
