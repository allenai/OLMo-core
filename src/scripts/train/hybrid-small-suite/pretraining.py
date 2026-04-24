"""
OLMo hybrid small suite — 275M, 810M, and 1.4B Cx100 pretraining runs.

Architecture: see arch.py.

Hyperparameters:
  - LR: tuned per model size
  - BSZ: set per model size (derived from Li 2025 step law, rounded to nearest num_gpus)
  - Scheduler: cosine with 2000-step warmup and 2000-step linear decay
  - Duration: 100x Chinchilla (20 * N * 100 tokens per non-embedding param)

Model sizes:
  275M:  275,493,760 total /  211,268,480 non-embedding params, D=422B tokens  (4 nodes)
  810M:  810,354,816 total /  707,594,368 non-embedding params, D=1.4T tokens  (8 nodes)
  1.4B: 1,422,110,720 total / 1,293,660,160 non-embedding params, D=2.6T tokens (32 nodes)

Usage::

  # Dry run — print resolved config without launching:
  uv run src/scripts/train/hybrid-small-suite/pretraining.py dry_run \\
      hybrid-small-275M ai2/jupiter

  # Launch on Beaker (model size is parsed from run name):
  uv run src/scripts/train/hybrid-small-suite/pretraining.py launch \\
      hybrid-small-275M ai2/titan \\
      --launch.num_nodes=4 \\
      --launch.priority=urgent \\
      --launch.budget=ai2/oe-other

  # Train on a single local node (e.g. inside a Beaker session):
  uv run torchrun --nproc-per-node=1 src/scripts/train/hybrid-small-suite/pretraining.py train \\
      hybrid-small-275M ai2/titan

  # Train on a single GPU without torchrun (quick local iteration):
  uv run src/scripts/train/hybrid-small-suite/pretraining.py train_single \\
      hybrid-small-275M ai2/titan

  # Dev / smoke-test — disable checkpointing, W&B, and downstream evals:
  uv run src/scripts/train/hybrid-small-suite/pretraining.py train_single \\
      hybrid-small-275M ai2/titan \\
      --trainer.callbacks.checkpointer=null \\
      --trainer.callbacks.wandb=null \\
      --trainer.max_duration.value=10 \\
      --trainer.max_duration.unit=steps
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from functools import partial

from arch import MODEL_CONFIGS, SEQUENCE_LENGTH, build_model_config, parse_model_size

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.optim import (
    CosWithWarmupAndLinearDecay,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

CHINCHILLA_MULTIPLE = 100
DATA_MIX = DataMix.OLMo_mix_0925

# Per-size learning rates.
LR = {
    "275m": 0.008,
    "810m": 0.002,
    "1.4b": 0.002,
}


def build_train_module_config(
    common: CommonComponents, model_size: str
) -> TransformerTrainModuleConfig:
    cfg = MODEL_CONFIGS[model_size]

    return TransformerTrainModuleConfig(
        rank_microbatch_size=cfg["rank_microbatch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR[model_size],
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmupAndLinearDecay(
            units=SchedulerUnits.steps,
            warmup=2000,
            decay=2000,
            decay_fraction=None,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=1.0,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_data_components(common: CommonComponents) -> DataComponents:
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        mix=DATA_MIX,
        tokenizer=common.tokenizer,
        mix_base_dir="gs://ai2-llm",
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(8192, common.max_sequence_length),
        work_dir=common.work_dir,
        instance_filter_config=InstanceFilterConfig(),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents, model_size: str) -> TrainerConfig:
    cancel_check_interval = 1000

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    model_config = build_model_config(common, model_size)
    train_duration = Duration.chinchilla_tokens(
        CHINCHILLA_MULTIPLE, model_params=model_config.num_non_embedding_params
    )

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            save_folder=common.save_folder,
            work_dir=common.work_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=train_duration,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                project="hybrid-small-suite",
                entity="ai2-llm",
                cancel_check_interval=cancel_check_interval,
                enabled=True,
                tags=["pretraining", model_size],
            ),
        )
        .with_recommended_evals(common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast")
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]\n"
            f"Run name must contain a model size: {list(MODEL_CONFIGS.keys())}"
        )

    model_size = parse_model_size(sys.argv[2])
    cfg = MODEL_CONFIGS[model_size]

    # Map cluster name substrings to the appropriate attention backend.
    CLUSTER_ATTN_BACKENDS = {
        "saturn": AttentionBackendName.flash_2,   # A100s — no flash_3 support
        "titan": AttentionBackendName.flash_4,    # B200, Blackwell
        "jupiter": AttentionBackendName.flash_3,
        "pluto": AttentionBackendName.flash_3,
    }
    cluster_arg = " ".join(sys.argv[2:4]).lower()
    attn_backend = AttentionBackendName.flash_3  # default
    for cluster, backend in CLUSTER_ATTN_BACKENDS.items():
        if cluster in cluster_arg:
            attn_backend = backend
            break
    sys.argv = [a for a in sys.argv if not a.startswith("--attn_backend=")]

    config_builder = partial(
        build_config,
        global_batch_size=cfg["global_batch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=cfg["num_nodes"],
        data_config_builder=build_data_components,
        model_config_builder=partial(build_model_config, model_size=model_size, attn_backend=attn_backend),
        train_module_config_builder=partial(build_train_module_config, model_size=model_size),
        trainer_config_builder=partial(build_trainer_config, model_size=model_size),
        include_default_evals=False,
        beaker_workspace="ai2/linear-rnns",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
