"""
OLMo hybrid small suite — 275M, 810M, and 1.4B long-context extension runs.

Extends each midtraining checkpoint to 65k sequence length.  The architecture
in arch.py already uses NoPE (rope=None) on attention layers, so no explicit
DroPE step is needed.  Fused-linear loss is enabled to reduce memory at long
sequence lengths, and context parallelism (Ulysses degree=2) is used to shard
the long sequences across GPU pairs.

Before launching:
  - Update ``load_path`` in LONG_CONTEXT_CONFIGS for each model size with the
    final midtraining checkpoint path.

Usage:
  # Dry run (print config without launching):
  python src/scripts/train/hybrid-small-suite/long_context.py dry_run \\
      hybrid-small-lc-275M ai2/jupiter

  # Launch on Beaker (model size is parsed from run name):
  python src/scripts/train/hybrid-small-suite/long_context.py launch \\
      hybrid-small-lc-1.4B ai2/jupiter
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from functools import partial

from arch import MODEL_CONFIGS, build_model_config as arch_build_model_config
from arch import parse_model_size

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
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
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import (
    LinearWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

LC_SEQUENCE_LENGTH = 65536
MAX_TOKENS = 100_000_000_000  # 100B

# Long-context data (same source as the 7B long-context run).
LC_DATA_GLOB = "gs://ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy"

# Per-size long-context settings.
# load_path must be updated to the final midtraining checkpoint path before launching.
# num_nodes doubles the midtraining node count to handle 8x longer sequences.
LONG_CONTEXT_CONFIGS = {
    "275m": dict(
        lr=2e-4,
        num_nodes=2,
        global_batch_size=2 * 1024 * 1024,  # ~1M tokens (same as 7B LC)
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-275M-lr2.83e-4/step38147/",
    ),
    "810m": dict(
        lr=2e-4,
        num_nodes=8,
        global_batch_size=4 * 1024 * 1024,  # same as pretraining
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-810M-lr2e4/step19074",
    ),
    "1.4b": dict(
        lr=2e-4,
        num_nodes=16,
        global_batch_size=1 * 1024 * 1024,
        load_path="",  # TODO: set to midtraining checkpoint
    ),
}


def build_model_config(
    common: CommonComponents, model_size: str, attn_backend: AttentionBackendName = AttentionBackendName.flash_3
) -> TransformerConfig:
    model_config = arch_build_model_config(common, model_size, attn_backend=attn_backend)
    # Enable fused-linear loss to reduce memory pressure at long sequence lengths.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    return model_config


def build_train_module_config(
    common: CommonComponents, model_size: str
) -> TransformerTrainModuleConfig:
    lc_cfg = LONG_CONTEXT_CONFIGS[model_size]

    return TransformerTrainModuleConfig(
        # One sequence per rank — memory budget is tight at 65k tokens.
        rank_microbatch_size=LC_SEQUENCE_LENGTH,
        max_sequence_length=LC_SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=lc_cfg["lr"],
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        # Linear decay from starting LR to 0 over the full LC run.
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        # Shard each long sequence across 2 GPUs via Ulysses context parallelism.
        # (Same setting as the 7B LC run.) Disabled for 275M.
        cp_config=None if model_size in ("275m", "810m") else TransformerContextParallelConfig.ulysses(degree=2),
        # Low activation budget — long sequences have large activation memory.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_data_components(
    common: CommonComponents,
    model_size: str,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    lc_cfg = LONG_CONTEXT_CONFIGS[model_size]

    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        LC_DATA_GLOB,
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=LC_SEQUENCE_LENGTH,
        generate_doc_lengths=intra_document_masking,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=lc_cfg["global_batch_size"],
        seed=34521,
        num_workers=16,
        prefetch_factor=8,
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents, model_size: str) -> TrainerConfig:
    cancel_check_interval = 10
    lc_cfg = LONG_CONTEXT_CONFIGS[model_size]

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    load_path = lc_cfg["load_path"]
    if not load_path:
        raise ValueError(
            f"load_path for size '{model_size}' is not set in LONG_CONTEXT_CONFIGS. "
            "Update it to the final midtraining checkpoint path before launching."
        )

    trainer_cfg = (
        TrainerConfig(
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            load_path=load_path,
            save_folder=common.save_folder,
            work_dir=common.work_dir,
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
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
                tags=["long-context", model_size],
            ),
        )
    )

    # Downstream evals require full logits which are unavailable with CP or TP.
    if model_size in ("275m", "810m"):
        trainer_cfg = trainer_cfg.with_recommended_evals(
            common.tokenizer, LC_SEQUENCE_LENGTH, cluster, task_set="fast"
        )

    return trainer_cfg


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]\n"
            f"Run name must contain a model size: {list(MODEL_CONFIGS.keys())}"
        )

    model_size = parse_model_size(sys.argv[2])
    lc_cfg = LONG_CONTEXT_CONFIGS[model_size]

    attn_backend_str = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--attn_backend=")), None)
    attn_backend = AttentionBackendName[attn_backend_str] if attn_backend_str else AttentionBackendName.flash_3
    # Only strip when running locally as 'train' — for 'launch', keep it in sys.argv
    # so Beaker embeds it in the remote command.
    if len(sys.argv) > 1 and sys.argv[1] in ("train", "train_single", "dry_run"):
        sys.argv = [a for a in sys.argv if not a.startswith("--attn_backend=")]

    config_builder = partial(
        build_config,
        global_batch_size=lc_cfg["global_batch_size"],
        max_sequence_length=LC_SEQUENCE_LENGTH,
        num_nodes=lc_cfg["num_nodes"],
        data_config_builder=partial(build_data_components, model_size=model_size),
        model_config_builder=partial(build_model_config, model_size=model_size, attn_backend=attn_backend),
        train_module_config_builder=partial(build_train_module_config, model_size=model_size),
        trainer_config_builder=partial(build_trainer_config, model_size=model_size),
        include_default_evals=False,
        include_instance_filter=True,
        beaker_workspace="ai2/linear-rnns",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
