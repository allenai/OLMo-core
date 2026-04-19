"""
OLMo hybrid small suite — 275M, 810M, and 1.4B midtraining runs.

Architecture: see arch.py (identical to pretraining — checkpoints load cleanly).

Scheduler: LinearWithWarmup — linear decay to 0 over MAX_TOKENS.

Before launching:
  - Update ``load_path`` in MIDTRAINING_CONFIGS for each model size.
  - Update SOURCE_MIXTURE_YAML to the desired midtraining data mix.
  - Verify MAX_TOKENS per model size.

Usage:
  # Dry run (print config without launching):
  python src/scripts/train/hybrid-small-suite/midtraining.py dry_run \\
      hybrid-small-midtraining-275M ai2/jupiter

  # Launch on Beaker (model size is parsed from run name):
  python src/scripts/train/hybrid-small-suite/midtraining.py launch \\
      hybrid-small-midtraining-1.4B ai2/jupiter
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from functools import partial

from arch import MODEL_CONFIGS, SEQUENCE_LENGTH, build_model_config, parse_model_size

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

MAX_TOKENS = 100_000_000_000  # 100B
SEED = 1337
INSTANCE_FILTER = True

# Update to the desired midtraining data mix before launching.
SOURCE_MIXTURE_YAML = (
    "src/olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
)

MIDTRAINING_CONFIGS = {
    "275m": dict(
        # Starting LR: ~10% of peak pretraining LR (0.008).
        lr=8e-4,
        global_batch_size=2_621_440,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-275M-Cx100/step161186/",
    ),
    "810m": dict(
        # Starting LR: ~10% of peak pretraining LR (0.002).
        lr=2e-4,
        global_batch_size=5_242_880,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-810M-Cx100/step269926/",
    ),
    "1.4b": dict(
        # Starting LR: ~10% of peak pretraining LR (0.002).
        lr=2e-4,
        global_batch_size=8_388_608,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-1.4B-Cx100/step308433/",
    ),
}


def build_train_module_config(
    common: CommonComponents, model_size: str
) -> TransformerTrainModuleConfig:
    cfg = MODEL_CONFIGS[model_size]
    mt_cfg = MIDTRAINING_CONFIGS[model_size]

    return TransformerTrainModuleConfig(
        rank_microbatch_size=cfg["rank_microbatch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=mt_cfg["lr"],
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        # Linear decay from starting LR to 0 over the full midtraining run.
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=100, alpha_f=0.0),
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


def build_data_components(
    common: CommonComponents,
    model_size: str,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
    global_batch_size: int = 0,
) -> DataComponents:
    mt_cfg = MIDTRAINING_CONFIGS[model_size]
    if global_batch_size <= 0:
        global_batch_size = mt_cfg["global_batch_size"]

    source_list = SourceMixtureList.from_yaml(SOURCE_MIXTURE_YAML)
    source_list.validate()

    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=MAX_TOKENS,
            global_batch_size=global_batch_size,
            processes=16,
            seed=SEED,
        ),
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=SEED, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents, model_size: str) -> TrainerConfig:
    cancel_check_interval = 10
    mt_cfg = MIDTRAINING_CONFIGS[model_size]

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            load_path=mt_cfg["load_path"],
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
                tags=["midtraining", model_size],
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
    mt_cfg = MIDTRAINING_CONFIGS[model_size]

    attn_backend_str = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--attn_backend=")), None)
    attn_backend = AttentionBackendName[attn_backend_str] if attn_backend_str else AttentionBackendName.flash_3
    # Only strip when running locally as 'train' — for 'launch', keep it in sys.argv
    # so Beaker embeds it in the remote command.
    if len(sys.argv) > 1 and sys.argv[1] in ("train", "train_single", "dry_run"):
        sys.argv = [a for a in sys.argv if not a.startswith("--attn_backend=")]

    config_builder = partial(
        build_config,
        global_batch_size=mt_cfg["global_batch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=cfg["num_nodes"],
        data_config_builder=partial(build_data_components, model_size=model_size),
        model_config_builder=partial(build_model_config, model_size=model_size, attn_backend=attn_backend),
        train_module_config_builder=partial(build_train_module_config, model_size=model_size),
        trainer_config_builder=partial(build_trainer_config, model_size=model_size),
        include_default_evals=False,
        include_instance_filter=INSTANCE_FILTER,
        beaker_workspace="ai2/linear-rnns",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
