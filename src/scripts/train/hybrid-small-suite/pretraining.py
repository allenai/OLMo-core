"""Reproduce W&B run ``ai2-llm/hybrid-small-suite/gpii1bqu`` for 3K steps.

The source configuration came from ``YashasSamaga/OLMo-core`` at
``f9893fe62ab11aef89699df0307171a053616c30``. This version targets Robert's
Beaker workspace and Weka prefix and uses the FLA source commit pinned by this
OLMo-core checkout.

Dry run::

    python src/scripts/train/hybrid-small-suite/pretraining.py dry_run \
        hybrid-small-2.7B-Cx1-lr4e-4-fla-cbb0a72-3k ai2/holmes

Launch::

    python src/scripts/train/hybrid-small-suite/pretraining.py launch \
        hybrid-small-2.7B-Cx1-lr4e-4-fla-cbb0a72-3k ai2/holmes \
        --launch.priority=urgent \
        --launch.follow=false \
        --launch.step_soft_timeout=null
"""

import os
import sys
from datetime import datetime
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch import MODEL_CONFIG, SEQUENCE_LENGTH, build_model_config

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
    CliContext,
    CommonComponents,
    DataComponents,
    build_common_components,
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

SOURCE_RUN = "ai2-llm/hybrid-small-suite/gpii1bqu"
FLA_COMMIT = "cbb0a72efb55c18ca0ef4f298298317573ad2cb3"

BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-2026-07-03"
BEAKER_WORKSPACE = "ai2/OLMo-3-moe-experiments"
WANDB_ENTITY = "ai2-llm"
WANDB_PROJECT = "hybrid-small-suite"
WEKA_ROOT = "/weka/oe-training-default/robertb"

GLOBAL_BATCH_SIZE = 4 * 1024 * 1024
ORIGINAL_MAX_TOKENS = 50_687_385_600
ORIGINAL_MAX_STEPS = 12_085
HARD_STOP_STEPS = 3_000
PEAK_LR = 4e-4


def build_repro_common_components(cli_context: CliContext, **kwargs) -> CommonComponents:
    common = build_common_components(cli_context, **kwargs)
    common.root_dir = WEKA_ROOT
    common.work_dir = f"{WEKA_ROOT}/dataset-cache"
    common.save_folder = f"{WEKA_ROOT}/checkpoints/{common.run_name}"
    return common


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=MODEL_CONFIG["rank_microbatch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=PEAK_LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts=dict(weight_decay=0.0),
                )
            ],
        ),
        scheduler=CosWithWarmupAndLinearDecay(
            units=SchedulerUnits.steps,
            warmup=2_000,
            decay=2_000,
            decay_fraction=None,
            # OLMo-core changed this scheduler after the source run. Pinning the
            # original 12,085-step horizon preserves its LR curve through 3K.
            t_max=ORIGINAL_MAX_STEPS,
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
        mix=DataMix.OLMo_mix_0925,
        tokenizer=common.tokenizer,
        mix_base_dir="gs://ai2-llm",
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(8192, common.max_sequence_length),
        work_dir=common.work_dir,
        instance_filter_config=InstanceFilterConfig(),
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=8,
    )
    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 1_000
    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]
    timestamped_name = (
        f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )

    return (
        TrainerConfig(
            save_folder=common.save_folder,
            work_dir=common.work_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(ORIGINAL_MAX_TOKENS),
            hard_stop=Duration.steps(HARD_STOP_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1_000,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=timestamped_name,
                group=common.run_name,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                cancel_check_interval=cancel_check_interval,
                enabled=True,
                tags=["pretraining", "2.7b", "fla-source", "3k-repro"],
            ),
        )
        .with_recommended_evals(
            common.tokenizer,
            SEQUENCE_LENGTH,
            cluster,
            task_set="fast",
        )
    )


def main_config(cli_context: CliContext):
    cluster = cli_context.cluster.lower()
    if "saturn" in cluster:
        attn_backend = AttentionBackendName.flash_2
    elif "titan" in cluster or "holmes" in cluster:
        attn_backend = AttentionBackendName.flash_4
    else:
        attn_backend = AttentionBackendName.flash_3

    return build_config(
        cli_context,
        common_config_builder=build_repro_common_components,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=MODEL_CONFIG["num_nodes"],
        data_config_builder=build_data_components,
        model_config_builder=partial(build_model_config, attn_backend=attn_backend),
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        beaker_image=BEAKER_IMAGE,
        beaker_workspace=BEAKER_WORKSPACE,
        num_execution_units=1,
    )


if __name__ == "__main__":
    main(config_builder=main_config)
