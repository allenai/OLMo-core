"""
Sweep script to launch a set of long-context 7B OLMo experiments.

This submits Beaker jobs for each configuration listed in `CONFIGS`.

Usage
-----
python OLMo2-7B-long-context-sweep.py CLUSTER_NAME

where ``CLUSTER_NAME`` is the Beaker cluster to run on, e.g. ``ai2/pluto-cirrascale``.
"""

import logging
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    SubCmd,
    build_config,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import (
    CometCallback,
    GPUMemoryMonitorCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.callbacks.console_logger import ConsoleLoggerCallback
from olmo_core.train.common import Duration
from olmo_core.train.train_module import (
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------------------------------------------------------
# Constants shared across all experiments.
# ----------------------------------------------------------------------------
CONTEXT_LENGTH = 4 * 16_384  # 65,536
INTRA_DOCUMENT_MASKING = True
AC_INTERVAL = 4  # Only relevant when AC is enabled.


# ----------------------------------------------------------------------------
# Experiment matrix.
# Each item defines a unique experiment – list adapted from comments in
# ``OLMo2-7B-long-context.py``.
# ----------------------------------------------------------------------------
CONFIGS: List[Dict] = [
    # 16 GPUs, GLOBAL_BATCH_SIZE = 32 * CONTEXT_LENGTH
    {
        "name": "tp4_dp4_ac_gqa",
        "num_gpus": 16,
        "global_bs_factor": 32,
        "tp": 4,
        "cp": None,
        "ac": True,
        "gqa_ratio": 1 / 4,
    },
    {
        "name": "cp4_dp4_ac_gqa",
        "num_gpus": 16,
        "global_bs_factor": 32,
        "tp": None,
        "cp": 4,
        "ac": True,
        "gqa_ratio": 1 / 4,
    },
    {
        "name": "cp2_tp2_dp4_ac_gqa",
        "num_gpus": 16,
        "global_bs_factor": 32,
        "tp": 2,
        "cp": 2,
        "ac": True,
        "gqa_ratio": 1 / 4,
    },
    {
        "name": "tp4_dp4_ac",
        "num_gpus": 16,
        "global_bs_factor": 32,
        "tp": 4,
        "cp": None,
        "ac": True,
        "gqa_ratio": None,
    },
    {
        "name": "cp4_dp4_ac",
        "num_gpus": 16,
        "global_bs_factor": 32,
        "tp": None,
        "cp": 4,
        "ac": True,
        "gqa_ratio": None,
    },
    {
        "name": "cp8_dp2_gqa",
        "num_gpus": 16,
        "global_bs_factor": 32,
        "tp": None,
        "cp": 8,
        "ac": False,
        "gqa_ratio": 1 / 4,
    },
    {
        "name": "cp8_dp2",
        "num_gpus": 16,
        "global_bs_factor": 32,
        "tp": None,
        "cp": 8,
        "ac": False,
        "gqa_ratio": None,
    },
    # # 32 GPUs, GLOBAL_BATCH_SIZE = 64 * CONTEXT_LENGTH
    # {
    #     "name": "tp4_dp8_ac_gqa",
    #     "num_gpus": 32,
    #     "global_bs_factor": 64,
    #     "tp": 4,
    #     "cp": None,
    #     "ac": True,
    #     "gqa_ratio": 1 / 4,
    # },
    # {
    #     "name": "tp4_dp8_ac",
    #     "num_gpus": 32,
    #     "global_bs_factor": 64,
    #     "tp": 4,
    #     "cp": None,
    #     "ac": True,
    #     "gqa_ratio": None,
    # },
    # {
    #     "name": "cp2_tp2_dp8_ac_gqa",
    #     "num_gpus": 32,
    #     "global_bs_factor": 64,
    #     "tp": 2,
    #     "cp": 2,
    #     "ac": True,
    #     "gqa_ratio": 1 / 4,
    # },
    # {
    #     "name": "cp4_dp8_ac_gqa",
    #     "num_gpus": 32,
    #     "global_bs_factor": 64,
    #     "tp": None,
    #     "cp": 4,
    #     "ac": True,
    #     "gqa_ratio": 1 / 4,
    # },
    # {
    #     "name": "cp8_dp4_gqa",
    #     "num_gpus": 32,
    #     "global_bs_factor": 64,
    #     "tp": None,
    #     "cp": 8,
    #     "ac": False,  # no AC specified in notes
    #     "gqa_ratio": 1 / 4,
    # },
]


# ----------------------------------------------------------------------------
# Helper builders that close over per-experiment hyper-parameters.
# ----------------------------------------------------------------------------
def make_build_model_config(
    gqa_ratio: Optional[float],
) -> Callable[[CommonComponents], TransformerConfig]:
    def build_model_config(common: CommonComponents) -> TransformerConfig:
        return TransformerConfig.olmo2_7B(
            vocab_size=common.tokenizer.padded_vocab_size(),
            use_flash=True,
            n_kv_heads=int(32 * gqa_ratio) if gqa_ratio else None,
        )

    return build_model_config


def make_build_train_module_config(
    tp_degree: Optional[int],
    cp_degree: Optional[int],
    ac: bool,
) -> Callable[[CommonComponents], TransformerTrainModuleConfig]:
    def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
        return TransformerTrainModuleConfig(
            rank_microbatch_size=1 * CONTEXT_LENGTH,
            max_sequence_length=common.dataset.effective_sequence_length,
            optim=AdamWConfig(
                lr=1e-5,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                ],
                fused=True,
            ),
            compile_model=True,
            z_loss_multiplier=1e-5,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.fine_grained,
            ),
            tp_config=TransformerTensorParallelConfig(degree=tp_degree) if tp_degree else None,
            cp_config=(
                TransformerContextParallelConfig.llama3(degree=cp_degree)
                if INTRA_DOCUMENT_MASKING
                else TransformerContextParallelConfig.zig_zag(degree=cp_degree)
            )
            if cp_degree
            else None,
            ac_config=(
                TransformerActivationCheckpointingConfig(
                    mode=TransformerActivationCheckpointingMode.selected_modules,
                    modules=[f"blocks.{i}.feed_forward" for i in range(32)]
                    + [f"blocks.{i}.attention" for i in range(0, 32, AC_INTERVAL)],
                )
                if ac
                else None
            ),
            float8_config=Float8Config(enabled=False),
            max_grad_norm=1.0,
            scheduler=CosWithWarmup(warmup_steps=2000),
        )

    return build_train_module_config


def make_build_trainer_config() -> Callable[[CommonComponents], TrainerConfig]:
    """Return a trainer config builder with profiling **disabled** and WandB **enabled**."""

    def build_trainer_config(common: CommonComponents) -> TrainerConfig:
        return (
            TrainerConfig(
                save_folder=common.save_folder,
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=1,
                max_duration=Duration.steps(30),
            )
            .with_callback(
                "comet",
                CometCallback(
                    name=common.run_name,
                    workspace="ai2",
                    project="OLMo-core-7B-long-context",
                    enabled=False,
                    cancel_check_interval=10,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=common.run_name,
                    entity="ai2-llm",
                    project="OLMo-core-7B-long-context",
                    enabled=True,
                    cancel_check_interval=10,
                ),
            )
            .with_callback(
                "profiler",
                ProfilerCallback(enabled=False),  # DISABLE PROFILING
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(enabled=False),
            )
            .with_callback("console_logger", ConsoleLoggerCallback(metrics_log_interval=4))
        )

    return build_trainer_config


# ----------------------------------------------------------------------------
# Main launcher.
# ----------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: python OLMo2-7B-long-context-sweep.py CLUSTER_NAME")
        sys.exit(1)

    cluster = sys.argv[1]
    date_str = datetime.now().strftime("%y%m%d")

    for cfg in CONFIGS:
        # Construct a unique run name.
        run_name = f"{cfg['name']}-{date_str}"

        # Builder functions specialised for this configuration.
        model_cfg_builder = make_build_model_config(cfg["gqa_ratio"])
        train_module_cfg_builder = make_build_train_module_config(
            tp_degree=cfg["tp"],
            cp_degree=cfg["cp"],
            ac=cfg["ac"],
        )
        trainer_cfg_builder = make_build_trainer_config()

        global_batch_size = cfg["global_bs_factor"] * CONTEXT_LENGTH
        num_nodes = cfg["num_gpus"] // 8

        log.info(
            "Launching %s | GPUs=%s, nodes=%s, tp=%s, cp=%s, ac=%s, gqa=%s, GBS=%s",
            run_name,
            cfg["num_gpus"],
            num_nodes,
            cfg["tp"],
            cfg["cp"],
            cfg["ac"],
            cfg["gqa_ratio"],
            global_batch_size,
        )

        # Build experiment config.
        exp_config = build_config(
            script=__file__,
            cmd=SubCmd.launch,
            run_name=run_name,
            cluster=cluster,
            overrides=[],
            global_batch_size=global_batch_size,
            model_config_builder=model_cfg_builder,
            train_module_config_builder=train_module_cfg_builder,
            trainer_config_builder=trainer_cfg_builder,
            sequence_length=CONTEXT_LENGTH,
            include_default_evals=False,
            intra_document_masking=INTRA_DOCUMENT_MASKING,
            num_nodes=num_nodes,
        )

        assert exp_config.launch is not None, "Beaker launch config missing!"
        exp_config.launch.num_gpus = 8  # every node gets 8 GPUs (H100).
        exp_config.launch.num_nodes = num_nodes

        # Submit to Beaker without following logs.
        exp_config.launch.launch(follow=False)


if __name__ == "__main__":
    main()
