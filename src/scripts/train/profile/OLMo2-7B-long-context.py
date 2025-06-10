"""
Train a 7B OLMo model on long contexts. Configured for profiling.
Run this script without any arguments to see usage info.
"""

import logging

# ruff: noqa: F401
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main
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

# 64K length, 32 GPUs, FP8, no intra-doc masking -> 2,750 TPS
# 64K length, 32 GPUs, no FP8, intra-doc masking -> 3,250 TPS
# 64K length, 32 GPUs, FP8, intra-doc masking    -> 3,500 TPS

# Tyler's Results (64K context length, no FP8, intra-doc masking):
# ---
# 16 GPUs, 32*CL bs -- CP8, DP2, GQA(1/4) -> 5,876 TPS (!! faster w/ fewer data-parallel groups)
# 16 GPUs, 32*CL bs -- TP4, DP4, AC, GQA(1/4) -> 5,822 TPS
# 16 GPUs, 32*CL bs -- CP4, DP4, AC, GQA(1/4) -> 5,443 TPS
# 16 GPUs, 32*CL bs -- TP4, DP4, AC -> 5,412 TPS (suggested by Dustin & Amanda)
# 16 GPUs, 32*CL bs -- CP4, DP4, AC -> 3,977 TPS
# 16 GPUs, 32*CL bs -- CP8, DP2 -> 3,735 TPS (bottlenecked by AllGather_RING)
# 16 GPUs, 32*CL bs -- TP4, DP4 -> OOM
# 16 GPUs, 32*CL bs -- CP4, DP4, GQA(1/4) -> OOM
# ---
# 32 GPUs, 64*CL bs -- CP8, DP4, GQA(1/4) ->
# 32 GPUs, 64*CL bs -- CP4, DP8, AC, GQA(1/4) ->
# 32 GPUs, 64*CL bs -- TP4, DP8, AC, GQA(1/4) ->
# 32 GPUs, 64*CL bs -- TP4, DP8, AC ->


CONTEXT_LENGTH = 4 * 16_384
GLOBAL_BATCH_SIZE = 64 * CONTEXT_LENGTH  # cp8, dp4
# GLOBAL_BATCH_SIZE = 32 * CONTEXT_LENGTH  # cp8, dp2
INTRA_DOCUMENT_MASKING = True

NUM_GPUS = 32
assert NUM_GPUS % 8 == 0
NUM_NODES = NUM_GPUS // 8

AC_ATTENTION_INTERVAL = 4
TP_DEGREE = 4
CP_DEGREE = None
GQA_RATIO = 0.25

log.info(
    f"TP_DEGREE: {TP_DEGREE}, CP_DEGREE: {CP_DEGREE}, NUM_GPUS: {NUM_GPUS}, NUM_NODES: {NUM_NODES}"
)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        use_flash=True,
        n_kv_heads=int(32 * GQA_RATIO) if GQA_RATIO else None,
    )


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
        tp_config=TransformerTensorParallelConfig(degree=TP_DEGREE) if TP_DEGREE else None,
        cp_config=(
            TransformerContextParallelConfig.llama3(degree=CP_DEGREE)
            if INTRA_DOCUMENT_MASKING
            else TransformerContextParallelConfig.zig_zag(degree=CP_DEGREE)
        )
        if CP_DEGREE
        else None,
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=[f"blocks.{i}.feed_forward" for i in range(32)]
            + [f"blocks.{i}.attention" for i in range(0, 32, AC_ATTENTION_INTERVAL)],
        )
        if AC_ATTENTION_INTERVAL
        else None,
        float8_config=Float8Config(enabled=False),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


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
                project="OLMo-core-7B-long-context-profile",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-7B-long-context-profile",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "profiler",
            ProfilerCallback(
                enabled=True,
                skip_first=15,
                wait=3,
                warmup=1,
                active=2,
                repeat=1,
                export_chrome_trace=True,
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(enabled=False),  # no need to checkpoint here
        )
        .with_callback("console_logger", ConsoleLoggerCallback(metrics_log_interval=4))
    )


if __name__ == "__main__":
    main(
        sequence_length=CONTEXT_LENGTH,
        global_batch_size=GLOBAL_BATCH_SIZE,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        intra_document_masking=INTRA_DOCUMENT_MASKING,
        num_nodes=NUM_NODES,
    )
