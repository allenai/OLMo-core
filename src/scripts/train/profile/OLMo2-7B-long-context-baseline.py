"""
Train a 7B OLMo model on long contexts. Configured for profiling.
Run this script without any arguments to see usage info.
"""

import logging

import torch

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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)

# Check and enable TF32 for better performance
log.info(f"torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
torch.backends.cuda.matmul.allow_tf32 = True
log.info("Enabled torch.backends.cuda.matmul.allow_tf32")


# 64K length, 32 GPUs, FP8, no intra-doc masking -> 2,750 TPS/device = 88,000 TPS overall
# 64K length, 32 GPUs, no FP8, intra-doc masking -> 3,250 TPS/device = 104,000 TPS overall
# 64K length, 32 GPUs, FP8, intra-doc masking    -> 3,500 TPS/device = 112,000 TPS overall

CONTEXT_LENGTH = 4 * 16_384
GLOBAL_BATCH_SIZE = 64 * CONTEXT_LENGTH  # cp8, dp4
INTRA_DOCUMENT_MASKING = True

NUM_GPUS = 32
GPU_PER_NODE = 8
assert NUM_GPUS % GPU_PER_NODE == 0
NUM_NODES = NUM_GPUS // GPU_PER_NODE

AC_ATTENTION_INTERVAL = 4
TP_DEGREE = 4
DP_SHARDS = NUM_GPUS // TP_DEGREE
GQA_RATIO = None

log.info(f"TP_DEGREE: {TP_DEGREE}, CP_DEGREE: 0, NUM_GPUS: {NUM_GPUS}, NUM_NODES: {NUM_NODES}")


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        use_flash=True,
        n_kv_heads=int(32 * GQA_RATIO) if GQA_RATIO else None,
        rope_theta=8 * 10**6,
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
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=DP_SHARDS,
        ),
        tp_config=TransformerTensorParallelConfig(degree=TP_DEGREE, enable_async=True),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            activation_memory_budget=0.5,
        ),
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
            max_duration=Duration.steps(22),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-7B-long-context-profile-results",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-7B-long-context-profile-results",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "profiler",
            ProfilerCallback(
                enabled=False,
                skip_first=15,
                wait=3,
                warmup=1,
                active=1,
                repeat=1,
                with_stack=True,
                enable_cuda_sync_events=True,
                ranks="all",
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
