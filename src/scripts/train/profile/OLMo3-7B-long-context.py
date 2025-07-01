"""
Train a 7B OLMo3 model on long contexts. Configured for profiling.
Run this script without any arguments to see usage info.
"""

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.console_logger import ConsoleLoggerCallback
from olmo_core.train.callbacks.gpu_memory_monitor import GPUMemoryMonitorCallback
from olmo_core.train.callbacks.profiler import ProfilerCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerContextParallelConfig,
    TransformerTensorParallelConfig,
)

CONTEXT_LENGTH = 4 * 16_384
GLOBAL_BATCH_SIZE = 64 * CONTEXT_LENGTH
INTRA_DOCUMENT_MASKING = True

NUM_GPUS = 32
assert NUM_GPUS % 8 == 0
NUM_NODES = NUM_GPUS // 8

# Node(TP = 4, CP = 2) x 2 DP shards x 2 DP replics
TP_DEGREE = None
CP_DEGREE = 4
DP_SHARDS = 2
# DP_REPLICAS = NUM_GPUS // (TP_DEGREE * CP_DEGREE * DP_SHARDS)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_kv_heads=8,
        hidden_size_multiplier=1.2,
        hidden_size_multiple_of=1024,
    )

    # NOTE: TP + CP + Sliding Window not yet supported
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[4096, 4096, 4096, -1]    
    )
    config.block.attention.use_flash = True
    config.block.attention.use_head_qk_norm = True

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=1 * CONTEXT_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=1e-5,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=False,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=DP_SHARDS,
        ),
        tp_config=TransformerTensorParallelConfig(degree=TP_DEGREE, enable_async=True)
        if TP_DEGREE
        else None,
        cp_config=(
            TransformerContextParallelConfig.llama3(degree=CP_DEGREE)
            if INTRA_DOCUMENT_MASKING
            else TransformerContextParallelConfig.zig_zag(degree=CP_DEGREE)
        ),
        float8_config=Float8Config(
            enabled=True,
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
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
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=CONTEXT_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        intra_document_masking=INTRA_DOCUMENT_MASKING,
        num_nodes=NUM_NODES,
    )
