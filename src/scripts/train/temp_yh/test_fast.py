"""
Train a 1B OLMo model with Repeated WSD scheduler for ladder/scaling laws runs.
"""

from datetime import datetime

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE, get_root_dir
from functools import partial
from olmo_core.internal.experiment import build_config, CommonComponents, ExperimentConfig, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig
# Import RepeatedWSD instead of WSD
from olmo_core.optim.scheduler import RepeatedWSD  # or CyclicWSD
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BatchSizeSchedulerCallback,
    CheckpointerCallback,
    CometCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

NUM_GPUS = 4
NUM_NODES = 1

SEQUENCE_LENGTH = 2048
N_LAYERS = 2

# Batch size configuration
GLOBAL_BATCH_SIZE = 32 * 2048  # 65,536 tokens per step

# Duration configuration in STEPS
MAX_DURATION_STEPS = 200  # Adjust as needed for your experiment

# Learning rate configuration
LR = 1e-4

# Scheduler configuration in STEPS
CYCLE_STEPS = 20  # Each cycle is 20 steps (based on your graph)
WARMUP_STEPS = 2  # 10% of cycle for warmup
DECAY_STEPS = 4   # 20% of cycle for decay
# This leaves 14 steps (70%) for stable phase

# Save and eval intervals in STEPS
SAVE_INTERVAL_STEPS = CYCLE_STEPS  # Save at the end of each cycle
EVAL_INTERVAL_STEPS = CYCLE_STEPS // 2  # Eval twice per cycle

PREEMPTIBLE = True


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_1B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=4,  # Reduced from 16
        hidden_size_multiple_of=256,  # Smaller for faster training
    )
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[1025, 1025, 1025, -1],  # Adjusted for shorter sequence
    )
    config.block.attention.use_flash = True
    config.block.attention.use_head_qk_norm = True
    return config

def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = 2 * SEQUENCE_LENGTH  # Reduced from 4
    
    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=False,
            step_increment_bugfix=False,
        ),
        compile_model=False,  # Disable for faster startup in testing
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=RepeatedWSD(
            units=SchedulerUnits.steps,  # Changed to steps!
            cycle_length=CYCLE_STEPS,     # 20 steps per cycle
            warmup=WARMUP_STEPS,          # 2 steps warmup
            decay=DECAY_STEPS,            # 4 steps decay
            decay_fraction=None,          # Using absolute steps, not fractions
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
            restart_warmup_from_min=True,
        ),
    )

def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 50

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    root_dir = "/weka/oe-training-default/ai2-llm"
    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/yanhongl/linear-rnns/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.steps(MAX_DURATION_STEPS),  # Changed to steps
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL_STEPS,  # Save every cycle (20 steps)
                # ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="yanhong-lbh",
                project="linear-rnns",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_recommended_evals(
            common.tokenizer, 
            SEQUENCE_LENGTH, 
            cluster, 
            task_set="fast", 
            eval_interval=EVAL_INTERVAL_STEPS  # Changed to steps
        )
    )

    # Optional: Add logging callback to monitor cycle info
    # You might need to implement this callback based on your framework
    # config.callbacks["cycle_logger"] = CycleLoggingCallback(
    #     log_interval=1,  # Log every step to see cycle phases
    # )

    return config


def set_preemptible(config: ExperimentConfig) -> None:
    if config.launch is not None:
        config.launch.preemptible = PREEMPTIBLE
        
        # Set GPU configuration
        if hasattr(config.launch, 'num_gpus'):
            config.launch.num_gpus = NUM_GPUS
        if hasattr(config.launch, 'num_nodes'):
            config.launch.num_nodes = NUM_NODES
            
        # Alternative: Set via cluster configuration
        # This depends on your specific cluster setup
        for cluster_config in config.launch.clusters:
            if hasattr(cluster_config, 'num_gpus'):
                cluster_config.num_gpus = NUM_GPUS
            if hasattr(cluster_config, 'num_nodes'):
                cluster_config.num_nodes = NUM_NODES


if __name__ == "__main__":
    from functools import partial
    from olmo_core.internal.experiment import build_config
    
    # Note: You might want to adjust these based on your actual needs
    # Since we're using steps, the global_batch_size still defines tokens per step
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,  # Still in tokens (65,536 tokens per step)
        max_sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=set_preemptible,
        include_instance_filter=False,
        include_default_evals=False,
        intra_document_masking=True,
        beaker_workspace="ai2/linear-rnns",
    )
    
    main(config_builder=config_builder)