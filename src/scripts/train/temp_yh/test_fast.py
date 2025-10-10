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

NUM_GPUS = 4  # Set your desired number of GPUs here
NUM_NODES = 1  # Number of nodes

# Model size - reduce layers dramatically
SEQUENCE_LENGTH = 2048  # Reduce from 8192
N_LAYERS = 2  # Reduce from 16 in build_model_config

# Batch and token settings for fast cycles
GLOBAL_BATCH_SIZE = 32 * 2048  # ~65K tokens per step (much smaller)

# Total duration - keep it short for testing
MAX_DURATION = int(10e6)  # 10M tokens total = ~10 cycles

ANNEAL_TOKENS = int(1e9)
# Learning rate - can keep the same or increase for visibility
LR = 1e-4

# Make cycles SHORT so you can see them quickly
CYCLE_TOKENS = int(1e6)  # 1M tokens per cycle (down from 10B!)
WARMUP_TOKENS = int(50e3)  # 50K tokens warmup
DECAY_TOKENS = int(100e3)  # 100K tokens decay
# Save and eval - align with cycles
SAVE_INTERVAL_TOKENS = CYCLE_TOKENS  # Save every cycle
EVAL_INTERVAL_TOKENS = CYCLE_TOKENS // 2  # Eval twice per cycle

# Use this to change whether the job is preemptible or not.
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
            units=SchedulerUnits.tokens,
            cycle_length=CYCLE_TOKENS,  # 1M tokens
            warmup=WARMUP_TOKENS,  # 50K tokens
            decay=DECAY_TOKENS,  # 100K 
            decay_fraction=None,
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
            max_duration=Duration.tokens(MAX_DURATION),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL_TOKENS,  # Changed from SAVE_INTERVAL
                ephemeral_save_interval=100,
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
            common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL_TOKENS  # Changed
        )
    )

    # # batch size warmup
    # config.callbacks["batchwup"] = BatchSizeSchedulerCallback(
    #     batch_sizes=[GLOBAL_BATCH_SIZE, GLOBAL_BATCH_SIZE * 2, GLOBAL_BATCH_SIZE * 4],
    #     schedule=[
    #         Duration.tokens(0),
    #         Duration.tokens(167_772_160_000),
    #         Duration.tokens(503_316_480_000),
    #     ],
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
    
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
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