from datetime import datetime
from functools import partial
from typing import List

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import build_config, CommonComponents, ExperimentConfig, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig
from olmo_core.optim.scheduler import WSDS  # <-- NEW
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BatchSizeSchedulerCallback,
    CometCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
# NEW callbacks
from olmo_core.train.callbacks.zero_lr_checkpointer import ZeroLRCheckpointerCallback
from olmo_core.train.callbacks.wandb import WandBCallback


NUM_GPUS = 4
NUM_NODES = 1

SEQUENCE_LENGTH = 2048

# GLOBAL batch size in *tokens per optimizer step*
# (this is what converts token budgets -> steps)
GLOBAL_BATCH_SIZE_TOKENS = 32 * 2048  # 65,536 tokens/step

# ---------------------------------------------------------
# WSD‑S configuration
# ---------------------------------------------------------
# Period budgets in TOKENS (example: 50B, 100B, 200B). Use integers.
# PERIOD_TOKEN_BUDGETS: List[int] = [
#     50_000_000_000,
#     100_000_000_000,
#     200_000_000_000,
# ]


CUMULATIVE_TOKEN_BUDGETS: List[int] = [
    5_000_000,
    10_000_000,
    20_000_000,
]

# Convert cumulative budgets to period durations (tokens to add each period)
PERIOD_TOKEN_BUDGETS: List[int] = []
prev_budget = 0
for cum_budget in CUMULATIVE_TOKEN_BUDGETS:
    period_duration = cum_budget - prev_budget
    PERIOD_TOKEN_BUDGETS.append(period_duration)
    prev_budget = cum_budget

# Convert token budgets to *period lengths in steps*.
def tokens_to_steps(tok: int) -> int:
    return int(round(tok / GLOBAL_BATCH_SIZE_TOKENS))

PERIOD_STEPS: List[int] = [tokens_to_steps(t) for t in PERIOD_TOKEN_BUDGETS]

# Calculate cumulative steps for checkpointing
CUM_PERIOD_STEPS: List[int] = []
_s = 0
for L in PERIOD_STEPS:
    _s += L
    CUM_PERIOD_STEPS.append(_s)

# Scheduler knobs
LR = 1e-4
WARMUP_FRACTION_FIRST = 0.10     # 10% of period 0
DECAY_FRACTION_EACH   = 0.10     # 10% decay inside every period (including period 0)
ETA_MIN = 0.0                    # WSD‑S default: go to exact zero at boundaries

# Total training duration in steps = sum of all periods
MAX_DURATION_STEPS = sum(PERIOD_STEPS)

# Logging / eval
EVAL_INTERVAL_STEPS = max(1, PERIOD_STEPS[0] // 2)  # e.g., twice per first period (tune as you like)
PREEMPTIBLE = True

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_1B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=4,
        hidden_size_multiple_of=256,
    )
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[1025, 1025, 1025, -1],
    )
    config.block.attention.use_flash = True
    config.block.attention.use_head_qk_norm = True
    return config

def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = 2 * SEQUENCE_LENGTH  # example

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
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=WSDS(
            units=SchedulerUnits.steps,              # we're feeding period lengths in *steps*
            period_lengths=PERIOD_STEPS,            # period 0, 1, ...
            warmup_fraction=WARMUP_FRACTION_FIRST,  # ONLY for the 1st period
            decay_fraction=DECAY_FRACTION_EACH,     # for every period
            warmup_min_lr=0.0,
            decay_min_lr=ETA_MIN,                   # 0.0 => exact 0 at boundaries
        ),
    )

def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 50
    root_dir = "/weka/oe-training-default/ai2-llm"
    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    # Where to save / log
    save_folder = f"{root_dir}/checkpoints/yanhongl/linear-rnns/{common.run_name}/"

    cfg = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.steps(MAX_DURATION_STEPS),
        )
        # NEW: save only at period boundaries (i.e., when LR==0)
        .with_callback(
            "checkpointer",
            ZeroLRCheckpointerCallback(
                save_interval=1_000_000_000,    # effectively disabled; we override save logic
                save_async=True,
                save_steps=CUM_PERIOD_STEPS,
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
            common.launch.clusters[0],
            task_set="fast",
            eval_interval=EVAL_INTERVAL_STEPS,
        )
    )
    return cfg

def set_preemptible(config: ExperimentConfig) -> None:
    if config.launch is not None:
        config.launch.preemptible = PREEMPTIBLE
        # Set GPU configuration
        if hasattr(config.launch, "num_gpus"):
            config.launch.num_gpus = NUM_GPUS
        if hasattr(config.launch, "num_nodes"):
            config.launch.num_nodes = NUM_NODES
        for cluster_config in config.launch.clusters:
            if hasattr(cluster_config, "num_gpus"):
                cluster_config.num_gpus = NUM_GPUS
            if hasattr(cluster_config, "num_nodes"):
                cluster_config.num_nodes = NUM_NODES

if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE_TOKENS,
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
