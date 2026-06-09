from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# Single-epoch SFT for Qwen3-4B DENSE baseline (full flash-attn 2 + YaRN).
#
# Initializes model weights from the dense *pretrained* checkpoint
# (q4b-dense-dolma3longmino/step2385) and fine-tunes on the rlhn SFT set. Uses YaRN
# (factor 2) to extend the native 32k context to the 64k SFT context, matching the
# dense pretraining run. No landmark tokens are inserted.
#
# Data pipeline (composable):
#   PackingInstanceSource(token_ids + labels_mask)   # bin-pack SFT conversations, carry loss mask
#
# DOC MASKING: this dense run leaves generate_doc_lengths=False so the data pipeline matches the
# fast/sparse landmark SFT runs (landmark attention cannot do intra-document masking), keeping
# attention type the only variable across the three SFT comparisons -- and matching the dense
# pretraining regime (ConcatAndChunk, cross-document attention). Dense attention *does* support
# intra-document masking, so flip generate_doc_lengths=True below if you instead want standard
# per-conversation SFT masking for the dense baseline.
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 65536  # 64k context (no landmark insertion)

# Tokenized SFT dataset (Qwen3 tokenizer): token_ids_part_*.npy + labels_mask_*.npy.
DATASET_PATH = "/weka/oe-training-default/ai2-llm/checkpoints/amandab/rlhn_sft_qwen_63k"

# Dense pretrained checkpoint to initialize from (model weights only).
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-dense-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 4  # 4 nodes x 8 GPUs = 32 GPUs; cp_degree=8 -> 4 DP replicas

# Global batch in *tokens*. With rank_microbatch=SEQUENCE_LENGTH and 4 DP replicas,
# SEQUENCE_LENGTH * 32 = 2097152 (~2M tokens) -> 8 grad-accum steps. Kept equal across the
# dense/fast/sparse SFT runs for a clean comparison.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 32

# SFT hyperparameters: no weight decay, low LR, linear decay to zero, one epoch.
LR = 5e-5
NUM_EPOCHS = 1


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=NUM_NODES,
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-4B with YaRN context extension: native 32k -> 64k, full (flash-attn 2) attention.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # 1 instance per rank with CP
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,  # NOTE: different from pretraining
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        # Qwen3-4B: n_heads=32, n_kv_heads=8 -> head_stride=4
        cp_config=TransformerContextParallelConfig.ulysses(degree=8),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,  # disabled for SFT (cf. OLMo SFT scripts)
        max_grad_norm=1.0,
    )

    # Composable SFT data pipeline (no landmark insertion):
    #   PackingInstanceSource (token_ids_part_*.npy + labels_mask_*.npy, packed to SEQUENCE_LENGTH)
    clean_path = DATASET_PATH.rstrip("/")
    instance_source_config = PackingInstanceSourceConfig.from_npy(
        f"{clean_path}/token_ids_part_*.npy",
        tokenizer=tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        expand_glob=True,
        long_doc_strategy=LongDocStrategy.truncate,  # truncate docs over SEQUENCE_LENGTH
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # generate_doc_lengths left False to match the landmark SFT runs (see module docstring).
        # Flip to True for standard per-conversation SFT masking (dense attention supports it).
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            # Initialize from the pretrained checkpoint, weights only. The trainer first tries to
            # resume from save_folder; if nothing is there it falls back to load_path.
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(NUM_EPOCHS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(name=run_name_with_ts, enabled=False),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=[instance_source_config],
        data_loader=data_loader_config,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Single-epoch SFT of the Qwen3-4B dense baseline from the dense pretrain ckpt.

        python src/scripts/train/sft/Qwen3-4B-dense-SFT.py \\
            launch q4b-dense-sft ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
