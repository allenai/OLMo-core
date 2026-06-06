from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    LandmarkInstanceSourceConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.lm_head import LMLossImplementation
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# Qwen3-4B + SPARSE LANDMARK attention at 64k context on the 15B-token dolma3_longmino sample.
# AttentionType.sparse_landmark: a query attends fully within its own chunk but sees past chunks
# only through their landmark tokens (sub-quadratic; faster than full/dense landmark at long
# context). num_landmarks=1 matches LandmarkInstanceSource's 1-landmark-per-chunk data.
# NOTE: SparseLandmarkAttention does NOT support context parallelism, so (unlike the landmark/fast
# scripts) there is no Ulysses CP here -- each rank processes the full 64k sequence; the sparse
# attention keeps that affordable, and FSDP shards params across the 32 GPUs.
# Pair with Qwen3-4B-{landmark,fast-landmark,dense}-dolma3longmino.py.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 65536  # 64k (must be divisible by BLOCK_SIZE)
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 64512

LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

DATA_DIR = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen"
)

GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
MAX_TOKENS = 10_000_000_000  # 10B
# StepFun optimal LR (Li et al. 2025): 1.79 * n^-0.713 * d^0.307 ≈ 3.2e-4 for n≈3.65B, d=10B
LR = 3.2e-4


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
        num_nodes=4,  # 4 nodes × 8 GPUs = 32 GPUs; cp_degree=8 → 4 DP replicas
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-4B with the SPARSE landmark attention mixer (AttentionType.sparse_landmark). No YaRN.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        sparse_landmark=True,
        mem_freq=MEM_FREQ,
        num_landmarks=1,
    )
    # Fused linear cross-entropy (Liger): never materializes the full 64k x 151936 logits
    # (~19-39GB), which is the dominant memory cost without sequence parallelism (no CP here).
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # full 64k per rank (no CP)
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=400, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            # Shard params/grads/optim 8-way within each node (replicate across the 4 nodes). Unlike
            # the CP'd landmark/fast/dense runs, there is no sequence parallelism here, so each rank
            # holds the full 64k activations -- shard_degree=1 (no sharding) OOMs because the full
            # 4B params + Adam state (~64GB) leave no room. 8-way sharding frees ~56GB/GPU.
            shard_degree=8,
        ),
        # No Ulysses CP: SparseLandmarkAttention.apply_cp raises NotImplementedError. Each rank
        # processes the full 64k sequence, so use FULL activation checkpointing (recompute every
        # block; only one block's activations are live at peak) rather than budget mode -- budget
        # mode kept too many full-64k activations resident and OOM'd.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Composable data pipeline on the new dolma3_longmino sample:
    #   NumpyDocumentSource (part-*.npy, Qwen3 uint32, EOS-separated)
    #     -> ConcatAndChunkInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=64512)
    #     -> LandmarkInstanceSource (insert landmark token every MEM_FREQ tokens -> seq_len=65536)
    instance_source_config = LandmarkInstanceSourceConfig(
        source=ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=[f"{DATA_DIR}/part-*.npy"],
                    tokenizer=tokenizer_config,
                    expand_glob=True,
                    source_group_size=1,
                )
            ],
            sequence_length=CONTENT_SEQUENCE_LENGTH,
        ),
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path="/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3-4B/model_and_optim",
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
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
    Qwen3-4B + SPARSE landmark attention at 64k on the 15B dolma3_longmino sample (4 nodes, urgent).
    Sub-quadratic: queries see past chunks only through their landmark tokens (num_landmarks=1).
    No context parallelism (SparseLandmarkAttention doesn't support it); each rank runs full 64k.

        python src/scripts/train/Qwen3/Qwen3-4B-sparse-landmark-dolma3longmino.py \\
            launch my-run ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
