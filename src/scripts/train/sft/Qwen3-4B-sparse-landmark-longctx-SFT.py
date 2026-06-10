from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    LandmarkInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
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

# ---------------------------------------------------------------------------
# Long-context task SFT (oolong / contradiction) for Qwen3-4B + SPARSE LANDMARK attention.
#
# Initializes model weights from the sparse-landmark *pretrained* checkpoint
# (q4b-sparse-landmark-dolma3longmino/step2385) and fine-tunes on ONE task dataset (no
# mixing). The task is selected from the run name (see resolve_dataset_path):
#
#     q4b-sparse-oolong-sft        -> longctx_sft_qwen/oolong         (plain answers)
#     q4b-sparse-oolong-cot-sft    -> longctx_sft_qwen/oolong_cot     (plan CoT)
#     q4b-sparse-contra-sft        -> longctx_sft_qwen/contradiction  (plain answers)
#     q4b-sparse-contra-cot-sft    -> longctx_sft_qwen/contradiction_cot (enumerate CoT)
#
# Datasets are produced by src/scripts/data/convert_longctx_tasks_to_sft.py (Qwen3 chat
# template, corpus-reasoning prompt formats, query_position=both).
#
# Data pipeline (composable):
#   ConcatAndChunkInstanceSource(token_ids + labels_mask)  # concat + chunk to CONTENT len, carry mask
#     -> LandmarkInstanceSource(mem_freq, mem_id)           # insert a landmark token every MEM_FREQ
#
# NOTE: we use ConcatAndChunk (not bin-packing) because PackingInstanceSource builds a SegmentTree
# whose size must be a power of 2, and the landmark content length (a multiple of MEM_FREQ) can never
# be a power of 2. ConcatAndChunk also matches the landmark pretraining regime exactly.
#
# PARALLELISM: SparseLandmarkAttention does NOT support Ulysses CP (apply_cp raises), so unlike the
# dense/fast SFT runs there is no sequence parallelism here -- each rank processes the full 64k
# sequence. We shard params/grads/optim 8-way within each node (hsdp shard_degree=8) to fit, use
# FULL activation checkpointing, and a fused-linear (Liger) LM loss to avoid materializing the full
# 64k x vocab logits. These mirror the sparse-landmark pretraining script.
#
# NOTE ON INTRA-DOCUMENT MASKING: landmark attention does not support intra-document (cu_doc_lens)
# masking, so generate_doc_lengths is left False (matches the landmark pretraining regime).
# ---------------------------------------------------------------------------

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 65536  # final instance length (must be divisible by BLOCK_SIZE)
# Content length fed to the packer (landmark tokens are inserted afterwards):
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 64512

LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

# Tokenized SFT datasets (Qwen3 tokenizer): token_ids_part_*.npy + labels_mask_*.npy per task,
# produced by src/scripts/data/convert_longctx_tasks_to_sft.py.
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/longctx_sft_qwen"


def resolve_dataset_path(run_name: str) -> str:
    """Pick the single task dataset (no mixing) from the run name."""
    if "oolong" in run_name:
        task = "oolong"
    elif "contra" in run_name:
        task = "contradiction"
    else:
        raise ValueError(
            f"Run name '{run_name}' must contain 'oolong' or 'contra' to select the SFT dataset."
        )
    if "cot" in run_name:
        task += "_cot"
    return f"{DATA_ROOT}/{task}"


# Sparse-landmark pretrained checkpoint to initialize from (model weights only).
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-sparse-landmark-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 1  # 1 node x 8 GPUs; no CP -> 8 DP ranks (8-way sharded within the node)

# Global batch in *tokens* (incl. landmark tokens). The task datasets are small (~50-150M train
# tokens), so we use a single node and a 512k-token global batch (the floor for sparse: 8 DP
# ranks x rank_microbatch=SEQUENCE_LENGTH) to get ~4x more optimizer steps than the 4-node rlhn
# setup. Kept equal across the dense/fast/sparse longctx SFT runs for a clean comparison.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8

# SFT hyperparameters: no weight decay, low LR, linear decay to zero. 3 epochs (matches the
# corpus-reasoning task-SFT configs; the task datasets are small).
LR = 5e-5
NUM_EPOCHS = 3


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    # Keep all artifacts from these experiments together under the prasanns/ namespace
    # (never write to the shared top-level checkpoints dir).
    save_dir = f"{root_dir}/checkpoints/prasanns/{cli_context.run_name}"

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

    # Qwen3-4B with the SPARSE landmark attention mixer (AttentionType.sparse_landmark). No YaRN.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        sparse_landmark=True,
        mem_freq=MEM_FREQ,
        num_landmarks=1,
    )
    # Fused linear cross-entropy (Liger): never materializes the full 64k x vocab logits, which is
    # the dominant memory cost without sequence parallelism (no CP here).
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # full 64k per rank (no CP)
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
            # Shard params/grads/optim 8-way within each node (replicate across the 4 nodes). With no
            # sequence parallelism each rank holds the full 64k activations, so shard_degree=1 OOMs
            # (4B params + Adam state ~64GB); 8-way sharding frees ~56GB/GPU.
            shard_degree=8,
        ),
        # No Ulysses CP: SparseLandmarkAttention.apply_cp raises NotImplementedError. Each rank
        # processes the full 64k sequence, so use FULL activation checkpointing (recompute every
        # block; only one block's activations live at peak) -- budget mode OOM'd.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,  # disabled for SFT (cf. OLMo SFT scripts)
        max_grad_norm=1.0,
    )

    # Composable SFT data pipeline:
    #   ConcatAndChunkInstanceSource (token_ids_part_*.npy + labels_mask_*.npy, chunked to CONTENT len)
    #     -> LandmarkInstanceSource (insert landmark token every MEM_FREQ tokens -> SEQUENCE_LENGTH)
    clean_path = resolve_dataset_path(cli_context.run_name)
    instance_source_config = LandmarkInstanceSourceConfig(
        source=ConcatAndChunkInstanceSourceConfig.from_npy(
            f"{clean_path}/token_ids_part_*.npy",
            tokenizer=tokenizer_config,
            sequence_length=CONTENT_SEQUENCE_LENGTH,
            label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
            expand_glob=True,
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
        # generate_doc_lengths left False: landmark attention does not support intra-document masking.
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
                entity="prasanns-allen-institute-for-ai",
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
    Long-context task SFT (oolong / contradiction) of Qwen3-4B + sparse-landmark attention from
    the sparse-landmark pretrain ckpt. The task is selected from the run name:

        python src/scripts/train/sft/Qwen3-4B-sparse-landmark-longctx-SFT.py \\
            launch q4b-sparse-oolong-sft ai2/jupiter-cirrascale-2
        python src/scripts/train/sft/Qwen3-4B-sparse-landmark-longctx-SFT.py \\
            launch q4b-sparse-contra-cot-sft ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
