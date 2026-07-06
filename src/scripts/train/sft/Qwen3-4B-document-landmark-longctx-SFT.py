from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PadToLengthInstanceSourceConfig,
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
# Long-context task SFT for Qwen3-4B + DOCUMENT-CHUNKED LANDMARK attention.
#
# Keeps the OLMo-core grouped-softmax landmark mechanism but masks attention with the corpus-reasoning
# "chunked" role logic (DocumentLandmarkAttention): context documents are mutually isolated, while the
# trailing query/answer (FREE) tokens bridge across documents through the landmark grouped softmax.
#
# Data is produced by src/scripts/data/convert_unified_to_document_landmark.py, which (per unified
# example) wraps each document in <|doc_start|>/<|doc_end|>, inserts a landmark every MEM_FREQ tokens,
# block-aligns every chunk, and EOS-terminates. The model reconstructs per-token chunk roles from the
# <|doc_start|>/<|doc_end|> tokens at runtime (model.document_chunk_attention, set below) -- so the
# data pipeline is just PadToLength (NO LandmarkInstanceSource; the landmarks are already inserted).
#
# Start small: SEQUENCE_LENGTH = 4096, eager (no fused kernel), no context parallelism (the chunked
# mask is built on the full local sequence). Initialized from the sparse-landmark pretrain checkpoint.
#
# Task is selected from the run name, e.g. `q4b-doclm-absence-sft` -> .../absence_doclandmark.
# ---------------------------------------------------------------------------

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 4096  # final instance length (must be divisible by BLOCK_SIZE)

# Reserved Qwen3 ids -- MUST match convert_unified_to_document_landmark.py.
EOS_TOKEN_ID = 151643
LANDMARK_TOKEN_ID = 151860
DOC_START_ID = 151861
DOC_END_ID = 151862

DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/longctx_sft_qwen"

# Sparse-landmark pretrained checkpoint to initialize from (model weights only).
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-sparse-landmark-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 1  # 1 node x 8 GPUs; no CP -> 8 DP ranks (8-way sharded within the node)
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8
LR = 5e-5
NUM_EPOCHS = 3


def resolve_dataset_path(run_name: str) -> str:
    """Pick the single document-landmark task dataset from the run name (e.g. 'absence', 'oolong')."""
    for task in ("absence", "contradiction", "retrieval", "oolong"):
        if task in run_name:
            return f"{DATA_ROOT}/{task}_doclandmark"
    raise ValueError(
        f"Run name '{run_name}' must contain a known task name to select the SFT dataset."
    )


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
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
    # The SFT shards are separated by single EOS tokens; qwen3 sets bos == eos, so drop BOS for
    # document-boundary detection (matches the other longctx SFT scripts).
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3-4B with DOCUMENT-LANDMARK attention. Runtime chunk_id reconstruction from the boundary
    # tokens is enabled via document_chunk_attention (mode "chunked": context chunks isolated, FREE
    # query/answer bridges).
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        document_landmark=True,
        mem_freq=MEM_FREQ,
    )
    model_config.document_chunk_attention = {
        "doc_start_id": DOC_START_ID,
        "doc_end_id": DOC_END_ID,
        "eos_id": EOS_TOKEN_ID,
        "mode": "chunked",
    }
    # Fused linear cross-entropy (Liger): avoids materializing the full seq x vocab logits.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # full 4k per rank (no CP)
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        # DocumentLandmarkAttention is @torch.compiler.disable (eager grouped softmax); keep compile
        # off to avoid graph breaks around the chunked-mask build.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=8,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # Data: one document-landmark instance per example (already landmark-inserted by the converter),
    # padded to SEQUENCE_LENGTH. No LandmarkInstanceSource, no packing, generate_doc_lengths=False.
    clean_path = resolve_dataset_path(cli_context.run_name)
    instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
        f"{clean_path}/token_ids_part_*.npy",
        tokenizer=doc_tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        label_mask_paths=[f"{clean_path}/labels_mask_part_*.npy"],
        expand_glob=True,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # generate_doc_lengths left False: chunk roles are reconstructed at runtime from the boundary
        # tokens, not from EOS-derived doc lengths.
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
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
    Long-context task SFT of Qwen3-4B + document-chunked landmark attention from the sparse-landmark
    pretrain ckpt. Task selected from the run name:

        python src/scripts/train/sft/Qwen3-4B-document-landmark-longctx-SFT.py \\
            launch q4b-doclm-absence-sft ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
