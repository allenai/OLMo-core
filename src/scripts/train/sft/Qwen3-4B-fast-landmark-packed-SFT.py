from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
)
from olmo_core.data.composable.numpy_document_source import NumpyDocumentSourceConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
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

# ---------------------------------------------------------------------------
# Single-epoch SFT for Qwen3-4B + FAST LANDMARK attention, with CORRECTLY-MASKED
# sequence packing (each conversation is its own document; no cross-document attention).
#
# This is the *packed* counterpart of ``Qwen3-4B-fast-landmark-SFT.py``. The difference is entirely
# in the data pipeline and one model-parallelism setting:
#
#   * Data: ``LandmarkPackingInstanceSource`` (NOT ``ConcatAndChunk -> LandmarkInstanceSource``).
#     It inserts landmark tokens *per document*, so every document occupies a whole number of
#     landmark blocks, then greedily packs whole documents into ``SEQUENCE_LENGTH`` windows and emits
#     block-aligned ``doc_lens``. The trainer forwards those ``doc_lens`` to the model, which masks
#     attention block-diagonally (a query never attends across a document boundary) and resets RoPE
#     positions per document -- i.e. packed training is equivalent to training each conversation
#     alone. Documents that don't fit in one window are dropped (a warning is logged if many are; if
#     so, raise ``SEQUENCE_LENGTH``).
#
#   * ``generate_doc_lengths`` is left False: the document boundaries come from the packing source,
#     not from EOS tokens. (EOS-derived boundaries are NOT block-aligned and would be rejected by the
#     landmark attention with a clear error -- this is the guard against the *wrong* kind of packing.)
#
#   * NO context parallelism. Packing relies on ``cu_doc_lens``, which the landmark attention does not
#     support together with context parallelism (it raises ``NotImplementedError``). We shard with
#     FSDP instead. If you OOM, increase ``shard_degree`` (or lower ``SEQUENCE_LENGTH``).
#
# Contrast with the cross-attending script: there, packed conversations attend across each other
# (matching landmark *pretraining*). Use this script when you want each SFT example masked off.
# ---------------------------------------------------------------------------

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 65536  # emitted (landmark-space) instance length; must be divisible by BLOCK_SIZE

LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

RLHN_DATASET_PATH = "/weka/oe-training-default/ai2-llm/checkpoints/amandab/rlhn_sft_qwen_63k"
RAG_NEAR_DATASET_PATH = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_sft_qwen/nq_hotpotqa_near"
)


def resolve_dataset_path(run_name: str) -> str:
    """NEAR rung (NQ+HotpotQA RAG-QA) when the run name contains 'rag', else the FAR (RLHN) rung."""
    return RAG_NEAR_DATASET_PATH if "rag" in run_name else RLHN_DATASET_PATH


BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-fast-landmark-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 4

# Global batch in *tokens* (incl. landmark and pad tokens). With rank_microbatch=SEQUENCE_LENGTH and
# 32 GPUs (no CP, so every rank is its own DP replica), SEQUENCE_LENGTH * 32 ~ 2M tokens.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 32

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
    # Qwen3 sets bos_token_id == eos_token_id, so drop the BOS for NumpyDocumentSource boundary
    # detection (single-EOS docs would otherwise merge into one giant doc and get dropped).
    from dataclasses import replace

    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_landmark=True,
        mem_freq=MEM_FREQ,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one packed window per rank
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
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            # No context parallelism with packing (cu_doc_lens + CP is unsupported); shard with FSDP.
            # Bump this if you OOM at SEQUENCE_LENGTH=65536.
            shard_degree=8,
        ),
        # NOTE: cp_config intentionally omitted -- packing (cu_doc_lens) is incompatible with CP.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # Packed landmark SFT data pipeline (block-aligned, intra-document masked):
    #   NumpyDocumentSource (EOS-delimited SFT examples + labels_mask)
    #     -> LandmarkPackingInstanceSource (per-doc landmark insertion + greedy packing + doc_lens)
    clean_path = resolve_dataset_path(cli_context.run_name).rstrip("/")
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=NumpyDocumentSourceConfig(
            source_paths=[f"{clean_path}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
            expand_glob=True,
        ),
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # IMPORTANT: leave generate_doc_lengths=False. The packing source emits block-aligned
        # doc_lens; EOS-derived doc_lens would NOT be block-aligned and the landmark attention would
        # reject them.
        generate_doc_lengths=False,
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
    Single-epoch packed (intra-document masked) SFT of Qwen3-4B + fast-landmark attention.

        python src/scripts/train/sft/Qwen3-4B-fast-landmark-packed-SFT.py \\
            launch q4b-fast-landmark-packed-sft ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
