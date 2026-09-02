"""
SANITY CHECK: **OOLONG-only** document-chunked DENSE SFT of Qwen3-4B at a **SMALL 8192 window**.

Purpose: the 5-task-mix doc-chunked model scores OOLONG only ~0.20 at 8k/16k/32k (vs dense/compressive
full-attention 0.64-0.68). The trace shows this is architectural -- under ``cross_doc_mode="chunked"``
each OOLONG ``||`` item is an isolated chunk, so cross-item counting/aggregation must funnel through the
FREE answer tokens over independently-encoded per-item reps; that degrades as the item count grows with
context. A dedicated ctx2048 doc-chunked OOLONG run previously scored 0.53 (few items).

This run isolates the failure at SMALL context: train OOLONG **alone** (no 5-task dilution) at an **8192
window** (PadToLength drops OOLONG examples longer than 8192 -> trains on the short, few-item examples),
from the dense CPT base, then eval at OOLONG 8k. If it recovers toward ~0.5, the mix@8k=0.202 was
depressed by dilution + long-window scale; if it stays ~0.2, the chunk-isolation mask is the ceiling.

Config mirrors the PROVEN-on-jupiter 5-task builder (``_docchunk_5task_32k_nocpt_common.py``): FULL-block
activation checkpointing + ``flex_block_size=128`` (the H100-safe pairing; FFN-only AC OOMs off-H200),
YaRN(factor 2), dense base. Single OOLONG source, ratio 1.0, FULL data (no subsample), 3 epochs.

    PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/Qwen3-4B-docchunk-oolong-sanity-8k-SFT.py \\
        dry_run q4b-docchunk_dense-oolong-sanity-ctx8k ai2/jupiter
    PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/Qwen3-4B-docchunk-oolong-sanity-8k-SFT.py \\
        launch  q4b-docchunk_dense-oolong-sanity-ctx8k ai2/jupiter
"""

import os
from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import (
    BeakerEnvVar,
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
)
from olmo_core.nn.lm_head import LMLossImplementation
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---- Geometry / reserved ids (match the converter + document_chunk_landmark defaults). ----
SEQUENCE_LENGTH = 8192  # SMALL window (the sanity variable): short, few-item OOLONG examples only.
NUM_NODES = 1  # single node; doc-chunked attention has no CP support.
EOS_TOKEN_ID = 151643
DOC_START_ID = 151648  # <|box_start|>
DOC_END_ID = 151649  # <|box_end|>
FLEX_BLOCK_SIZE = 128  # kernel-valid; H100-safe (see _docchunk_5task_32k_nocpt_common.py).

EPOCHS = 3
LR = 1e-5  # match the 5-task-mix builder LR.

# Same box-marker OOLONG shard the 5-task mix reads (single_task_docchunk_v2/oolong_dense).
DOCCHUNK_DATA_ROOT = os.environ.get(
    "DOCCHUNK_DATA_ROOT",
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_docchunk_v2",
)
DENSE_BASE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-dense-dolma3longmino/step2385/model_and_optim"
)

WORLD_SIZE = NUM_NODES * 8
GLOBAL_BATCH_SIZE = WORLD_SIZE * SEQUENCE_LENGTH  # 8 * 8192 -> 8 instances/step


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    # Pick the OOLONG shard from the RUN NAME (not an env var): the internal-experiment `launch`
    # RE-BUILDS this config ON THE NODE, where a locally-exported DOCCHUNK_DATA_ROOT is NOT set, so
    # an env override silently falls back to the default. The run name IS shipped, so key off it.
    _CR = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
    _rn = cli_context.run_name
    if "nocotmshard" in _rn:
        _data_root = f"{_CR}/single_task_docchunk_nocotm"  # matched no-CoT control (same docs, answer-only labels)
    elif "cotshard" in _rn:
        _data_root = f"{_CR}/single_task_docchunk_cot"  # CoT (plan) shard
    else:
        _data_root = DOCCHUNK_DATA_ROOT
    task_root = f"{_data_root}/oolong_dense"
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
        beaker_launch_config.allow_dirty = True
        beaker_launch_config.env_vars.append(
            BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True")
        )

    tokenizer_config = TokenizerConfig.qwen3()
    # EOS-separated instances; qwen3 ties bos==eos, so drop BOS for document-boundary detection.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model: document-chunked DENSE attention + YaRN, flex128 (matches the 5-task mix). ----
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        document_chunked=True,
        cross_doc_mode="chunked",
        flex_block_size=FLEX_BLOCK_SIZE,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
    )
    model_config.document_chunk_attention = {
        "doc_start_id": DOC_START_ID,
        "doc_end_id": DOC_END_ID,
        "eos_id": EOS_TOKEN_ID,
        "mode": "chunked",
    }
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
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
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=WORLD_SIZE,
        ),
        # FULL-block AC (matches the 5-task mix; recompute-stable via the S2 block-mask cache).
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Single OOLONG source, ratio 1.0, FULL data (no subsample), NO CPT. ----
    task_source = NumpyDocumentSourceConfig(
        source_paths=[f"{task_root}/token_ids_part_*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{task_root}/labels_mask_*.npy"],
        expand_glob=True,
    )
    mixing = MixingDocumentSourceConfig(
        source_specs=[
            MixingDocumentSourceSpecConfig(
                source=task_source,
                ratio=1.0,
                max_repetition_factor=8.0,
                label="oolong",
            )
        ]
    )
    instance_source_config = PadToLengthInstanceSourceConfig(
        sources=[mixing],
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=doc_tokenizer_config,
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
            load_path=DENSE_BASE,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(EPOCHS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100000,
                ephemeral_save_interval=500,
                max_checkpoints=2,
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
            "slack_notifier", SlackNotifierCallback(name=run_name_with_ts, enabled=False)
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
    main(config_builder=build_experiment_config)
