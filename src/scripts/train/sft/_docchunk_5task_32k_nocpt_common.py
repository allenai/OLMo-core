"""
Shared builder for the **document-chunked** 5-task 32k no-CPT SFT matrix rows (Beaker/gantry).

Three variants, all reading the SAME doc-chunked weka data (built by
``convert_docchunk_5task_gantry.sh`` -> ``cptmix_docchunk_ladder40k/{task}_{dense|landmark}``) and
mixing the 5 tasks (contradiction / nq / oolong / rerank / outlier) at the headline weights:

  * ``dense``        -> :class:`DocumentChunkedAttention` (cross_doc_mode="chunked"), dense base.
  * ``hierarchical`` -> :class:`DocumentChunkedAttention` (cross_doc_mode="hierarchical_dilated",
                        dilation_n=4, dilation_m=2), SAME dense docchunk data + dense base.
  * ``landmark``     -> :class:`DocumentLandmarkAttention` (grouped-softmax), landmark docchunk data +
                        fast-landmark base.

Differs from the packed dense/landmark 32k matrix scripts (Qwen3-4B-dense-5task-32k-nocpt-SFT.py):
the document-chunked attention reconstructs per-token ``chunk_id`` roles from the
``<|box_start|>``/``<|box_end|>`` markers, which requires ONE EOS-terminated example per instance
(everything after the first EOS is PAD) -- so it CANNOT use ConcatAndChunk packing or context
parallelism. Layout is therefore PadToLength (one already-chunked example per 40960 window, padded)
over a MixingDocumentSource, FSDP-sharded on a single 8xH200 node (NUM_NODES=1, no CP).
"""

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
from olmo_core.internal.experiment import CliContext, ExperimentConfig
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
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
# Geometry / reserved ids (match the converter + olmo_core.data.document_chunk_landmark defaults).
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 40960  # 32k-scale window (matrix-comparable); landmark: 40960 / 64 = 640 blocks.
MEM_FREQ = 63           # landmark block size = 64
NUM_NODES = 1           # single 8xH200 node; doc-chunked attn has no CP support.
EOS_TOKEN_ID = 151643
LANDMARK_TOKEN_ID = 151860
DOC_START_ID = 151648   # <|box_start|>
DOC_END_ID = 151649     # <|box_end|>
PAD_TOKEN_ID = 151863   # interior window-fill padding (landmark only)

# ---------------------------------------------------------------------------
# Doc-chunked data (weka) -- built by convert_docchunk_5task_gantry.sh.
# ---------------------------------------------------------------------------
DOC_DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/cptmix_docchunk_ladder40k"
)
# Matched CPT bases on weka (weights-only). dense base also feeds the hierarchical variant.
DENSE_BASE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-dense-dolma3longmino/step2385/model_and_optim"
)
LANDMARK_BASE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-fast-landmark-dolma3longmino/step2385/model_and_optim"
)
# Compressive CPT base (the landmark-token embedding + compressive grouped-softmax were trained here).
COMPRESSIVE_BASE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-base-fast-compressive-landmark-8node/step2385/model_and_optim"
)
NONSELECTED_LANDMARK_MASS = 0.1  # alpha for compressive attention

# ---------------------------------------------------------------------------
# Mix weights -- IDENTICAL to the packed 32k no-CPT rows (sum 7).
# ---------------------------------------------------------------------------
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())

# ---------------------------------------------------------------------------
# Optimization / budget. One PadToLength window per GPU/step (grad-accum 1) -> 8 instances/step.
# MAX_STEPS targets ~1 epoch over the mixed doc-chunked instances (overridable at launch via
# --trainer.max_duration.value=...). 8 GPUs * MAX_STEPS instance-views.
# ---------------------------------------------------------------------------
LR = 1e-5
WORLD_SIZE = NUM_NODES * 8
GLOBAL_BATCH_SIZE = WORLD_SIZE * SEQUENCE_LENGTH  # tokens; 8 * 40960 = 327680 -> 8 instances/step
MAX_STEPS = 2000


def _task_source(emit: str, name: str, doc_tok) -> NumpyDocumentSourceConfig:
    r = f"{DOC_DATA_ROOT}/{name}_{emit}"
    return NumpyDocumentSourceConfig(
        source_paths=[f"{r}/token_ids_part_*.npy"],
        tokenizer=doc_tok,
        label_mask_paths=[f"{r}/labels_mask_*.npy"],
        expand_glob=True,
    )


def build_docchunk_experiment(cli_context: CliContext, variant: str) -> ExperimentConfig:
    assert variant in ("dense", "hierarchical", "landmark", "compressive")
    # compressive consumes the SAME landmark-format doc-chunked data (block-aligned + landmark tokens).
    emit = "landmark" if variant in ("landmark", "compressive") else "dense"
    base_checkpoint = {
        "dense": DENSE_BASE,
        "hierarchical": DENSE_BASE,
        "landmark": LANDMARK_BASE,
        "compressive": COMPRESSIVE_BASE,
    }[variant]

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
        # The OLMo-core working tree on this host is a shared checkout actively modified by other
        # concurrent jobs, so it's never clean. gantry clones the committed HEAD regardless, which is
        # what we want (all four variants' code is committed + pushed), so bypass the clean-tree guard.
        beaker_launch_config.allow_dirty = True
        # Doc-chunked attention cannot use CP, so the full 40960-token activation set lives on ONE GPU
        # and FlexAttention needs ~12 GiB of working memory on top. PyTorch's default caching allocator
        # stranded ~16 GiB as reserved-but-unallocated (fragmentation) -> OOM. expandable_segments
        # reclaims that. Requires an 80 GiB+ GPU cluster (e.g. jupiter H100); 44 GiB nodes are too small.
        beaker_launch_config.env_vars.append(
            BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True")
        )

    tokenizer_config = TokenizerConfig.qwen3()
    # EOS-separated instances; qwen3 ties bos==eos, so drop BOS for document-boundary detection.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model: document-chunked attention (dense / hierarchical-dilated / landmark) ----
    if variant == "landmark":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_landmark=True,
            mem_freq=MEM_FREQ,
            # EAGER grouped-softmax (no fused kernel): PadToLength right-pads variable-length
            # doc-landmark instances, whose pad tail has no landmarks -> the kernel's positional
            # is_mem assert would fail. Eager tolerates the pad tail.
            landmark_use_kernel=False,
        )
        model_config.document_chunk_attention = {
            "doc_start_id": DOC_START_ID,
            "doc_end_id": DOC_END_ID,
            "eos_id": EOS_TOKEN_ID,
            "mode": "chunked",
            "pad_id": PAD_TOKEN_ID,
        }
    elif variant == "compressive":
        # Same chunked mask + grouped softmax as landmark, but each past block's landmark token also
        # contributes its VALUE (a compressed block summary). Eager-only; compressive CPT base.
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_compressive=True,
            mem_freq=MEM_FREQ,
            nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
        )
        model_config.document_chunk_attention = {
            "doc_start_id": DOC_START_ID,
            "doc_end_id": DOC_END_ID,
            "eos_id": EOS_TOKEN_ID,
            "mode": "chunked",
            "pad_id": PAD_TOKEN_ID,
        }
    elif variant == "hierarchical":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_chunked=True,
            cross_doc_mode="hierarchical_dilated",
            dilation_n=4,
            dilation_m=2,
        ).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
        )
        model_config.document_chunk_attention = {
            "doc_start_id": DOC_START_ID,
            "doc_end_id": DOC_END_ID,
            "eos_id": EOS_TOKEN_ID,
            "mode": "chunked",
        }
    else:  # dense
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_chunked=True,
            cross_doc_mode="chunked",
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
        # The chunked mask / grouped softmax are eager (@torch.compiler.disable); keep compile off.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=WORLD_SIZE,
        ),
        # Checkpoint ONLY the FFN (largest activations), NOT the attention: full-block AC recomputes
        # the doc-chunked attention, whose FlexAttention block-mask / eager chunked-mask build does not
        # save a recompute-stable number of tensors (-> CheckpointError on torch 2.9). FFN-only AC
        # keeps attention out of the recompute and still fits 40960 on H200.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Data: PadToLength over a 5-task MixingDocumentSource (one chunked example per window) ----
    specs = [
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "contra", doc_tokenizer_config),
            ratio=_W["contra"] / _WSUM, max_repetition_factor=8.0, label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "nq", doc_tokenizer_config),
            ratio=_W["nq"] / _WSUM, max_repetition_factor=8.0, label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "oolong", doc_tokenizer_config),
            ratio=_W["oolong"] / _WSUM, max_repetition_factor=8.0, label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "rerank", doc_tokenizer_config),
            ratio=_W["rerank"] / _WSUM, max_repetition_factor=8.0, label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "outlier", doc_tokenizer_config),
            ratio=_W["outlier"] / _WSUM, max_repetition_factor=8.0, label="outlier",
        ),
    ]

    instance_source_config = PadToLengthInstanceSourceConfig(
        sources=[MixingDocumentSourceConfig(source_specs=specs)],
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=doc_tokenizer_config,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # chunk roles are reconstructed from boundary tokens, NOT EOS-derived doc lengths.
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=base_checkpoint,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100000,
                ephemeral_save_interval=MAX_STEPS,
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
