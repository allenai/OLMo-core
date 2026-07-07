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
from olmo_core.internal.experiment import CliContext, ExperimentConfig
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

# ---------------------------------------------------------------------------
# Geometry / reserved ids (match the converter + olmo_core.data.document_chunk_landmark defaults).
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 40960  # 32k-scale window (matrix-comparable); landmark: 40960 / 64 = 640 blocks.
MEM_FREQ = 63  # landmark block size = 64
NUM_NODES = 2  # 2x8=16 H200 data-parallel (docchunk has no CP; DP only). 16 inst/step.
EOS_TOKEN_ID = 151643
LANDMARK_TOKEN_ID = 151860
DOC_START_ID = 151648  # <|box_start|>
DOC_END_ID = 151649  # <|box_end|>
PAD_TOKEN_ID = 151863  # interior window-fill padding (landmark only)

# ---------------------------------------------------------------------------
# Doc-chunked data (weka). NOTE: the original ``cptmix_docchunk_ladder40k`` root is EMPTY on
# weka/s3 (verified 2026-07) -- the current per-task box-marker (dense-emit) shards live under
# ``single_task_docchunk_v2/{task}_dense`` (the SAME root the docchunk singletask-ladder launcher
# reads). Only the DENSE emit exists there; the ``landmark``/``compressive`` variants still need
# their ``{task}_landmark`` shards built before they can run. Override via env DOCCHUNK_DATA_ROOT.
# ---------------------------------------------------------------------------
DOC_DATA_ROOT = os.environ.get(
    "DOCCHUNK_DATA_ROOT",
    # FIXED tokenization (leak fix + titles off), full 20k/task, built 2026-07-07. Hardcoded as the
    # default so the on-node config rebuild (Beaker) uses it WITHOUT relying on an env var (which is
    # NOT propagated to the job -> would silently fall back to the old buggy shards).
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/docchunk_5task_fixed40k",
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
# random_doc variant: each doc attends itself + a seeded-random ~this-fraction of EARLIER docs.
# Overridable per launch via env for a sweep (encode it in the run name).
DOC_KEEP_PROB = float(os.environ.get("DOCCHUNK_RANDOM_DOC_KEEP_PROB", "0.1"))
RANDOM_DOC_SEED = int(os.environ.get("DOCCHUNK_RANDOM_DOC_SEED", "42"))

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
MAX_STEPS = 2200  # ~700M content tokens @ 16 inst/step (match landmark-ref budget)


def _task_source(emit: str, name: str, doc_tok) -> NumpyDocumentSourceConfig:
    r = f"{DOC_DATA_ROOT}/{name}_{emit}"
    return NumpyDocumentSourceConfig(
        source_paths=[f"{r}/token_ids_part_*.npy"],
        tokenizer=doc_tok,
        label_mask_paths=[f"{r}/labels_mask_*.npy"],
        expand_glob=True,
    )


def build_docchunk_experiment(
    cli_context: CliContext, variant: str, flex_block_size: Optional[int] = None
) -> ExperimentConfig:
    assert variant in ("dense", "hierarchical", "random_doc", "landmark", "compressive")
    # compressive consumes the SAME landmark-format doc-chunked data (block-aligned + landmark tokens).
    emit = "landmark" if variant in ("landmark", "compressive") else "dense"
    base_checkpoint = {
        "dense": DENSE_BASE,
        "hierarchical": DENSE_BASE,
        "random_doc": DENSE_BASE,  # dense DocumentChunkedAttention (random cross-doc mask), dense base.
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
        # The Beaker job RE-BUILDS this config on the node, so propagate the random_doc hyperparameters
        # (resolved from the launch-host env here) or the on-node rebuild would silently fall back to the
        # module defaults (0.1 / 42). Harmless for the non-random_doc variants.
        beaker_launch_config.env_vars.append(
            BeakerEnvVar(name="DOCCHUNK_RANDOM_DOC_KEEP_PROB", value=repr(DOC_KEEP_PROB))
        )
        beaker_launch_config.env_vars.append(
            BeakerEnvVar(name="DOCCHUNK_RANDOM_DOC_SEED", value=str(RANDOM_DOC_SEED))
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
            # FUSED block-sparse Triton kernel (CHUNK_MASK path): avoids materializing the (B,H,T,T)
            # scores (~100 GiB at T=40960 -> OOM in eager). The kernel now TOLERATES the PadToLength
            # pad tail: pad positions never attend / are attended / are treated as landmarks, and the
            # self-diagonal guard uses the query's actual position (not the landmark-decremented one)
            # so interior window-fill pad@p-1 is never force-attended. Validated fwd+grad identical to
            # eager WITH a pad tail (fp32 fwd 1.2e-7 / dv 4.8e-7; bf16 fwd 7.8e-3 / dv 1.6e-2).
            landmark_use_kernel=True,
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
        # contributes its VALUE (a compressed block summary). Compressive CPT base. Uses the FUSED
        # compressive Triton kernel (CHUNK_MASK path ported from plain landmark, pad-tail tolerant) so
        # it fits memory at 40960 -- eager materializes the (B,H,T,T) scores (~100 GiB) and OOMs.
        # Validated fwd+grad identical to eager compressive WITH a pad tail (fp32 fwd 1.2e-7 / dv
        # 4.8e-7; bf16 fwd 3.9e-3 / dv 1.6e-2).
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_compressive=True,
            mem_freq=MEM_FREQ,
            nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
            landmark_use_kernel=True,
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
    elif variant == "random_doc":
        # Dense DocumentChunkedAttention, but each context doc attends itself + a seeded-random
        # ~DOC_KEEP_PROB subset of EARLIER docs (free query/answer stay global). FlexAttention path.
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_chunked=True,
            cross_doc_mode="random_doc",
            doc_keep_prob=DOC_KEEP_PROB,
            random_doc_seed=RANDOM_DOC_SEED,
            flex_block_size=flex_block_size,
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
            # FlexAttention block-mask granularity. Default (None -> 128) misses the block-sparsity of
            # sub-128-token chunks (the mix is dominated by ~100-word docs); 32 recovers ~40-60% of it.
            flex_block_size=flex_block_size,
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
        # FULL-block AC (checkpoint attention + FFN): needed to fit 40960 on 80GB H100 (jupiter);
        # FFN-only AC leaves the doc-chunked-attention activations resident and OOMs (76+ GiB on H100,
        # only fit H200's 141GB). The earlier CheckpointError concern (flex block-mask build not
        # recompute-stable) is resolved by the S2 block-mask cache: on recompute `_get_or_build_block_mask`
        # returns the SAME cached BlockMask (keyed on chunk_ids identity/version), so the recompute is
        # deterministic. If a CheckpointError resurfaces, fall back to selected_modules + smaller seq_len.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Data: PadToLength over a 5-task MixingDocumentSource (one chunked example per window) ----
    specs = [
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "contra", doc_tokenizer_config),
            ratio=_W["contra"] / _WSUM,
            max_repetition_factor=8.0,
            label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "nq", doc_tokenizer_config),
            ratio=_W["nq"] / _WSUM,
            max_repetition_factor=8.0,
            label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "oolong", doc_tokenizer_config),
            ratio=_W["oolong"] / _WSUM,
            max_repetition_factor=8.0,
            label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "rerank", doc_tokenizer_config),
            ratio=_W["rerank"] / _WSUM,
            max_repetition_factor=8.0,
            label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_task_source(emit, "outlier", doc_tokenizer_config),
            ratio=_W["outlier"] / _WSUM,
            max_repetition_factor=8.0,
            label="outlier",
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
                # Intermediate ephemeral saves (every 500 steps, keep last 2) so the UNTESTED docchunk
                # ladder eval can be smoked on a step-500 checkpoint (~1.4h) with buffer to fix bugs
                # before the final; also a fallback if the final step is late. save_async -> minimal cost.
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
