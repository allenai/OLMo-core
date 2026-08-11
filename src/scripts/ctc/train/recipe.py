"""
One parameterised training recipe, shared by SFT and CPT and by both clusters.

The pre-migration tree encoded each run as its own launcher: 20 CPT scripts of ~210 lines differing
only in architecture, learning rate and node count, and a parallel set for SFT. This is those
scripts with the varying parts lifted into :class:`~options.TrainOptions`.

Three things differ between SFT and CPT, and nothing else:

===========  ==========================================  ==============================
             SFT                                         CPT
===========  ==========================================  ==============================
layout       one already-chunked example per window,     documents concatenated and
             padded (``PadToLength``)                    chunked (``ConcatAndChunk``)
loss         answer tokens only (``labels_mask``)        every token
budget       steps                                       tokens
===========  ==========================================  ==============================

The document-chunked masks reconstruct per-token chunk roles from the ``<|box_start|>`` /
``<|box_end|>`` markers, which needs one EOS-terminated example per instance. That is why SFT
cannot use packing or context parallelism, not a preference.

Everything olmo-core is imported inside functions, so ``options.py`` and this module's docstring
stay readable on a machine with no torch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from options import TrainOptions

__all__ = ["build_experiment", "build_model_config", "build_trainer_config"]

#: Per-architecture ``TransformerConfig`` keyword arguments. A table rather than a branch chain:
#: adding a mask variant should be one row, and the pre-migration version of this was a 150-line
#: if/elif in which the shared parts had drifted apart.
ARCH_KWARGS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "chunked": {"document_chunked": True, "cross_doc_mode": "chunked"},
    "hierarchical": {
        "document_chunked": True,
        "cross_doc_mode": "hierarchical_dilated",
        "dilation_n": 4,
        "dilation_m": 2,
        "dilation_cycle": 3,
    },
    "landmark": {"document_landmark": True, "landmark_use_kernel": True},
}


def build_model_config(options: TrainOptions, vocab_size: int):
    """
    Build the model config for the requested architecture.

    :param options: Resolved options.
    :param vocab_size: Padded vocabulary size.

    :returns: A ``TransformerConfig``.

    :raises ValueError: If ``options.model`` is not a ``TransformerConfig`` factory.
    """
    from olmo_core.data.document_chunk_landmark import reserved_ids
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.rope import YaRNRoPEScalingConfig
    from olmo_core.nn.transformer import TransformerConfig

    factory = getattr(TransformerConfig, options.model, None)
    if factory is None:
        raise ValueError(f"TransformerConfig has no factory {options.model!r}")

    kwargs = dict(ARCH_KWARGS[options.arch])
    if options.arch == "landmark":
        kwargs["mem_freq"] = options.mem_freq
    config = factory(vocab_size=vocab_size, **kwargs)

    # YaRN extends Qwen3's native 32k. Applied for the dense-mask variants only, matching the
    # reference runs; the landmark kernel carries its own positional handling.
    if options.seq_len > 32768 and options.arch != "landmark":
        config = config.with_rope_scaling(
            YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
        )

    if options.arch != "full":
        ids = reserved_ids(options.tokenizer)
        # The mask needs the boundary ids to derive chunk roles. pad_id matters only for the
        # landmark layout, whose window fill must not be treated as content.
        config.document_chunk_attention = {
            "doc_start_id": ids.doc_start,
            "doc_end_id": ids.doc_end,
            "eos_id": ids.eos,
            "mode": "chunked",
            **({"pad_id": ids.pad} if options.arch == "landmark" else {}),
        }
    config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    return config


def build_train_module_config(options: TrainOptions):
    """
    :param options: Resolved options.

    :returns: A ``TransformerTrainModuleConfig``.
    """
    from olmo_core.config import DType
    from olmo_core.distributed.parallel import DataParallelType
    from olmo_core.float8 import Float8Config
    from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
    from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
    from olmo_core.train.train_module import (
        TransformerActivationCheckpointingConfig,
        TransformerDataParallelConfig,
        TransformerDataParallelWrappingStrategy,
        TransformerTrainModuleConfig,
    )

    return TransformerTrainModuleConfig(
        rank_microbatch_size=options.seq_len,
        max_sequence_length=options.seq_len,
        optim=SkipStepAdamWConfig(
            lr=options.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=options.warmup_fraction, alpha_f=0.0),
        # The chunked mask and grouped softmax are eager (@torch.compiler.disable), so compiling
        # buys nothing and has cost a lot of debugging time.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=options.world_size,
        ),
        # Full-block AC. Without context parallelism the whole 40960-token activation set lives on
        # one GPU and FFN-only checkpointing OOMs on an 80 GiB card.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )


def build_dataset_config(options: TrainOptions, tokenizer_config):
    """
    Build the instance source: padded one-per-window for SFT, packed for CPT.

    :param options: Resolved options.
    :param tokenizer_config: The document tokenizer config (BOS dropped).

    :returns: An instance-source config.
    """
    from olmo_core.data.composable import (
        ConcatAndChunkInstanceSourceConfig,
        MixingDocumentSourceConfig,
        MixingDocumentSourceSpecConfig,
        NumpyDocumentSourceConfig,
        PadToLengthInstanceSourceConfig,
    )

    def source(spec) -> Any:
        return NumpyDocumentSourceConfig(
            source_paths=[f"{spec.path}/token_ids_part_*.npy"],
            tokenizer=tokenizer_config,
            # CPT trains on every token, so it has no label mask to read.
            label_mask_paths=(
                [f"{spec.path}/labels_mask_*.npy"] if options.mode == "sft" else None
            ),
            expand_glob=True,
        )

    specs = [
        MixingDocumentSourceSpecConfig(
            source=source(spec), ratio=ratio, max_repetition_factor=8.0, label=spec.name
        )
        for spec, ratio in options.weights()
    ]
    mixed = MixingDocumentSourceConfig(source_specs=specs)

    if options.mode == "sft":
        return PadToLengthInstanceSourceConfig(
            sources=[mixed], sequence_length=options.seq_len, tokenizer=tokenizer_config
        )
    # No `tokenizer` here, unlike PadToLength: concatenation never needs to pad, so it has no pad
    # id to look up. Passing one is a TypeError.
    return ConcatAndChunkInstanceSourceConfig(sources=[mixed], sequence_length=options.seq_len)


def build_trainer_config(options: TrainOptions, *, save_folder: str, work_dir: str):
    """
    :param options: Resolved options.
    :param save_folder: Checkpoint destination.
    :param work_dir: Scratch directory for the data loader.

    :returns: A ``TrainerConfig`` with the callbacks every run should have.
    """
    from olmo_core.train import Duration, LoadStrategy, TrainerConfig
    from olmo_core.train.callbacks import (
        CheckpointerCallback,
        ConfigSaverCallback,
        WandBCallback,
    )

    stamped = f"{options.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    duration = (
        Duration.steps(options.max_steps)
        if options.max_steps is not None
        else Duration.tokens(options.max_tokens)
    )

    config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            load_path=options.base,
            load_strategy=LoadStrategy.always if options.base else LoadStrategy.if_available,
            # Weights only: this is a new run over new data, and inheriting the base's optimizer
            # state or step count would make the schedule and the logged step numbers meaningless.
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=duration,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=options.save_interval,
                ephemeral_save_interval=options.ephemeral_save_interval,
                max_checkpoints=2,
                save_async=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )
    if options.wandb_project:
        config = config.with_callback(
            "wandb",
            WandBCallback(
                name=stamped,
                group=options.run_name,
                project=options.wandb_project,
                enabled=True,
                cancel_check_interval=10,
            ),
        )
    if options.fingerprint:
        from ctc.train import FormatFingerprintCallback

        # Collected from the shard directories themselves, not declared here: a launcher's
        # declaration is a claim about the data, and when the two disagree the launcher is wrong.
        config = config.with_callback(
            "format_fingerprint",
            FormatFingerprintCallback(collect_from=[spec.path for spec in options.data]),
        )
    return config


def build_experiment(options: TrainOptions, *, save_folder: str, work_dir: str):
    """
    Assemble the whole run.

    :param options: Resolved options.
    :param save_folder: Checkpoint destination.
    :param work_dir: Scratch directory.

    :returns: ``(model_config, train_module_config, dataset_config, data_loader_config,
        trainer_config)``.
    """
    from dataclasses import replace

    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import ComposableDataLoaderConfig

    tokenizer_config = getattr(TokenizerConfig, options.tokenizer)()
    # Qwen3 ties bos == eos, so BOS is dropped for document-boundary detection.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    model_config = build_model_config(options, tokenizer_config.padded_vocab_size())
    dataset_config = build_dataset_config(options, doc_tokenizer_config)
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        global_batch_size=options.global_batch_tokens,
        seed=options.seed,
        num_workers=4,
    )
    return (
        model_config,
        build_train_module_config(options),
        dataset_config,
        data_loader_config,
        build_trainer_config(options, save_folder=save_folder, work_dir=work_dir),
    )


def beaker_launch_config(options: TrainOptions, *, cmd: List[str], root_dir: str) -> Optional[Any]:
    """
    Build the Beaker launch config, or ``None`` for a local run.

    :param options: Resolved options.
    :param cmd: The command Beaker should run on the node.
    :param root_dir: Cluster root directory.

    :returns: A ``BeakerLaunchConfig``, or ``None``.
    """
    if options.is_local:
        return None
    from olmo_core.internal.common import build_launch_config
    from olmo_core.launch.beaker import BeakerEnvVar, OLMoCoreBeakerImage

    config = build_launch_config(
        name=options.run_name,
        cmd=cmd,
        cluster=options.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=options.nodes,
    )
    if config is not None:
        # Repo rule: every Beaker job launches urgent. Anything lower pends behind capacity.
        config.priority = "urgent"
        # Without context parallelism the full activation set is on one GPU; the default caching
        # allocator strands enough reserved-but-unallocated memory to OOM at 40960.
        config.env_vars.append(
            BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True")
        )
    return config
