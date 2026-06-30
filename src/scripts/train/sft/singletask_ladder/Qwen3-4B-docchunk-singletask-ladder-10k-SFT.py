"""
**SINGLE-TASK** document-chunked **DENSE** (``DocumentChunkedAttention``, ``cross_doc_mode="chunked"``)
Beaker/gantry SFT of Qwen3-4B on a **50% (~10k) subsample** of one task's ladder, **NO CPT mix**.

This is the doc-chunked-dense row of the overnight 5-task x 4-variant 10k matrix. It mirrors
``Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py`` (single SFT source, ratio 1.0, seeded 50%
document subsample) but swaps in the document-chunked attention + data layout from
``_docchunk_5task_32k_nocpt_common.py``:

  * model: ``document_chunked=True, cross_doc_mode="chunked"`` + YaRN(factor 2) (dense base).
  * layout: ``PadToLength`` (ONE already-chunked example per 40960 window, padded) -- doc-chunked
    attention reconstructs per-token ``chunk_id`` roles from the ``<|box_start|>``/``<|box_end|>``
    boundary tokens, so it CANNOT use ConcatAndChunk packing or context parallelism.
  * single 8xH200 node, FSDP-sharded, no CP, FFN-only activation checkpointing, ``compile_model=False``.

The task is parsed from the run name (one of contra | nq | oolong | rerank | outlier), exactly like
the 3-variant launcher (the framework only does scalar dotlist overrides, no structural CLI plumbing).

!!! DATA REQUIREMENT (see the feasibility flag in the overnight plan) !!!
DocumentChunkedAttention is a strict superset of plain causal attention: with NO ``<|box_start|>``/
``<|box_end|>`` markers in the tokens it degenerates to ordinary causal attention (every token is a
FREE token, so the chunked mask is a no-op). The PLAIN single-task ladder shards under
``prasanns/single_task_ladders/<task>/`` contain **zero** box markers (verified) -- so this launcher
MUST read **box-marker** tokenization produced by ``convert_unified_to_document_landmark.py
--emit dense`` (the same converter behind the 5-task ``cptmix_docchunk_ladder40k/<task>_dense`` shards),
NOT the plain ladder shards. ``DOCCHUNK_DATA_ROOT`` points at that box-marker root; it defaults to the
existing per-task 5-task doc-chunked root but SHOULD be re-pointed at a freshly-regenerated single-task
box-marker root that matches the LATEST natural data (incl. CE-graded rerank). Confirm the root before
launch.

    PYTHONPATH=src python src/scripts/train/sft/singletask_ladder/Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py \\
        dry_run q4b-docchunk_dense-contra-ladder32k-10k ai2/neptune
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
    SamplingDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
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

# ---- Geometry / reserved ids (match the converter + document_chunk_landmark defaults). ----
SEQUENCE_LENGTH = 40960
MEM_FREQ = 63
NUM_NODES = 1  # single 8xH200 node; doc-chunked attention has no CP support.
EOS_TOKEN_ID = 151643
DOC_START_ID = 151648  # <|box_start|>
DOC_END_ID = 151649  # <|box_end|>

EPOCHS = 2
LR = 2e-5  # overnight 10k matrix LR (matches the 3-variant launcher).

# ---- 50% subsample (whole-document seeded sampling), identical to the 3-variant launcher. ----
SUBSAMPLE_FACTOR = 0.5
SUBSAMPLE_SEED = 7411

# Box-marker (doc-chunked) per-task data root. SEE THE DATA REQUIREMENT in the module docstring.
# Default: the freshly-regenerated single-task box-marker root (``--emit dense``, ``--cot-mode none``,
# seq-len 40960) matching the LATEST natural data -- contra/nq/oolong/outlier from the 20k ladder
# sources and rerank from the CE-graded ce_gen source -- built by
# ``src/scripts/data/convert_docchunk_singletask_v2_local.sbatch`` and uploaded to weka/s3. Each task
# lives at ``<root>/<task>_dense/`` with balanced ``<|box_start|>``/``<|box_end|>`` markers (verified).
# Override via env DOCCHUNK_DATA_ROOT if needed.
DOCCHUNK_DATA_ROOT = os.environ.get(
    "DOCCHUNK_DATA_ROOT",
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_docchunk_v2",
)
# dense doc-chunked CPT base (same step2385 checkpoint as the packed dense row).
DENSE_BASE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-dense-dolma3longmino/step2385/model_and_optim"
)

# Per-task box-marker shard subdir (``<task>_dense``) under DOCCHUNK_DATA_ROOT, and the loss label.
_TASK_DIR = {"contra": "contra", "nq": "nq", "oolong": "oolong",
             "rerank": "rerank", "outlier": "outlier"}
_TASK_LABEL = {"contra": "contradiction", "nq": "nq_retrieval", "oolong": "oolong",
               "rerank": "rerank", "outlier": "outlier"}

WORLD_SIZE = NUM_NODES * 8
GLOBAL_BATCH_SIZE = WORLD_SIZE * SEQUENCE_LENGTH  # 8 * 40960 -> 8 instances/step


def _task_from_run_name(run_name: str) -> str:
    for key in _TASK_DIR:
        if key in run_name:
            return key
    raise SystemExit(
        f"run name {run_name!r} must contain one of {sorted(_TASK_DIR)} to pick the task."
    )


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    task = _task_from_run_name(cli_context.run_name)
    task_root = f"{DOCCHUNK_DATA_ROOT}/{_TASK_DIR[task]}_dense"
    base_checkpoint = DENSE_BASE
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
        # Doc-chunked attention has no CP, so the full 40960-token activation set lives on ONE GPU and
        # FlexAttention needs ~12 GiB on top -> expandable_segments reclaims fragmented reserved mem.
        # Requires an 80 GiB+ GPU cluster (e.g. jupiter H100); 44 GiB nodes are too small.
        beaker_launch_config.env_vars.append(
            BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True")
        )

    tokenizer_config = TokenizerConfig.qwen3()
    # EOS-separated instances; qwen3 ties bos==eos, so drop BOS for document-boundary detection.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model: document-chunked DENSE attention + YaRN (native 32k -> 64k). ----
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
            lr=LR, weight_decay=0.0, betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        # The chunked mask is eager (@torch.compiler.disable); keep compile off.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full, shard_degree=WORLD_SIZE,
        ),
        # FFN-only AC (NOT full): full-block AC recomputes the FlexAttention block-mask build, which is
        # not recompute-stable on torch 2.9 (-> CheckpointError). FFN-only keeps attention out of the
        # recompute and still fits 40960 on H200/H100-80G.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Single SFT source, ratio 1.0, NO CPT, wrapped in a seeded 50% document subsample. ----
    task_source = NumpyDocumentSourceConfig(
        source_paths=[f"{task_root}/token_ids_part_*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{task_root}/labels_mask_*.npy"],
        expand_glob=True,
    )
    subsampled_source = SamplingDocumentSourceConfig(
        sources=[task_source], factor=SUBSAMPLE_FACTOR, seed=SUBSAMPLE_SEED,
        label=_TASK_LABEL[task],
    )
    mixing = MixingDocumentSourceConfig(source_specs=[
        MixingDocumentSourceSpecConfig(
            source=subsampled_source, ratio=1.0, max_repetition_factor=8.0, label=_TASK_LABEL[task],
        )
    ])
    # PadToLength: one already-chunked example per 40960 window (chunk roles from box markers).
    instance_source_config = PadToLengthInstanceSourceConfig(
        sources=[mixing], sequence_length=SEQUENCE_LENGTH, tokenizer=doc_tokenizer_config,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config, work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE, seed=34521, num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir, save_overwrite=True, load_path=base_checkpoint,
            load_strategy=LoadStrategy.always, load_trainer_state=False, load_optim_state=False,
            metrics_collect_interval=10, cancel_check_interval=10,
            max_duration=Duration.epochs(EPOCHS),
        )
        .with_callback("checkpointer", CheckpointerCallback(
            save_interval=100000, ephemeral_save_interval=500, max_checkpoints=2, save_async=True))
        .with_callback("wandb", WandBCallback(
            name=run_name_with_ts, group=cli_context.run_name,
            entity="prasanns-allen-institute-for-ai", project="memory-networks",
            enabled=True, cancel_check_interval=10))
        .with_callback("slack_notifier", SlackNotifierCallback(name=run_name_with_ts, enabled=False))
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
