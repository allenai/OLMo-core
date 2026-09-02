"""
32k, context-parallel (Ulysses 8) Beaker/gantry **SINGLE-TASK** length-ladder SFT of Qwen3-4B,
**variant-aware** (dense | landmark | compressive | sparselandmark), on a subsample of one task's
20k-example ladder (default 50%, ~10k). **NO CPT mix** (one SFT source at ratio 1.0).

Generalises ``Qwen3-4B-compressive-singletask-ladder-32k-SFT.py`` to all three attention variants and
adds a seeded 50% document subsample. Both the **task** and the **variant** are parsed from the run
name (so no structural CLI plumbing through the internal framework, which only does scalar dotlist
overrides). The run name MUST contain:
  * exactly one task keyword:  contra | nq | oolong | rerank | outlier
  * exactly one variant keyword: dense | landmark | compressive | sparselandmark
    (``sparselandmark`` contains ``landmark``; the parser prefers the longest match.)

Per-variant model + packing + weka CPT base (all but sparselandmark under ``amandab/`` -- weka-only):
  * dense       -> flash-attn 2 + YaRN(factor2) + ConcatAndChunk packing (varlen EOS masking).
                   base: q4b-dense-dolma3longmino/step2385/model_and_optim
  * landmark    -> fast_landmark (mem_freq=63) + LandmarkPacking, no YaRN.
                   base: q4b-fast-landmark-dolma3longmino/step2385/model_and_optim
  * compressive -> fast_compressive_landmark (mem_freq=63, nonselected_landmark_mass=0.1)
                   + LandmarkPacking, no YaRN.
                   base: q4b-base-fast-compressive-landmark-8node/step2385/model_and_optim
  * sparselandmark -> sparse_landmark (mem_freq=63, num_landmarks=1) + LandmarkPacking, no YaRN.
                   Attends fully within its own chunk; past chunks are visible ONLY through their
                   single landmark token (no expansion into past-chunk content).
                   base: q4b-sparse-landmark-dolma3longmino/step2385/model_and_optim
                   NOTE: no context parallelism (see the parallelism branch below).

Data (weka): per-task ladder shards under
``prasanns/single_task_ladders/<task>/{token_ids_part_*.npy,labels_mask_*.npy}``.

50% subsample: each task source is wrapped in a seeded ``SamplingDocumentSource`` with
``factor=SUBSAMPLE_FACTOR`` (=0.5), which deterministically samples whole documents down to ~50% of
the source's tokens (~10k of the 20k examples). Edit ``SUBSAMPLE_FACTOR`` below to change the fraction.

    PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py \\
        dry_run q4b-dense-contra-ladder32k-10k ai2/neptune
    PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py \\
        launch  q4b-dense-contra-ladder32k-10k ai2/neptune
"""

import os
from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    LandmarkPackingInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
    SamplingDocumentSourceConfig,
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
from olmo_core.nn.attention import AttentionBackendName
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
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---- Landmark / window geometry (matches the cptmix/single-task 32k launchers) ----
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 40960
LANDMARK_TOKEN_ID = 151860
NONSELECTED_LANDMARK_MASS = 0.1
CP_DEGREE = 8
NUM_NODES = 1
EPOCHS = 1

# ---- 50% subsample: whole-document seeded sampling down to this fraction of source tokens ----
SUBSAMPLE_FACTOR = float(os.environ.get("STL_SUBSAMPLE", "0.5"))
SUBSAMPLE_SEED = 7411

# weka per-task ladder data root + per-variant weka CPT bases (all under amandab/).
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
_AMANDAB = "/weka/oe-training-default/ai2-llm/checkpoints/amandab"
BASE_CHECKPOINTS = {
    "dense": f"{_AMANDAB}/q4b-base-dense-lr1.1e-4/step2385/model_and_optim",
    "landmark": f"{_AMANDAB}/q4b-base-fast-landmark-lr1p1e-4/step2385/model_and_optim",
    "compressive": f"{_AMANDAB}/qwen4b-base-compressive-lr1.1e-4/step2385/model_and_optim",
    # Verified on weka 2026-07-21 (step1000/step2000/step2385, each with config.json +
    # model_and_optim). NOTE: the four sft_longctx/Qwen3-4B-sparse-landmark-*.py launchers point at
    # ``checkpoints/q4b-sparse-landmark-dolma3longmino/...`` WITHOUT the ``amandab/`` segment, which
    # does not exist -- loading it dies with "No checkpoint found in save folder or load path".
    # The alternative base ``amandab/q4b-base-sparse-landmark-8node/step2385`` also exists.
    "sparselandmark": f"{_AMANDAB}/q4b-sparse-landmark-dolma3longmino/step2385/model_and_optim",
}

_TASK_DIR = {
    "contra": "contradiction",
    "nq": "nq",
    "oolong": "oolong",
    "rerank": "rerank",
    "outlier": "outlier",
}
_TASK_LABEL = {
    "contra": "contradiction",
    "nq": "nq_retrieval",
    "oolong": "oolong",
    "rerank": "rerank",
    "outlier": "outlier",
}
_VARIANTS = ("dense", "landmark", "compressive", "sparselandmark")

LR = 2e-5  # overnight 10k matrix: bumped from 1e-5 (coordinator request 2026-06-30).
# Global batch = 8 sequence-windows per optimizer step. With CP=8 (DP=1) this is pure GRADIENT
# ACCUMULATION: rank_microbatch_size stays = SEQUENCE_LENGTH (one window, no extra memory), and the
# trainer runs GLOBAL_BATCH_SIZE/(rank_microbatch*dp)=8 accumulation microbatches per step.
GLOBAL_BATCH_WINDOWS = 8
GLOBAL_BATCH_SIZE = NUM_NODES * SEQUENCE_LENGTH * GLOBAL_BATCH_WINDOWS


def _task_from_run_name(run_name: str) -> str:
    for key in _TASK_DIR:
        if key in run_name:
            return key
    raise SystemExit(
        f"run name {run_name!r} must contain one of {sorted(_TASK_DIR)} to pick the task."
    )


def _variant_from_run_name(run_name: str) -> str:
    found = [v for v in _VARIANTS if v in run_name]
    # "sparselandmark" contains "landmark", so a plain substring scan matches both for a legitimate
    # sparse-landmark run name. Drop any match that is a substring of a longer match; the
    # exactly-one check below still catches genuinely ambiguous names (e.g. "dense" + "landmark").
    found = [v for v in found if not any(v != other and v in other for other in found)]
    if len(found) != 1:
        raise SystemExit(
            f"run name {run_name!r} must contain exactly one variant of {list(_VARIANTS)} "
            f"(found {found})."
        )
    return found[0]


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    task = _task_from_run_name(cli_context.run_name)
    variant = _variant_from_run_name(cli_context.run_name)
    task_root = f"{DATA_ROOT}/{_TASK_DIR[task]}"
    base_checkpoint = BASE_CHECKPOINTS[variant]
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
        # Ship the (uncommitted) launcher via an ephemeral ref -- no permanent commit needed.
        beaker_launch_config.allow_dirty = True
        # The Beaker job RE-BUILDS this config on the node, so propagate the subsample fraction
        # (resolved from the launch-host env here) or the on-node rebuild would silently fall back to
        # the module default (0.5) -- breaking the data-scaling sweep. Follows the DOCCHUNK_* pattern
        # in _docchunk_5task_32k_nocpt_common.py.
        beaker_launch_config.env_vars.append(
            BeakerEnvVar(name="STL_SUBSAMPLE", value=repr(SUBSAMPLE_FACTOR))
        )

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model (per variant) ----
    if variant == "sparselandmark":
        # SPARSE landmark (``AttentionType.sparse_landmark``): a query attends FULLY within its own
        # chunk, but sees PAST chunks only through their single landmark token -- the non-landmark
        # content of past chunks is never attended (no expansion). Distinct from ``landmark``
        # (fast_landmark), which gates blocks by their landmark and then attends the selected
        # blocks' full content.
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            sparse_landmark=True,
            mem_freq=MEM_FREQ,
            num_landmarks=1,
        )
    elif variant == "compressive":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            fast_compressive_landmark=True,
            nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
            mem_freq=MEM_FREQ,
        )
    elif variant == "landmark":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            fast_landmark=True,
            mem_freq=MEM_FREQ,
        )
    else:  # dense -- flash-attn 2 + YaRN context extension (native 32k -> 64k)
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_2,
        ).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
        )

    # ---- Parallelism / memory (per variant) ----
    # ``SparseLandmarkAttention.apply_cp`` raises NotImplementedError, so the sparse-landmark arm
    # cannot use Ulysses CP. Each rank then holds a FULL SEQUENCE_LENGTH activation set, which needs
    # 8-way FSDP sharding + FULL activation checkpointing to fit -- both proven by the sparse-landmark
    # CPT and longctx-SFT launchers, where shard_degree=1 and budget-mode AC each OOM'd. Dropping CP
    # turns DP=1 into DP=8, and with rank_microbatch_size = SEQUENCE_LENGTH that yields the SAME
    # 8-window global batch as the CP=8 variants (8 ranks x 1 window instead of 1 rank x 8 grad-accum
    # microbatches), so the optimizer-step count matches and the data-scaling curves stay comparable.
    if variant == "sparselandmark":
        cp_config = None
        shard_degree = 8
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        )
        # Without sequence parallelism the 40960 x padded-vocab float logit tensor dominates memory.
        model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    else:
        cp_config = TransformerContextParallelConfig.ulysses(degree=CP_DEGREE)
        shard_degree = 1
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        )

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
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=shard_degree,
        ),
        cp_config=cp_config,
        ac_config=ac_config,
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Single SFT source, ratio 1.0, NO CPT, wrapped in a seeded 50% document subsample ----
    task_source = NumpyDocumentSourceConfig(
        source_paths=[f"{task_root}/token_ids_part_*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{task_root}/labels_mask_*.npy"],
        expand_glob=True,
    )
    subsampled_source = SamplingDocumentSourceConfig(
        sources=[task_source],
        factor=SUBSAMPLE_FACTOR,
        seed=SUBSAMPLE_SEED,
        label=_TASK_LABEL[task],
    )
    mixing = MixingDocumentSourceConfig(
        source_specs=[
            MixingDocumentSourceSpecConfig(
                source=subsampled_source,
                ratio=1.0,
                max_repetition_factor=8.0,
                label=_TASK_LABEL[task],
            )
        ]
    )

    if variant == "dense":
        instance_source_config = ConcatAndChunkInstanceSourceConfig(
            sources=[mixing],
            sequence_length=SEQUENCE_LENGTH,
        )
        generate_doc_lengths = True  # block-diagonal (varlen) masking at EOS doc boundaries
    else:  # landmark + compressive both use LandmarkPacking
        instance_source_config = LandmarkPackingInstanceSourceConfig(
            source=mixing,
            sequence_length=SEQUENCE_LENGTH,
            mem_freq=MEM_FREQ,
            mem_id=LANDMARK_TOKEN_ID,
            pad_id=tokenizer_config.pad_token_id,
        )
        generate_doc_lengths = False

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        generate_doc_lengths=generate_doc_lengths,
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
