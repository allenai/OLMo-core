"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B
**FAST-COMPRESSIVE-LANDMARK + BLOCK-LOCAL RoPE** CPT model on a MIX of 5 long-context tasks
(contradiction, nq, oolong, rerank, outlier). NO-CPT variant: cpt_frac=0 (pure downstream FT).

This is identical to ``Qwen3-4B-compressive-5task-32k-nocpt-SFT.py`` except for the RoPE variant and
base checkpoint, mirroring what ``Qwen3-4B-base-fast-compressive-landmark-blocklocalrope-dolma3longmino.py``
did to the plain compressive CPT script:

  1. RoPE: positions reset at every landmark block boundary (``RoPEType.block_local``, ``block_size=64``)
     instead of the standard global-position rotation -- token ``t`` gets RoPE position ``t % BLOCK_SIZE``.
     This is a stock RoPE variant; the fast/compressive landmark kernels only see Q/K *after* RoPE, so it
     is transparent to them (see the CPT script for the full explanation).

  2. Base checkpoint: the BLOCK-LOCAL-RoPE compressive CPT run on the 15B-token dolma3_longmino sample
     (``qwen3-4b-fclm-blocklocal-d3lm15b/step2385``, 10B tokens, final checkpoint), NOT the plain
     compressive base. Loaded weights-only (load_optim_state=False).

Everything else (data pipeline, packing, mixing fractions, geometry, optimizer/schedule) is unchanged
from ``Qwen3-4B-compressive-5task-32k-nocpt-SFT.py`` -- see that script's docstring for the no-pack ->
packed-with-cu_doc_lens data-pipeline notes and the SEQUENCE_LENGTH=40960 window justification.

    PYTHONPATH=src python src/scripts/train/sft/amanda-landmark/Qwen3-4B-compressive-blocklocal-5task-32k-nocpt-SFT.py \\
        dry_run q4b-comp-blocklocal-5task-32k-nocpt ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft/amanda-landmark/Qwen3-4B-compressive-blocklocal-5task-32k-nocpt-SFT.py \\
        launch  q4b-comp-blocklocal-5task-32k-nocpt ai2/neptune --launch.num_nodes=2
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.rope import RoPEType
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
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

# ---------------------------------------------------------------------------
# Landmark geometry
# ---------------------------------------------------------------------------
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = (
    40960  # landmark-token-space window; divisible by BLOCK_SIZE; >= max ladder40k doc
)
CONTENT_LEN = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 40320 content tokens (pre-landmark)
LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token
NONSELECTED_LANDMARK_MASS = 0.1  # alpha for compressive attention

# Context parallel (Ulysses) degree. Qwen3-4B: n_heads=32, n_kv_heads=8 -> CP=8 splits both cleanly.
CP_DEGREE = 8
NUM_NODES = 2  # 2 nodes x 8 GPUs = 16 GPUs; cp_degree=8 -> NUM_NODES DP replicas (2 windows/step)

# ---------------------------------------------------------------------------
# Data (weka) -- ladder40k (rungs up to 32k context; max doc ~40k tokens).
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
# nq: p10 pipeline (hard-neg ~10% + CE filter), NOT the 98%-hard v2/nq (standing directive).
NQ_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"

# BLOCK-LOCAL-RoPE compressive-landmark CPT base (model+optim) on weka, from
# Qwen3-4B-base-fast-compressive-landmark-blocklocalrope-dolma3longmino.py (10B tokens, final
# checkpoint). Loaded weights-only (load_optim_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/qwen3-4b-fclm-blocklocal-d3lm15b/"
    "step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions. NO CPT: the SFT budget is the whole mix, split contra 2x / rerank 1.5x /
# outlier 1.5x / nq 1x / oolong 1x (sum 7).
# ---------------------------------------------------------------------------
CPT_FRAC = 0.0  # NO-CPT variant: pure downstream FT on the 5 SFT tasks only
SFT_BUDGET = 1.0 - CPT_FRAC
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())
NQ_FRAC = SFT_BUDGET * _W["nq"] / _WSUM
OOLONG_FRAC = SFT_BUDGET * _W["oolong"] / _WSUM
RERANK_FRAC = SFT_BUDGET * _W["rerank"] / _WSUM
OUTLIER_FRAC = SFT_BUDGET * _W["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, 1.0 - CPT_FRAC - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

# ---------------------------------------------------------------------------
# Optimization / budget
# ---------------------------------------------------------------------------
LR = 1e-5
# ~700M training tokens: 8550 steps x 2 DP windows x 40960 = 700M tokens (~39% of the 1.81B SFT mix).
# Token-matched to the plain-compressive and dense runs.
TARGET_STEPS = 8550
GLOBAL_BATCH_SIZE = NUM_NODES * SEQUENCE_LENGTH  # NUM_NODES windows per step (CP=8 DP replicas)
TARGET_TOKENS = GLOBAL_BATCH_SIZE * TARGET_STEPS
MAX_STEPS = max(1, round(TARGET_TOKENS / GLOBAL_BATCH_SIZE))


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
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3-4B with FAST COMPRESSIVE LANDMARK attention (no YaRN: landmark memory extends context).
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_compressive_landmark=True,
        nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
        mem_freq=MEM_FREQ,
    )

    # Switch to block-local RoPE, matching the base CPT checkpoint: positions reset every BLOCK_SIZE
    # tokens (RoPEType.block_local) instead of the standard global-position rotation. See
    # Qwen3-4B-base-fast-compressive-landmark-blocklocalrope-dolma3longmino.py for the full rationale.
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert isinstance(model_config.block.sequence_mixer, AttentionConfig)
    assert model_config.block.sequence_mixer.rope is not None
    model_config.block.sequence_mixer.rope.name = RoPEType.block_local
    model_config.block.sequence_mixer.rope.block_size = BLOCK_SIZE

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
            shard_degree=1,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- N-way mixed document source: 5 SFT tasks (no CPT) ----
    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        r = root.rstrip("/")
        return NumpyDocumentSourceConfig(
            source_paths=[f"{r}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{r}/labels_mask_*.npy"],
            expand_glob=True,
        )

    specs = [
        MixingDocumentSourceSpecConfig(
            source=_sft_source(CONTRA_DATA_ROOT),
            ratio=CONTRA_FRAC,
            max_repetition_factor=8.0,
            label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(NQ_DATA_ROOT),
            ratio=NQ_FRAC,
            max_repetition_factor=8.0,
            label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OOLONG_DATA_ROOT),
            ratio=OOLONG_FRAC,
            max_repetition_factor=8.0,
            label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(RERANK_DATA_ROOT),
            ratio=RERANK_FRAC,
            max_repetition_factor=8.0,
            label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OUTLIER_DATA_ROOT),
            ratio=OUTLIER_FRAC,
            max_repetition_factor=8.0,
            label="outlier",
        ),
    ]

    # PACKED with intra-document masking: block-aligned greedy packing + per-doc landmarks (the fused
    # compressive kernel supports cu_doc_lens / DOC_MASK, commit ffb2c461).
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=MixingDocumentSourceConfig(source_specs=specs),
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
        # Block-aligned doc boundaries from LandmarkPackingInstanceSource (-> cu_doc_lens -> DOC_MASK).
        generate_doc_lengths=False,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=BASE_CHECKPOINT,
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
                entity="ai2-llm",  # matches the base CPT run's entity (prasanns-... 403s for amandab's launches)
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
    main(config_builder=build_experiment_config)
