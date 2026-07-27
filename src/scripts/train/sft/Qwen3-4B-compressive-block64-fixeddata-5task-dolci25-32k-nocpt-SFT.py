"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B
**FAST-COMPRESSIVE-LANDMARK** CPT model, block 64 (``mem_freq=63``), on 75% the 5-task mix
(contradiction, nq, oolong, rerank, outlier) / 25% ``allenai/Dolci-Instruct-SFT``, no CPT.

**This is the data-matched control for the gate-temperature arm.** It is a fork of
``Qwen3-4B-compressive-gate-temp-5task-dolci25-32k-nocpt-SFT.py`` with the *only* change being
``gate_temperature`` removed (and, with no new parameter to tolerate, strict weight loading
restored). Every data path, mixing ratio, base checkpoint, budget, and parallelism setting is
inherited verbatim from that script, so "gate temperature on vs. off" is the sole difference between
the two runs.

Why this exists rather than reusing ``Qwen3-4B-compressive-block64-5task-dolci25-32k-nocpt-SFT.py``:
that block-size-sweep script draws all five tasks from the older
``cptmix_data_ladder40k/{contradiction,nq,oolong,rerank,outlier}``, while the gate-temp run draws
from ``single_task_ladders_v2/*`` plus the p10-fixed ``single_task_ladders_p10/nq``. Comparing the
gate-temp run against that block-64 arm therefore confounds the feature with a whole-mix data
change (the nq difference alone is known to move eval numbers -- see the block-sweep NQ note). This
script removes that confound.

Model/geometry (identical to both parents): ``fast_compressive_landmark=True``, ``mem_freq=63``
(block 64), ``nonselected_landmark_mass=0.1``, initialized from the block-64 compressive CPT base
``amandab/q4b-base-fast-compressive-landmark-8node/step2385/model_and_optim`` (weights only,
``load_optim_state=False``).

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-compressive-block64-fixeddata-5task-dolci25-32k-nocpt-SFT.py \\
        dry_run q4b-comp-block64-fixeddata-5task-dolci25-32k
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-compressive-block64-fixeddata-5task-dolci25-32k-nocpt-SFT.py \\
        launch  q4b-comp-block64-fixeddata-5task-dolci25-32k ai2/jupiter --launch.num_nodes=2
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
SEQUENCE_LENGTH = 40960  # landmark-token-space window; divisible by BLOCK_SIZE; >= max ladder40k doc
CONTENT_LEN = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 40320 content tokens (pre-landmark)
LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token
NONSELECTED_LANDMARK_MASS = 0.1  # alpha for compressive attention

# Context parallel (Ulysses) degree. Qwen3-4B: n_heads=32, n_kv_heads=8 -> CP=8 splits both cleanly.
CP_DEGREE = 8
NUM_NODES = 2  # 2 nodes x 8 GPUs = 16 GPUs; cp_degree=8 -> NUM_NODES DP replicas (2 windows/step)

# ---------------------------------------------------------------------------
# Data (weka) -- ladder40k (rungs up to 32k context; max doc ~40k tokens). Same 5-task roots as the
# pure-5task baseline (including the p10-fixed nq); same 75/25 top-level blend against
# Dolci-Instruct-SFT as the block-size sweep scripts (see module docstring point 3).
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
# nq: p10 pipeline (hard-neg ~10% + CE filter), NOT the 98%-hard v2/nq (standing directive) -- NOTE
# this differs from the block-size sweep scripts, which use the older cptmix_data_ladder40k/nq.
NQ_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"

# allenai/Dolci-Instruct-SFT, tokenized with the Qwen3 chat template (token_ids_part_*.npy +
# labels_mask_*.npy, EOS-separated). Same source the block-size sweep scripts use. Note: the Tool Use
# subset of Dolci-Instruct-SFT is silently dropped by the converter (Qwen3 template ignores the
# 'environment' role) -- left as-is per prior user decision.
DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen3"

# Block-64 compressive-landmark CPT base (model+optim) on weka -- SAME base as the gate-temperature
# run and the block-size sweep's block-64 arm. Loaded weights-only (load_optim_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-base-fast-compressive-landmark-8node/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions WITHIN the 5-task group (sum to 1.0): contra 2x / rerank 1.5x / outlier 1.5x /
# nq 1x / oolong 1x -- same weighting as the pure-5task baseline and the block-size sweep.
# ---------------------------------------------------------------------------
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())
NQ_FRAC = _W["nq"] / _WSUM
OOLONG_FRAC = _W["oolong"] / _WSUM
RERANK_FRAC = _W["rerank"] / _WSUM
OUTLIER_FRAC = _W["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, 1.0 - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

# Top-level blend: 75% the 5-task mix (internally weighted per _W above), 25% Dolci-Instruct-SFT.
# No CPT source -- pure downstream FT (matches the block-size sweep).
FIVE_TASK_FRAC = 0.75
DOLCI_FRAC = 0.25

# ---------------------------------------------------------------------------
# Optimization / budget -- identical to the baseline compressive SFT run (token-matched comparison).
# ---------------------------------------------------------------------------
LR = 1e-5
# ~700M training tokens: 8550 steps x 2 DP windows x 40960 = 700M tokens (~39% of the 1.81B SFT mix),
# ~5.8h on jupiter H100 (~2.45s/step). Token-matched to the dense and baseline-compressive runs.
TARGET_STEPS = 8550
GLOBAL_BATCH_SIZE = NUM_NODES * SEQUENCE_LENGTH  # NUM_NODES windows per step (CP=8 DP replicas); grad-accum 1
TARGET_TOKENS = GLOBAL_BATCH_SIZE * TARGET_STEPS
MAX_STEPS = max(1, round(TARGET_TOKENS / GLOBAL_BATCH_SIZE))


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/amandab/{cli_context.run_name}"

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
    # No gate_temperature here -- that is the single difference from the gate-temp run this controls
    # for (see module docstring), so every parameter exists in the CPT base and loading stays strict.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_compressive_landmark=True,
        nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
        mem_freq=MEM_FREQ,
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

    # ---- Two-way mixed document source: 5-task group + Dolci-Instruct-SFT (no CPT) ----
    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        r = root.rstrip("/")
        return NumpyDocumentSourceConfig(
            source_paths=[f"{r}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{r}/labels_mask_*.npy"],
            expand_glob=True,
        )

    five_task_specs = [
        MixingDocumentSourceSpecConfig(
            source=_sft_source(CONTRA_DATA_ROOT), ratio=CONTRA_FRAC,
            max_repetition_factor=8.0, label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(NQ_DATA_ROOT), ratio=NQ_FRAC,
            max_repetition_factor=8.0, label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OOLONG_DATA_ROOT), ratio=OOLONG_FRAC,
            max_repetition_factor=8.0, label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(RERANK_DATA_ROOT), ratio=RERANK_FRAC,
            max_repetition_factor=8.0, label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OUTLIER_DATA_ROOT), ratio=OUTLIER_FRAC,
            max_repetition_factor=8.0, label="outlier",
        ),
    ]

    # Top-level blend: the whole 5-task mix (internally weighted per five_task_specs) at
    # FIVE_TASK_FRAC, Dolci-Instruct-SFT at DOLCI_FRAC. No CPT source.
    specs = [
        MixingDocumentSourceSpecConfig(
            source=MixingDocumentSourceConfig(source_specs=five_task_specs),
            ratio=FIVE_TASK_FRAC,
            label="five_task_mix",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(DOLCI_DATA_ROOT),
            ratio=DOLCI_FRAC,
            max_repetition_factor=8.0,
            label="dolci_instruct_sft",
        ),
    ]

    # PACKED with intra-document masking: block-aligned greedy packing + per-doc landmarks. The
    # compressive fused kernel supports cu_doc_lens (DOC_MASK), so this uses the SAME data path as
    # plain fast-landmark -- efficient packing AND proper intra-doc masking.
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
    main(config_builder=build_experiment_config)
