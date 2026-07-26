"""
Shared builder for the **MULTI-LANDMARK compressive** (block-64) 5-task 32k no-CPT SFT arms.

Four arms, one per finished CPT run of
``src/scripts/train/Qwen3/Qwen3-4B-base-multi-compressive-landmark-block64-{2,4}lm-{mean,max}-dolma3longmino.py``
(all four completed at ``step2385``, the same step count as the single-landmark block-64 CPT):

  * ``2lm-mean`` -> num_landmarks=2, landmark_gate_pool="mean", mem_freq=62
  * ``2lm-max``  -> num_landmarks=2, landmark_gate_pool="max",  mem_freq=62
  * ``4lm-mean`` -> num_landmarks=4, landmark_gate_pool="mean", mem_freq=60
  * ``4lm-max``  -> num_landmarks=4, landmark_gate_pool="max",  mem_freq=60

Each arm's ``num_landmarks`` / ``landmark_gate_pool`` / ``mem_freq`` are taken from ``_ARMS`` below and
applied to BOTH the model (:class:`~olmo_core.nn.attention.MultiCompressiveLandmarkAttention`) and the
data pipeline (:class:`~olmo_core.data.composable.LandmarkPackingInstanceSourceConfig`), so the SFT
landmark geometry is guaranteed to match the geometry the checkpoint was CPT'd with. Getting these out
of sync would silently mis-place the landmark tokens relative to the kernel's block tiling.

The recipe is otherwise an exact clone of
``Qwen3-4B-compressive-block64-5task-dolci25-32k-nocpt-SFT.py`` (the single-landmark block-64 arm of
the block-size sweep) so these four are directly comparable to the ``block64`` row in
``results/block_sweep_sft_5task.csv``:

  * Data: 75% the 5 long-context tasks (contra 2x / rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x
    from ``prasanns/cptmix_data_ladder40k``) + 25% ``allenai/Dolci-Instruct-SFT``. No raw CPT text.
  * Layout: ``LandmarkPackingInstanceSource`` (block-aligned greedy packing, per-document landmarks,
    intra-document ``cu_doc_lens`` -> ``DOC_MASK`` in the fused multi-landmark kernel).
  * Budget: LR 1e-5, TARGET_STEPS=8550 (~700M tokens at GLOBAL_BATCH_SIZE=NUM_NODES*40960),
    2 nodes x 8 GPUs, Ulysses CP degree 8.

.. note::
    ``SEQUENCE_LENGTH`` stays at 40960 (= 640 blocks of 64) so the window, batch size and token budget
    match the rest of the sweep. Because landmarks eat block slots, the *content* capacity per window
    falls as ``num_landmarks`` rises: 40320 tokens at 1 landmark, 39680 at 2, 38400 at 4. A handful of
    the longest ladder40k documents (~40k content tokens) will therefore be dropped by the packer on
    the 4-landmark arms; the drop fraction is logged. This is the same trade the existing block-size
    sweep already accepted -- its ``block16`` arm has a 38400-token content capacity, identical to the
    4-landmark arms here.
"""

from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

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
from olmo_core.internal.experiment import CliContext, ExperimentConfig
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
# Per-arm landmark geometry + the CPT checkpoint it must match. ``mem_freq`` is derived from
# BLOCK_SIZE so the two can never disagree: the fused kernel tiles by
# ``block_size = mem_freq + num_landmarks`` and requires a power of two.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 64
_CPT_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints"

# arm name -> (num_landmarks, landmark_gate_pool), verified against each CPT job's logged config.
_ARMS: Dict[str, Tuple[int, str]] = {
    "2lm-mean": (2, "mean"),
    "2lm-max": (2, "max"),
    "4lm-mean": (4, "mean"),
    "4lm-max": (4, "max"),
}

SEQUENCE_LENGTH = 40960  # landmark-token-space window; 640 blocks of BLOCK_SIZE
LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token
NONSELECTED_LANDMARK_MASS = 0.1  # alpha; eval/decode-time only, ignored during training

# Context parallel (Ulysses) degree. Qwen3-4B: n_heads=32, n_kv_heads=8 -> CP=8 splits both cleanly.
CP_DEGREE = 8
NUM_NODES = 2  # 2 nodes x 8 GPUs, cp_degree=8 -> NUM_NODES DP replicas

# ---------------------------------------------------------------------------
# Data (weka) -- ladder40k (rungs up to 32k context; max doc ~40k tokens).
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/cptmix_data_ladder40k"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
NQ_DATA_ROOT = f"{DATA_ROOT}/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"

# allenai/Dolci-Instruct-SFT, tokenized with the Qwen3 chat template (token_ids_part_*.npy +
# labels_mask_*.npy, EOS-separated). Note: the Tool Use subset is silently dropped by the converter
# (the Qwen3 template ignores the 'environment' role) -- left as-is per prior user decision.
DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen3"

# ---------------------------------------------------------------------------
# Mixing fractions WITHIN the 5-task group (this group is then given ratio=FIVE_TASK_FRAC against
# Dolci-Instruct-SFT below -- these fractions sum to 1.0). Same weighting as the block-size sweep:
# contra 2x / rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x.
# ---------------------------------------------------------------------------
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())
NQ_FRAC = _W["nq"] / _WSUM
OOLONG_FRAC = _W["oolong"] / _WSUM
RERANK_FRAC = _W["rerank"] / _WSUM
OUTLIER_FRAC = _W["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, 1.0 - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

# Top-level blend: 75% the 5-task mix (internally weighted per _W above), 25% Dolci-Instruct-SFT.
# No CPT source -- pure downstream FT.
FIVE_TASK_FRAC = 0.75
DOLCI_FRAC = 0.25

# ---------------------------------------------------------------------------
# Optimization / budget -- matched to the block-size sweep for token parity.
# ---------------------------------------------------------------------------
LR = 1e-5
TARGET_STEPS = 8550
GLOBAL_BATCH_SIZE = NUM_NODES * SEQUENCE_LENGTH  # NUM_NODES windows/step (CP=8), grad-accum 1
MAX_STEPS = TARGET_STEPS


def arm_geometry(arm: str) -> Dict[str, Any]:
    """
    Resolve an arm name to its full landmark geometry + CPT checkpoint.

    :param arm: One of ``"2lm-mean"``, ``"2lm-max"``, ``"4lm-mean"``, ``"4lm-max"``.

    :returns: ``num_landmarks``, ``landmark_gate_pool``, ``mem_freq``, ``content_capacity`` and
        ``base_checkpoint`` for the arm.

    :raises KeyError: If ``arm`` is not a known arm.
    """
    num_landmarks, landmark_gate_pool = _ARMS[arm]
    mem_freq = BLOCK_SIZE - num_landmarks
    return dict(
        num_landmarks=num_landmarks,
        landmark_gate_pool=landmark_gate_pool,
        mem_freq=mem_freq,
        # Content tokens that fit in one SEQUENCE_LENGTH window (documents needing more are dropped).
        content_capacity=SEQUENCE_LENGTH // BLOCK_SIZE * mem_freq,
        base_checkpoint=f"{_CPT_ROOT}/qwen3-4b-mcl-block64-{arm}/step2385/model_and_optim",
    )


def build_mcl_experiment(cli_context: CliContext, *, arm: str) -> ExperimentConfig:
    """
    Build the SFT config for one multi-landmark compressive arm.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :param arm: Which arm to build; see :func:`arm_geometry`.

    :returns: The full experiment config.
    """
    geom = arm_geometry(arm)
    num_landmarks = geom["num_landmarks"]
    landmark_gate_pool = geom["landmark_gate_pool"]
    mem_freq = geom["mem_freq"]

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
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3-4B with MULTI-LANDMARK COMPRESSIVE attention (no YaRN: landmark memory extends context).
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        multi_compressive_landmark=True,
        mem_freq=mem_freq,
        num_landmarks=num_landmarks,
        landmark_gate_pool=landmark_gate_pool,
        nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
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
    # ``num_landmarks`` here MUST match the model's, or the landmark tokens land in the wrong columns
    # of the kernel's block tiling.
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=MixingDocumentSourceConfig(source_specs=specs),
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=mem_freq,
        mem_id=LANDMARK_TOKEN_ID,
        num_landmarks=num_landmarks,
        pad_id=tokenizer_config.pad_token_id,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # Block-aligned doc boundaries come from LandmarkPackingInstanceSource (-> cu_doc_lens ->
        # DOC_MASK in the multi-landmark kernel). EOS-derived boundaries would not be block-aligned.
        generate_doc_lengths=False,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=geom["base_checkpoint"],
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
