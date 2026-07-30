"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B compressive-landmark
CPT model with **GQA GROUP-MEAN BLOCK GATING** (``CompressiveGQAGroupedAttention``) on the same MIX of
5 long-context tasks as the plain compressive run (contradiction, nq, oolong, rerank, outlier). NO-CPT
variant (cpt_frac=0).

This is a *near-exact fork* of ``Qwen3-4B-compressive-5task-32k-nocpt-SFT.py`` -- SAME compressive CPT
base, SAME data + packing pipeline, SAME optimizer / steps / CP -- with ONE change: the attention is
``compressive_gqa_grouped=True`` instead of ``fast_compressive_landmark=True``. That makes each past
block's cross-block GATE (block rescaling) use the KV group's MEAN landmark score, i.e. a GQA group's
query heads share one block rescaling; the within-block softmax and the local section stay per-head.
Because the variant adds NO new parameters (it just averages the query for the gate dot product), it
initializes cleanly from the compressive CPT base, and the resulting model is directly comparable to
the plain compressive SFT run ``q4b-compressive-5task-32k-nocpt-fixdata`` (same recipe, grouped gate).

See ``analysis/group_landmark_selection/DESIGN_gqa_grouped_training.md`` and
:class:`~olmo_core.nn.attention.CompressiveGQAGroupedAttention` for the math.

**Kernel status (READ):** the fused grouped kernel's forward/backward is validated on CPU via the eager
path and via a GPU fused-vs-eager parity test (``landmark_gqa_grouped_test.py``); the DOC_MASK (packing)
branch reuses the validated compressive kernel's masking verbatim, so only the gate-from-``q_gate``
logic is new (covered by the doc_id=None parity test). This run was launched *before* the GPU parity
job finished (cluster-load constraint) -- if that job reports a mismatch, cancel this run.

    PYTHONPATH=src python src/scripts/train/sft/amanda-landmark/Qwen3-4B-compressive-gqa-grouped-5task-32k-nocpt-SFT.py \\
        dry_run q4b-compressive-gqa-grouped-5task-32k-nocpt ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft/amanda-landmark/Qwen3-4B-compressive-gqa-grouped-5task-32k-nocpt-SFT.py \\
        launch  q4b-compressive-gqa-grouped-5task-32k-nocpt ai2/jupiter --launch.num_nodes=2
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
# Landmark geometry (identical to the plain compressive run).
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
# NOTE: the group-mean gate averages over each KV group's n_rep=4 heads; CP=8 keeps whole KV groups on
# a rank (n_kv_heads=8 % cp=8), so the group average is computed after the head all-to-all as usual.
CP_DEGREE = 8
NUM_NODES = 2  # 2 nodes x 8 GPUs = 16 GPUs; cp_degree=8 -> NUM_NODES DP replicas (2 windows/step)

# ---------------------------------------------------------------------------
# Data (weka) -- single_task_ladders_v2, identical to the plain compressive run.
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
NQ_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"
CPT_DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen"
)

# Compressive-landmark CPT base (model+optim). Loaded weights-only (load_optim_state=False). The
# GQA-grouped variant has IDENTICAL parameters to plain compressive, so this base loads cleanly.
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-base-fast-compressive-landmark-8node/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions (identical to the plain compressive run).
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
# Optimization / budget (identical to the plain compressive run).
# ---------------------------------------------------------------------------
LR = 1e-5
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

    # Qwen3-4B with GQA GROUP-MEAN GATED compressive landmark attention (the ONLY change vs the plain
    # compressive run).
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        compressive_gqa_grouped=True,
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

    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        r = root.rstrip("/")
        return NumpyDocumentSourceConfig(
            source_paths=[f"{r}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{r}/labels_mask_*.npy"],
            expand_glob=True,
        )

    cpt = CPT_DATA_ROOT.rstrip("/")
    cpt_doc_source = NumpyDocumentSourceConfig(
        source_paths=[f"{cpt}/part-*.npy"],
        tokenizer=doc_tokenizer_config,
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
    if CPT_FRAC > 1e-6:
        specs.append(
            MixingDocumentSourceSpecConfig(
                source=cpt_doc_source,
                ratio=CPT_FRAC,
                max_repetition_factor=3.0,
                label="cpt_longmino",
            )
        )

    # PACKED with intra-document masking (same as the plain compressive run). The grouped fused kernel
    # handles cu_doc_lens (DOC_MASK) with the compressive kernel's masking code verbatim; only the gate
    # logit source (group-mean query) is new.
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
                entity="ai2-llm",  # NOT prasanns-* (that entity 403s for amandab launches)
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
