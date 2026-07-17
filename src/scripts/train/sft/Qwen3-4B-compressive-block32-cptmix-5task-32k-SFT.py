"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B
**FAST-COMPRESSIVE-LANDMARK** CPT model (block-32 variant) on a THREE-WAY MIX of raw
continued-pretraining (CPT) text, the 5 long-context tasks (contradiction, nq, oolong, rerank,
outlier), and ``allenai/Dolci-Instruct-SFT`` (Qwen3-tokenized general instruction data).

This is the BLOCK-32 counterpart of ``Qwen3-4B-compressive-cptmix-5task-32k-SFT.py`` (block 64).
Differences, both forced by the smaller landmark block:

  1. Model: ``fast_compressive_landmark=True, mem_freq=31`` (block 32), with
     ``nonselected_landmark_mass=0.1``. Init from the block-32 compressive CPT base on weka
     (``qwen3-4b-compressive-landmark-block32/step2385``), NOT the block-64 or dense/plain-landmark
     base -- the landmark-token (151860) embedding + compressive grouped-softmax attention were
     trained during THIS block's CPT run. Loaded weights-only (load_optim_state=False).

  2. Data pipeline: same packed route as the block-64 script (``LandmarkPackingInstanceSourceConfig``
     over a ``MixingDocumentSourceConfig``) -- the compressive fused kernel supports cu_doc_lens
     (DOC_MASK) for arbitrary block sizes, so no no-pack fallback is needed here.

Mixing: top level is ``CPT_FRAC=0.85`` raw CPT text vs a 0.15 SFT budget. That SFT budget is split
75% the 5-task mix (internally weighted contra 2x / rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x)
/ 25% Dolci-Instruct-SFT -- the same 25/75 Dolci/5-task blend ratio introduced in
``Qwen3-4B-dense-5task-dolci25-32k-nocpt-SFT.py`` (amandab/sft-dolci branch), applied here on top of
the existing CPT mix rather than replacing it.

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-compressive-block32-cptmix-5task-32k-SFT.py \\
        dry_run q4b-comp-block32-cptmix-5task-32k ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-compressive-block32-cptmix-5task-32k-SFT.py \\
        launch  q4b-comp-block32-cptmix-5task-32k ai2/neptune --launch.num_nodes=2
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
# Landmark geometry (block-32 variant)
# ---------------------------------------------------------------------------
MEM_FREQ = 31
BLOCK_SIZE = MEM_FREQ + 1  # 32
SEQUENCE_LENGTH = (
    40960  # landmark-token-space window; divisible by BLOCK_SIZE; >= max ladder40k doc
)
CONTENT_LEN = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 39680 content tokens (pre-landmark)
LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token
NONSELECTED_LANDMARK_MASS = 0.1  # alpha for compressive attention

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
CPT_DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/" "dolma3_longmino_mix_sample15B_qwen"
)

# allenai/Dolci-Instruct-SFT, tokenized with the Qwen3 chat template (token_ids_part_*.npy +
# labels_mask_*.npy, EOS-separated). See src/scripts/data/convert_dolci_instruct_sft.py (on
# amandab/sft-dolci) for the converter. Note: the Tool Use subset of Dolci-Instruct-SFT is silently
# dropped by the converter (Qwen3 template ignores the 'environment' role) -- left as-is per prior
# user decision.
DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen3"

# Block-32 compressive-landmark CPT base (model+optim) on weka. Loaded weights-only
# (load_optim_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "qwen3-4b-compressive-landmark-block32/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions. CPT = 85%; the remaining 15% SFT budget is split 75% the 5-task mix (internally
# weighted contra 2x / rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x) / 25% Dolci-Instruct-SFT.
# (Realised CPT is lower -- see header caveat: no-pack skips long CPT docs.)
# ---------------------------------------------------------------------------
CPT_FRAC = 0.85
SFT_BUDGET = 1.0 - CPT_FRAC
FIVE_TASK_FRAC_OF_BUDGET = 0.75
DOLCI_FRAC_OF_BUDGET = 0.25
FIVE_TASK_FRAC = SFT_BUDGET * FIVE_TASK_FRAC_OF_BUDGET
DOLCI_FRAC = SFT_BUDGET * DOLCI_FRAC_OF_BUDGET

# Fractions WITHIN the 5-task group; these sum to 1.0 (the group is then nested at ratio
# FIVE_TASK_FRAC against CPT and Dolci below).
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())
NQ_FRAC = _W["nq"] / _WSUM
OOLONG_FRAC = _W["oolong"] / _WSUM
RERANK_FRAC = _W["rerank"] / _WSUM
OUTLIER_FRAC = _W["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, 1.0 - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

# ---------------------------------------------------------------------------
# Optimization / budget
# ---------------------------------------------------------------------------
LR = 1e-5
TARGET_STEPS = 1465
GLOBAL_BATCH_SIZE = (
    NUM_NODES * SEQUENCE_LENGTH
)  # one window per CP=8 DP replica/step (grad-accum 1)
TARGET_TOKENS = GLOBAL_BATCH_SIZE * TARGET_STEPS
MAX_STEPS = max(1, round(TARGET_TOKENS / GLOBAL_BATCH_SIZE))


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
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

    # Qwen3-4B with FAST COMPRESSIVE LANDMARK attention (no YaRN: landmark memory extends context).
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

    # ---- Three-way mixed document source: CPT text, 5-task group, Dolci-Instruct-SFT ----
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

    # Top-level blend: the whole 5-task mix (internally weighted per five_task_specs) at
    # FIVE_TASK_FRAC, Dolci-Instruct-SFT at DOLCI_FRAC, and raw CPT text at CPT_FRAC.
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
    if CPT_FRAC > 1e-6:
        specs.append(
            MixingDocumentSourceSpecConfig(
                source=cpt_doc_source,
                ratio=CPT_FRAC,
                max_repetition_factor=3.0,
                label="cpt_longmino",
            )
        )

    # PACKED with intra-document masking: block-aligned greedy packing + per-doc landmarks. The
    # compressive fused kernel supports cu_doc_lens (DOC_MASK) for this block size, so this uses the
    # SAME data path as the block-64 script -- efficient packing (no padding waste, no dropped docs
    # beyond >1-window) AND proper intra-doc masking (docs don't attend across each other within a
    # window).
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
        # Block-aligned doc boundaries come from LandmarkPackingInstanceSource (-> cu_doc_lens ->
        # DOC_MASK in the compressive kernel). EOS-derived boundaries would not be block-aligned.
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
