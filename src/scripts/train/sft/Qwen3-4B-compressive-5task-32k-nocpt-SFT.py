"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B
**FAST-COMPRESSIVE-LANDMARK** CPT model on a MIX of 5 long-context tasks (contradiction, nq, oolong,
rerank, outlier). NO-CPT variant: cpt_frac=0 (pure downstream FT, no continued-pretraining text).

This is the COMPRESSIVE counterpart of ``Qwen3-4B-fast-landmark-cptmix-5task-32k-SFT.py``. Two things
differ, both forced by the compressive attention kernel:

  1. Model: ``fast_compressive_landmark=True, mem_freq=63`` (block 64), with
     ``nonselected_landmark_mass=0.1`` (each past block's landmark token contributes its value as a
     compressed block summary; alpha reserves 10% of attention mass for the non-selected blocks).
     Init from the COMPRESSIVE CPT base on weka
     (``amandab/q4b-base-fast-compressive-landmark-8node/step2385``), NOT the dense/plain-landmark
     base -- the landmark-token (151860) embedding + compressive grouped-softmax attention were
     trained during compressive CPT. Loaded weights-only (load_optim_state=False).

  2. Data pipeline: ``MixingDocumentSource -> PadToLength -> Landmark insertion`` (the **no-pack**
     route), NOT ``LandmarkPackingInstanceSource``. The compressive fused Triton kernel raises
     ``NotImplementedError`` on ``cu_doc_lens`` (intra-document packing), which the landmark packing
     source emits. So we emit ONE document per sequence: ``PadToLengthInstanceSource`` right-pads
     each document to ``content_len`` (a multiple of ``mem_freq``), then ``LandmarkInstanceSource``
     inserts a landmark token every ``mem_freq`` positions through the WHOLE (padded) sequence, so
     the kernel's positional ``is_mem`` (``pos % block_size == block_size - 1``) holds and NO
     ``cu_doc_lens`` is emitted. ``generate_doc_lengths=False``.

CAVEATS (no-pack is inefficient -- inherent to the compressive kernel limitation):
  * Short SFT docs are right-padded to the full 40320-token content window (wasted compute on pad).
  * Docs longer than ``content_len`` (40320 content tokens) are SKIPPED by PadToLength -- this drops
    long CPT (dolma3longmino) docs in particular, so the REALISED CPT fraction is below the nominal
    0.85 (the MixingDocumentSource token-count log prints the realised mix). It also drops the very
    largest ladder40k SFT examples (max doc ~40407 tokens; only a handful exceed 40320).

Window choice (SEQUENCE_LENGTH=40960, NOT 32768): the 32k-rung SFT examples tokenize to up to ~40k
tokens (context + chat template + answer); a 32768 window would skip/split exactly those examples,
defeating a "32k" run. 40960 (divisible by block 64) is the validated dense-32k template's window and
is >= the max ladder40k doc length, so no 32k example is lost to the window. content_len = 40960 //
64 * 63 = 40320.

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-compressive-cptmix-5task-32k-SFT.py \\
        dry_run q4b-comp-cptmix-5task-32k ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-compressive-cptmix-5task-32k-SFT.py \\
        launch  q4b-comp-cptmix-5task-32k ai2/neptune --launch.num_nodes=2
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
# Data (weka) -- ladder40k (rungs up to 32k context; max doc ~40k tokens).
# ---------------------------------------------------------------------------
# Updated data: single_task_ladders_v2 (same 5-task shards as dense mix), tokenized to 40960 so the
# landmark-space window is fully fed. (Was cptmix_data_ladder40k.)
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
# nq: p10 pipeline (hard-neg ~10% + CE filter), NOT the 98%-hard v2/nq (standing directive).
NQ_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"
CPT_DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "dolma3_longmino_mix_sample15B_qwen"
)

# Compressive-landmark CPT base (model+optim) on weka. Loaded weights-only (load_optim_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-base-fast-compressive-landmark-8node/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions. CPT = 85%; the 15% SFT budget is split contra 2x / rerank 1.5x / outlier 1.5x /
# nq 1x / oolong 1x (sum 7). (Realised CPT is lower -- see header caveat: no-pack skips long CPT docs.)
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
TARGET_STEPS = 1465
GLOBAL_BATCH_SIZE = NUM_NODES * SEQUENCE_LENGTH  # NUM_NODES windows per step (CP=8 DP replicas); grad-accum 1
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

    # ---- N-way mixed document source: 5 SFT tasks + CPT ----
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
    if CPT_FRAC > 1e-6:
        specs.append(
            MixingDocumentSourceSpecConfig(
                source=cpt_doc_source, ratio=CPT_FRAC,
                max_repetition_factor=3.0, label="cpt_longmino",
            )
        )

    # PACKED with intra-document masking: block-aligned greedy packing + per-doc landmarks. The
    # compressive fused kernel now supports cu_doc_lens (DOC_MASK, commit ffb2c461), so compressive
    # uses the SAME data path as plain fast-landmark -- efficient packing AND proper intra-doc masking
    # (docs don't attend across each other). Makes compressive-vs-landmark apples-to-apples.
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


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
