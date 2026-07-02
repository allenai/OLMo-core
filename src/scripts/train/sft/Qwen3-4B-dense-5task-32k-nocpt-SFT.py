"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B DENSE CPT model on
a MIX of 5 long-context tasks (contradiction, nq, oolong, rerank, outlier). NO-CPT variant: cpt_frac=0
(pure downstream FT, no continued-pretraining text mixed in).

Mirrors ``Qwen3-4B-dense-cptmix-5task-32k-SFT.py`` (dense, YaRN factor 2, flash-2) but: (a) drops the
CPT source (CPT_FRAC=0), and (b) corrects the base-checkpoint path to its real weka location under
``amandab/``. SEQUENCE_LENGTH=32768 (true 32k; power of 2 required by ``PackingInstanceSource``). The
real-data scan (17618 docs) shows only ~1% exceed 32768, so the vast majority pack whole; the few
over-length docs are DROPPED (LongDocStrategy.exclude) rather than truncated -- these are long-context
tasks whose answers sit at the end, so truncating would drop the answer and leave a prompt-only
(NaN-loss) window (measured 128 such windows at 32768 under truncate).

Packing: unlike the concat-and-chunk variant (which concatenates the whole mix and slices at fixed
SEQUENCE_LENGTH boundaries, splitting documents across windows and -- because qwen3's
bos==eos==151643 defeats the EOS+BOS boundary heuristic -- letting documents attend across each
other), this script uses ``PackingInstanceSource`` (Best-Fit-Decreasing bin-packing of whole
documents, padded to SEQUENCE_LENGTH) and a loader tokenizer with ``bos_token_id=None`` so the
EOS-based doc-length detection yields correct block-diagonal (varlen) masking.

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-dense-5task-32k-nocpt-SFT.py \\
        dry_run q4b-dense-5task-32k-nocpt ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-dense-5task-32k-nocpt-SFT.py \\
        launch  q4b-dense-5task-32k-nocpt ai2/neptune --launch.num_nodes=2
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LongDocStrategy,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
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
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# Packing requires a power-of-2 window (InstancePacker's SegmentTree). 32768 (true 32k) fits the vast
# majority of docs: per the real-data scan only 173/17618 (~1%) exceed it (worst task: outlier 3.2%).
# Those over-long docs are truncated to their first 32768 tokens (LongDocStrategy.truncate); see the
# all-masked-window count from sanity_check_packing.py at seq-len 32768 for the NaN-loss exposure.
SEQUENCE_LENGTH = 32768
CP_DEGREE = 8
NUM_NODES = 2  # 2 nodes x 8 GPUs = 16 GPUs; cp_degree=8 -> NUM_NODES DP replicas

# ---------------------------------------------------------------------------
# Data (weka) -- single_task_ladders_v2: the per-task length ladders (more data points than the
# original cptmix_data_ladder40k). Same 5 task subdirs and file layout
# (token_ids_part_*.npy + labels_mask_*.npy) as the singletask-ladder launchers.
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
# nq: use the p10 pipeline (hard-neg ~10% + CE filter), NOT the 98%-hard single_task_ladders_v2/nq
# (standing directive). Staged to weka from s3://.../single_task_ladders_p10/nq.
NQ_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"

# Dense CPT base (model+optim) on weka. Loaded weights-only (load_optim_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-dense-dolma3longmino/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions. NO CPT: the SFT budget is the whole mix. Base split is contra 2x / rerank 1.5x /
# outlier 1.5x / nq 1x / oolong 1x, but contra and oolong are upsampled to offset the docs dropped by
# LongDocStrategy.exclude at 32768 (single_task_ladders_v2 scan: contradiction ~18.3% of docs and
# oolong ~12.4% exceed 32768; other tasks lose <1%). Dropped docs are the LONGEST (32768-40955), so
# token-level loss is larger than the doc-count loss -- eyeballed at ~31% of contra tokens and ~23%
# of oolong tokens. Upsample by 1/(1 - token_loss): contra 2.0/~0.69=2.9, oolong 1.0/~0.77=1.3.
# ---------------------------------------------------------------------------
CPT_FRAC = 0.0
SFT_BUDGET = 1.0 - CPT_FRAC
_W = {"contra": 2.9, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.3}
_WSUM = sum(_W.values())
NQ_FRAC = SFT_BUDGET * _W["nq"] / _WSUM
OOLONG_FRAC = SFT_BUDGET * _W["oolong"] / _WSUM
RERANK_FRAC = SFT_BUDGET * _W["rerank"] / _WSUM
OUTLIER_FRAC = SFT_BUDGET * _W["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, SFT_BUDGET - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

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

    # Qwen3-4B with YaRN context extension (native 32k -> 64k), full flash-attn 2 attention.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
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

    # Best-Fit-Decreasing bin-packing of WHOLE documents into each window (no document is sliced
    # across a window boundary; leftover space is padded). Documents longer than SEQUENCE_LENGTH
    # (~1% of docs at 32768) are DROPPED (LongDocStrategy.exclude): truncating them would cut off the
    # end-of-sequence answer and leave a fully-masked, NaN-loss window (measured 128 such windows at
    # 32768 under truncate). Fragmenting would train on answer spans with only partial context.
    instance_source_config = PackingInstanceSourceConfig(
        sources=[MixingDocumentSourceConfig(source_specs=specs)],
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=doc_tokenizer_config,
        long_doc_strategy=LongDocStrategy.exclude,
    )

    # NOTE: the loader must use ``doc_tokenizer_config`` (bos_token_id=None). qwen3 has
    # bos_token_id == eos_token_id == 151643, and the EOS-based doc-length detection only marks a
    # boundary at an EOS *followed by* a BOS -- which never occurs in single-EOS-separated SFT data.
    # With bos=None it splits on every EOS, giving correct block-diagonal (varlen) masking and
    # isolating the padding tokens at the tail of each packed window.
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=doc_tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        generate_doc_lengths=True,  # block-diagonal (varlen) masking at EOS doc boundaries
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
                entity="prasanns-allen-institute-for-ai",  # prasann's entity (this run uses PRASANNS_WANDB_API_KEY); ai2-llm 404s the project for this key
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
