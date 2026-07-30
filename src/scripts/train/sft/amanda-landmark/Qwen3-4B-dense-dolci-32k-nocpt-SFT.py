"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B DENSE CPT model,
with the SFT mix FULLY REPLACED by ``allenai/Dolci-Instruct-SFT`` (Qwen3-tokenized general instruction
data) instead of the 5 long-context tasks (contradiction, nq, oolong, rerank, outlier). NO-CPT variant:
cpt_frac=0 (pure downstream FT, no continued-pretraining text mixed in).

Mirrors ``Qwen3-4B-dense-5task-32k-nocpt-SFT.py`` (dense, YaRN factor 2, flash-2, same base checkpoint,
geometry, and packing strategy) but swaps the *entire* data source for the single Dolci-Instruct-SFT
dataset -- see ``Qwen3-4B-dense-5task-dolci50-32k-nocpt-SFT.py`` for a 50/50 blend of the two instead.
SEQUENCE_LENGTH=32768 (true 32k; power of 2 required by ``PackingInstanceSource``).

Packing: uses ``PackingInstanceSourceConfig`` (Best-Fit-Decreasing bin-packing of whole documents,
padded to SEQUENCE_LENGTH) and a loader tokenizer with ``bos_token_id=None`` so the EOS-based
doc-length detection yields correct block-diagonal (varlen) masking. NOTE: the doc-length
distribution of Dolci-Instruct-SFT at 32768 has not been scanned (see
``scan_doc_lengths.py``/``sanity_check_packing.py`` for the tool used to do this for the 5-task mix);
``LongDocStrategy.exclude`` is kept for parity with the 5-task script, but the drop rate here is
unmeasured.

    PYTHONPATH=src python src/scripts/train/sft/amanda-landmark/Qwen3-4B-dense-dolci-32k-nocpt-SFT.py \\
        dry_run q4b-dense-dolci-32k-nocpt ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft/amanda-landmark/Qwen3-4B-dense-dolci-32k-nocpt-SFT.py \\
        launch  q4b-dense-dolci-32k-nocpt ai2/neptune --launch.num_nodes=2
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LongDocStrategy,
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
# Packing requires a power-of-2 window (InstancePacker's SegmentTree).
SEQUENCE_LENGTH = 32768
CP_DEGREE = 8
NUM_NODES = 2  # 2 nodes x 8 GPUs = 16 GPUs; cp_degree=8 -> NUM_NODES DP replicas

# ---------------------------------------------------------------------------
# Data (weka) -- allenai/Dolci-Instruct-SFT, tokenized with the Qwen3 chat template
# (token_ids_part_*.npy + labels_mask_*.npy, EOS-separated). See
# src/scripts/data/convert_dolci_instruct_sft.py for the converter. Note: the Tool Use subset of
# Dolci-Instruct-SFT is silently dropped by the converter (Qwen3 template ignores the 'environment'
# role) -- left as-is per prior user decision.
# ---------------------------------------------------------------------------
DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen3"

# Dense CPT base (model+optim) on weka. Loaded weights-only (load_optim_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-dense-dolma3longmino/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Optimization / budget
# ---------------------------------------------------------------------------
LR = 1e-5
# Kept identical to Qwen3-4B-dense-5task-32k-nocpt-SFT.py for token-parity comparability across
# variants: 10700 steps x 2 DP windows x 32768 = 701M tokens (~4.2h on jupiter H100, ~1.4s/step).
# Note this is measured against the 1.81B-token 5-task mix (~39%); Dolci-Instruct-SFT is a
# differently-sized corpus (~1.21B tokens per the converter run), so the equivalent fraction differs.
TARGET_STEPS = 10700
GLOBAL_BATCH_SIZE = (
    NUM_NODES * SEQUENCE_LENGTH
)  # NUM_NODES windows per step (CP=8 DP replicas); grad-accum 1
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

    # ---- Single document source: Dolci-Instruct-SFT (replaces the 5-task mix entirely) ----
    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        r = root.rstrip("/")
        return NumpyDocumentSourceConfig(
            source_paths=[f"{r}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{r}/labels_mask_*.npy"],
            expand_glob=True,
        )

    # Best-Fit-Decreasing bin-packing of WHOLE documents into each window (no document is sliced
    # across a window boundary; leftover space is padded). Documents longer than SEQUENCE_LENGTH are
    # DROPPED (LongDocStrategy.exclude), kept for parity with the 5-task script; see module docstring.
    instance_source_config = PackingInstanceSourceConfig(
        sources=[_sft_source(DOLCI_DATA_ROOT)],
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
                entity="ai2-llm",  # prasanns-allen-institute-for-ai 403s (amandab's Beaker-injected WANDB_API_KEY lacks write access there)
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
