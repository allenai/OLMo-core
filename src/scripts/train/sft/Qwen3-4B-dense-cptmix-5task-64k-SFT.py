"""
32k-context, context-parallel (Ulysses degree 8), multi-node Beaker/gantry SFT of the Qwen3-4B
DENSE CPT model on a MIX of 5 long-context tasks (contradiction, nq, oolong, rerank, outlier) +
raw continued-pretraining (CPT) text.

This is the weka/Beaker scale-up of the recipe validated LOCALLY at 8k by
``Qwen3-4B-dense-cptmix-contra-local.py``: cpt0.85 + a weighted 5-task SFT budget at lr1e-5, which
preserved RULER (~0.799) while all 5 tasks learned. It reuses the 32k / CP / multinode / gantry
skeleton of ``Qwen3-4B-dense-longctx-base32k-SFT.py`` but swaps the SINGLE-task PadToLength data
pipeline for the local launcher's N-way ``MixingDocumentSource`` + ``ConcatAndChunk`` (packing)
pipeline.

Mixing (token fractions of the global budget). The SFT budget is ``1 - cpt_frac`` (default 0.15);
within it the tasks are weighted contradiction 2x, rerank/outlier 1.5x, nq/oolong 1x (sum of
weights = 7), and CPT takes ``cpt_frac`` (default 0.85)::

    cpt          = 0.85
    contradiction= 0.15 * 2.0/7 = 0.042857   (assigned as the leftover: 1 - cpt - others)
    rerank       = 0.15 * 1.5/7 = 0.032143
    outlier      = 0.15 * 1.5/7 = 0.032143
    nq           = 0.15 * 1.0/7 = 0.021429
    oolong       = 0.15 * 1.0/7 = 0.021429

Each SFT task is completion-masked (loss only on assistant tokens); the CPT source carries an
explicit all-True mask (full-sequence loss). Documents are ConcatAndChunk-packed into 32k windows
with ``generate_doc_lengths=True`` so flash-attn varlen applies block-diagonal masking at EOS
document boundaries (no cross-document attention within a packed window).

Data lives on weka under ``prasanns/cptmix_data/`` (uploaded from the local /scratch data):
  * contradiction_8k          (token_ids_part_*.npy + labels_mask_*.npy)
  * nq_aligned_k20_qwen       ("")
  * oolong_qwen               ("")
  * rerank_qwen               ("")
  * outlier_qwen              ("")
  * dolma3_longmino_qwen3_sample (part-*.npy + all-True mask-*.npy -> full CPT loss)

The mixing fractions / max_repetition_factor logic mirror the local launcher exactly. Internal
script pattern (build_experiment_config + main, commands launch/train/dry_run)::

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-dense-cptmix-5task-64k-SFT.py \\
        dry_run q4b-cptmix-5task-64k ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-dense-cptmix-5task-64k-SFT.py \\
        launch q4b-cptmix-5task-64k ai2/jupiter-cirrascale-2
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
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
SEQUENCE_LENGTH = 65536  # 64k context (full ladder length); YaRN (factor 2, old 32k) extends to 64k.

# Context parallel (Ulysses) degree. Qwen3-4B: n_heads=32, n_kv_heads=8 -> head_stride=4, so an
# all-to-all CP degree of 8 splits both the 32 query heads and (with stride) the 8 kv heads cleanly.
CP_DEGREE = 8

# 1 node x 8 GPUs gives exactly one CP=8 replica (no DP). Set >1 to add data-parallel replicas
# (CP stays within a node; DP across nodes). Default 2 nodes -> 2 DP replicas x CP=8.
NUM_NODES = 2

# ---------------------------------------------------------------------------
# Data (weka). Uploaded from the local /scratch data that the winning 8k recipe used.
# ---------------------------------------------------------------------------
# LENGTH-LADDER (to-64k) SFT data: real long-context examples (ctx up to 64k) replacing the old
# <=8k packed shards. Tokenized with max_seq_len 65536, EOS 151643. CPT text is unchanged and stays
# in the original cptmix_data dir (reused, not re-uploaded).
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/cptmix_data_ladder64k"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
NQ_DATA_ROOT = f"{DATA_ROOT}/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"
CPT_DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/cptmix_data/"
    "dolma3_longmino_qwen3_sample"
)

# Dense CPT base (model+optim) on weka -- the dolma3longmino CPT of Qwen3-4B-Base. Loaded
# weights-only (load_optim_state=False), so only the model tensors are read from model_and_optim.
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "q4b-dense-dolma3longmino/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions (mirror the local launcher's weighting). CPT = 85% of tokens; the 15% SFT
# budget is split contradiction 2x / rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x (sum 7).
# ---------------------------------------------------------------------------
CPT_FRAC = 0.85
SFT_BUDGET = 1.0 - CPT_FRAC  # 0.15
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())  # 7.0
NQ_FRAC = SFT_BUDGET * _W["nq"] / _WSUM
OOLONG_FRAC = SFT_BUDGET * _W["oolong"] / _WSUM
RERANK_FRAC = SFT_BUDGET * _W["rerank"] / _WSUM
OUTLIER_FRAC = SFT_BUDGET * _W["outlier"] / _WSUM
# contradiction takes the leftover (== SFT_BUDGET * 2/7), matching the local launcher's remainder.
CONTRA_FRAC = max(0.0, 1.0 - CPT_FRAC - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

# ---------------------------------------------------------------------------
# Optimization / budget
# ---------------------------------------------------------------------------
LR = 1e-5  # the winning-recipe LR (cpt0.85 needs the lower 1e-5 to preserve RULER).

# Match the validated 8k recipe's optimizer dynamics (~1465 steps at lr1e-5). The smallest valid
# global batch here is ONE sequence/optimizer step (= SEQUENCE_LENGTH tokens); size the token budget
# to TARGET_STEPS. The earlier 16x-larger batch gave only ~61 steps at 64k and the SFT tasks never
# learned (dense-32k v1 contra f1 0.04 vs the 8k recipe's 0.76).
TARGET_STEPS = 1465
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH  # one instance/optimizer step (grad-accum 1 on the CP=8 replica)
TARGET_TOKENS = GLOBAL_BATCH_SIZE * TARGET_STEPS  # 64k -> 96.0M tokens
MAX_STEPS = max(1, round(TARGET_TOKENS / GLOBAL_BATCH_SIZE))  # = TARGET_STEPS = 1465


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
    # SFT/CPT shards are separated by SINGLE EOS tokens; qwen3 ties bos==eos, so drop BOS so the
    # document splitter treats each EOS as a doc boundary (not a doubled eos+bos).
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3-4B with YaRN context extension (native 32k -> 64k), full flash-attn 2 attention.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # 1 instance per CP replica per micro-step
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

    # ---- N-way mixed document source: 5 SFT tasks + CPT (mirrors the local launcher) ----
    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        """Completion-only masked SFT shards (loss only on assistant tokens)."""
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
        label_mask_paths=[f"{cpt}/mask-*.npy"],  # explicit all-True => full-sequence CPT loss
        expand_glob=True,
    )

    # (label, source, ratio, max_repetition_factor); rep factors mirror the local launcher.
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
        # Allow up to 3 passes of the ~40M-token CPT sample so a high CPT ratio still holds when the
        # token budget is scaled up (0.85 * 64M = 54.4M > 40M).
        MixingDocumentSourceSpecConfig(
            source=cpt_doc_source, ratio=CPT_FRAC,
            max_repetition_factor=3.0, label="cpt_longmino",
        ),
    ]

    instance_source_config = ConcatAndChunkInstanceSourceConfig(
        sources=[MixingDocumentSourceConfig(source_specs=specs)],
        sequence_length=SEQUENCE_LENGTH,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
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
                save_interval=100000,  # only at the end (ephemeral handles mid-run)
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
    """
    cpt0.85 + weighted 5-task SFT (contradiction/nq/oolong/rerank/outlier) of the Qwen3-4B dense
    CPT model at 32k context with Ulysses CP degree 8, multi-node.

        python src/scripts/train/sft/Qwen3-4B-dense-cptmix-5task-64k-SFT.py \\
            dry_run  q4b-cptmix-5task-64k ai2/jupiter-cirrascale-2
        python src/scripts/train/sft/Qwen3-4B-dense-cptmix-5task-64k-SFT.py \\
            launch   q4b-cptmix-5task-64k ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
