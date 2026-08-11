"""
256k-context Beaker/gantry SFT of the Qwen3.5-4B **DENSE** 256k CPT model (the GDN hybrid with its
full-attention blocks left alone) on 75% the **query-after** 2k-256k 5-task mix / 25%
``allenai/Dolci-Instruct-SFT``.

THE CONTRAST
------------
This arm vs. ``src/scripts/train/sft/amanda-landmark/Qwen3.5-4B-dense-xlong5-dolci25-256k-SFT.py``
(run ``q35-4b-dense-xlong5-dolci25-256k``): same model, same base checkpoint, same 75/25 blend,
same within-mix weights, same token budget -- the 5-task shards are swapped from
``xlong5_2k256k_qwen35/shards_full`` to ``xlong5_2k256k_qwen35_qafter/shards_full``. Per that
tree's README, it is a ``--query-position after`` rebuild of the same pools with the converter
default ``both`` replaced; pools, tokenizer, markers, chunking and every known-bad knob are held
identical, so the two roots are a controlled pair on where the task ask sits.

Two things about that pair, both from the qafter README, that constrain how it can be read:

  * **Outlier is a deliberate no-op.** It goes through the legacy ``grouping/outlier`` branch, which
    has no positioned query and was already query-after; its shards are rebuilt only so the root is
    self-contained. The two arms' outlier data are the same. **Exclude outlier from any
    query-position ablation** -- four of the five tasks change, not five.
  * **The length cap moved, 262,144 -> 250,000.** The converter drops over-length examples rather
    than truncating them, so this build drops 112 of 99,944 full-arm instances (~0.12%) that the
    ``both`` build kept, concentrated in the 128-256k band -- i.e. in exactly the rungs under test.
    Tiny in aggregate; worth naming if a long-rung delta is small.

READOUT
  The 5-task ladder evals (contradiction / nq / oolong / rerank -- **not** outlier, per above) at
  every rung, plus the xlong rungs and the OOD set, per the ``run-evals`` skill's standing rules.
  The headline number is whether moving the query after the context changes long-rung accuracy.

  **Eval must pass ``--query-position after``.** There is no ``eval/`` directory on the qafter root:
  the xlong5 eval rungs are raw unified JSONL rendered at eval time, so query position is an
  eval-time flag rather than something baked into files. Keep reading
  ``xlong5_2k256k_qwen35/eval/`` and pass the flag, or train and eval disagree on the format and the
  run reads as a collapse.

NOT MATCHED, AND WHICH WAY IT BIASES
  **Global batch size AND learning rate.** The run being mimicked used DP=16 x 262,144 = 4.19M
  tokens/step at LR 1e-5; this one uses DP=4 x 262,144 = **1.05M tokens/step at LR 4e-5** (see
  "Parallelism" and "Optimization" below). The token budget is held fixed at 2.35B, so this arm
  takes 2,240 optimizer steps where that one took 560.

  Apart from the intended data swap this is the axis the two runs do not share, and it is not a
  small one: 4x the steps at 4x the LR is ~16x the mimicked run's path length. The direction
  of the bias is that this arm fine-tunes considerably harder, so it should look better on
  in-distribution ladder rungs and is the more likely of the two to have given up base long-context
  behavior. **A qafter-vs-standard delta from these two runs is therefore not a clean read on the
  data.** The honest control is a standard-data arm rerun at this batch size and LR; until that
  exists, treat this run as a pilot on the qafter build rather than as half of a controlled pair.

  The LR change is deliberate: the mimicked run's 1e-5 at a 4.19M batch is regarded as mis-set
  rather than as a precedent worth matching. See "Optimization" for the scaling derivation.

Data (all Qwen3.5-tokenized -- these models do NOT share the Qwen3 vocabulary):

  * 5-task qafter mix, 75%, from ``xlong5_2k256k_qwen35_qafter/shards_full`` (the ``doc_markers:
    false`` "standard" arm; the ``shards_chunked`` sibling carries ``<|box_start|>``/``<|box_end|>``
    boundaries and is for the chunked arm, NOT this one). Measured from the per-task
    ``metadata.json`` on 2026-08-11: contradiction 343.3M / nq 346.3M / oolong 339.9M /
    outlier 347.5M / rerank 354.8M = **1.732B tokens**, 99,832 documents. Longest example
    249,950 tokens (the build's own 250,000 cap), so the 262,144 packer drops nothing.
  * ``allenai/Dolci-Instruct-SFT``, 25%, from the Qwen3.5 retokenization at
    ``amandab/dolci-instruct-sft/qwen35``. The ``dolci-instruct-sft/qwen3`` tree is a *different
    vocabulary* and must not be used here.

    S=src/scripts/train/memexpress/sft_xlong256k/Qwen3.5-4B-dense-xlong5-qafter-dolci25-256k-SFT.py

    PYTHONPATH=src python $S dry_run q35-4b-dense-xlong5-qafter-dolci25-256k ai2/jupiter-cirrascale-2

    # Build the mixture on CPU first -- dry_run does NOT touch the data. Read the
    # 'MixingInstanceSource: NNB tokens' and 'packed N windows' lines out of the job log and confirm
    # the per-task shares and that N/DP is near the 2,240 steps below.
    PYTHONPATH=src python $S launch_prep q35-4b-dense-xlong5-qafter-dolci25-256k-prep \\
        ai2/jupiter-cirrascale-2

    PYTHONPATH=src python $S launch q35-4b-dense-xlong5-qafter-dolci25-256k \\
        ai2/jupiter-cirrascale-2 --launch.follow=false --launch.step_soft_timeout=null
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
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
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

SEQUENCE_LENGTH = 262144  # 256k; also a power of 2, which PackingInstanceSource requires

# ---------------------------------------------------------------------------
# Parallelism. The architecture forces two of these:
#   * TP=1     -- GatedDeltaNet.apply_tp raises NotImplementedError.
#   * Ulysses  -- ring CP is rejected by GatedDeltaNet.
#
# What shrinks the batch relative to the mimicked run is the NODE COUNT, not the CP degree: 2 nodes
# instead of 8, at the same CP=4, gives DP = 16 / 4 = 4 against that run's 16. CP is deliberately
# left at the mimicked run's value -- it is the only 256k setting *measured* to fit in 80GB, and
# raising it to 16 here (legal on both halves of the hybrid: Ulysses needs only ``cp | n_heads``
# since the 2026-08-01 KV-replication fix, and ``GatedDeltaNet.apply_cp`` asserts cp divides
# n_v_heads=32 / key_dim=2048 / value_dim=4096) would drive DP to 1 and cut the global batch by a
# further 4x, which is not the batch this run was designed around.
#
# Cost of 2 nodes over 8: ~4x the wall-clock for the same 2.35B tokens. Nothing else moves.
# ---------------------------------------------------------------------------
CP_DEGREE = 4
NUM_NODES = 2  # 2 x 8 = 16 GPUs -> DP = 16 / 4 = 4
GPUS_PER_NODE = 8
DP_DEGREE = NUM_NODES * GPUS_PER_NODE // CP_DEGREE  # 4

# Shard params+optim across all DP ranks (pure FSDP), same intent as the CPT run's shard_degree=16.
#
# ** THE MEMORY RISK IN THIS CONFIG LIVES HERE. ** shard_degree can be at most DP, and DP is 4, so
# the optimizer state is spread over 4 ranks where the mimicked run spread it over 16 -- roughly
# 18 GiB/rank against ~4.5 GiB for a 4B model under AdamW with fp32 moments. Activations are NOT
# reduced to compensate: at CP=4 each rank still holds 262144/4 = 65,536 tokens, exactly as that run
# did. So this is the mimicked run's measured-to-fit activation load plus ~13 GiB of extra state.
# It should still fit in 80GB with full AC and expandable_segments, but it is the untested corner of
# this config -- an OOM, if it comes, comes in the first few steps.
#
# If it does OOM, the fix that preserves the experiment is 4 nodes at CP=8: DP stays 4, so the
# global batch, LR and step count are all unchanged, while per-rank activations halve to 32,768
# tokens. Raising CP at 2 nodes does NOT work -- it drops DP and changes the batch.
SHARD_DEGREE = DP_DEGREE

# ---------------------------------------------------------------------------
# Data (weka). All Qwen3.5-tokenized; EOS/BOS/pad 248044.
#
# The ONLY difference from the mimicked run: '_qafter' in DATA_ROOT. Per-task metadata.json on this
# tree reports query_position="after"; the standard tree it replaces does not.
# ---------------------------------------------------------------------------
DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/"
    "xlong5_2k256k_qwen35_qafter/shards_full"
)
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction_train"
NQ_DATA_ROOT = f"{DATA_ROOT}/nq_train"  # built from the p10 pool, not the banned 98%-hard build
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong_train"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank_train"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier_train"

DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen35"

# Dense 256k CPT base (model+optim), loaded weights-only -- the same checkpoint the mimicked run
# loads, which is what makes the two comparable at all.
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "q35-4b-dense-256k-fix/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions WITHIN the 5-task group (sum to 1.0): contra 2x / rerank 1.5x / outlier 1.5x /
# nq 1x / oolong 1x -- carried over unchanged from the mimicked 256k run.
#
# These are deliberately NOT the 32k family's {contra: 2.9, oolong: 1.3} weights. That set is
# *compensation* for the 32768-window dense packer dropping ~31% of contradiction tokens and ~23% of
# oolong tokens to LongDocStrategy.exclude. At 262,144 nothing is dropped -- this build's longest
# example is 249,950 tokens (max over the five metadata.json files) -- so applying the compensation
# here would upweight contradiction by 45% against nothing at all.
# ---------------------------------------------------------------------------
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())
NQ_FRAC = _W["nq"] / _WSUM
OOLONG_FRAC = _W["oolong"] / _WSUM
RERANK_FRAC = _W["rerank"] / _WSUM
OUTLIER_FRAC = _W["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, 1.0 - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

FIVE_TASK_FRAC = 0.75
DOLCI_FRAC = 0.25

# ---------------------------------------------------------------------------
# Optimization / budget.
#
# TOKEN-MATCHED to q35-4b-dense-xlong5-dolci25-256k: that run did 560 steps x 16 DP windows x
# 262,144 = 2.35B model tokens. This run reaches the same 2.35B in 2,240 steps of 4 windows.
#
# That also lands at ~one epoch, as it did there: the 5-task side is 1.732B tokens (measured) and
# binds the blend, so 75/25 puts the mixture at ~2.31B tokens -> ~8.8k windows of 262,144 ->
# ~2,200 steps at DP=4. Confirm against the packer's actual instance count in the launch_prep log
# before reading the run as a full epoch.
#
# LR IS RESCALED, and this is a deliberate break from the mimicked run. That run inherited 1e-5
# from the 32k SFT family (65,536 tokens/step) while running a 4,194,304-token batch -- a 64x batch
# at an unchanged LR, which we are treating as a bug rather than a precedent. So the anchor here is
# the 32k family directly, not the 256k pair:
#
#     LR = 1e-5 * sqrt(1_048_576 / 65_536) = 1e-5 * 4 = 4e-5
#
# Square-root rather than linear (which would give 1.6e-4) because linear scaling is derived for
# SGD, where k sequential steps sum to one k-times-larger step. Adam normalizes by the second
# moment, so its per-step update size is ~LR regardless of gradient magnitude and that argument does
# not carry; the noise-preserving SDE rule for Adam/RMSProp is sqrt (Malladi et al. 2022). Caveats
# recorded rather than hidden:
#
#   * That rule also asks for eps ~ sqrt(B) and beta2 -> 1 as the batch grows. We do neither, so the
#     4e-5 comes with the rule's LR but not its supporting conditions -- it is approximate.
#   * sqrt preserves gradient-NOISE scale, not distance travelled. At a fixed token budget a 16x
#     batch means 16x fewer steps against only a 4x larger step, so total path length is ~4x shorter
#     than the 32k recipe's. Expect this arm to sit CLOSER to the base checkpoint than the 32k
#     family does, and read a weak-fine-tuning signature as undertraining before reading it as data.
#   * The 1e-5 anchor was itself inherited, never swept, so this extrapolates from an untuned point.
#
# Warmup stays at the family's 3%, which is 67 steps here against 17 in the mimicked run.
# ---------------------------------------------------------------------------
LR = 4e-5  # = 1e-5 * sqrt(GLOBAL_BATCH_SIZE / 65_536), anchored on the 32k SFT family

# The mimicked run's budget, written out rather than folded into a tautology.
REFERENCE_STEPS = 560
REFERENCE_GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 16  # DP=16 there
TARGET_TOKENS = REFERENCE_STEPS * REFERENCE_GLOBAL_BATCH_SIZE  # 2,348,810,240

GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * DP_DEGREE  # 1,048,576 -- grad-accum 1
MAX_STEPS = round(TARGET_TOKENS / GLOBAL_BATCH_SIZE)  # 2,240
# model tokens: GLOBAL_BATCH_SIZE * MAX_STEPS = 2,348,810,240 (reference run: 2,348,810,240)


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
        # Carried over from the CPT run: at this sequence length the allocator fragments badly
        # enough to OOM with ~20% of the card stranded in reserved-but-unusable segments. Both
        # spellings -- torch 2.9 renamed the variable and warns on the old name, but older images
        # only honour the old one.
        for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            beaker_launch_config.env_vars.append(
                BeakerEnvVar(name=_var, value="expandable_segments:True")
            )

    tokenizer_config = TokenizerConfig.qwen3_5()
    # Qwen3.5 ties bos == eos == 248044, and the EOS-based document split only fires on an EOS
    # *followed by* a BOS -- which never happens in single-EOS-separated SFT data. bos=None makes
    # every EOS a boundary, which is what produces correct block-diagonal (varlen) masking.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # flash_3 (Hopper FA3), as in the CPT run: at 256k the attention kernel dominates wall-clock
    # even though only 8 of 32 blocks are full attention. Fall back with
    # --model.block.attn.sequence_mixer.backend=flash_2 if it misbehaves (costs ~a third of
    # throughput).
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )

    # Mandatory at 256k: dense logits over a 248,320 vocab would be tens of GB in bf16 before
    # cross-entropy upcasts them, even at the 16,384 tokens/rank CP=16 leaves.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one full sequence per DP rank, split across CP
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
        # GatedDeltaNet custom kernels; compile off, which also rules out 'budget' AC.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=SHARD_DEGREE,
        ),
        # Ulysses only: GatedDeltaNet.apply_cp() rejects ring CP.
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Two-way mixed document source: 5-task qafter group + Dolci-Instruct-SFT ----
    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        r = root.rstrip("/")
        return NumpyDocumentSourceConfig(
            source_paths=[f"{r}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            # Matches both this build's labels_mask_part_*.npy and the Dolci converter's
            # labels_mask_*.npy; the loader pairs token/mask files by sorted position.
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

    # Best-Fit-Decreasing bin-packing of WHOLE documents into each window (no document is sliced
    # across a window boundary; leftover space is padded). Documents longer than SEQUENCE_LENGTH are
    # DROPPED rather than truncated -- truncating a long-context example cuts off its trailing
    # answer and leaves a fully-masked, NaN-loss window. Nothing is dropped here: the qafter shards
    # were built against a 262,144 cap and their longest example is 249,950 tokens.
    instance_source_config = PackingInstanceSourceConfig(
        sources=[MixingDocumentSourceConfig(source_specs=specs)],
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=doc_tokenizer_config,
        long_doc_strategy=LongDocStrategy.exclude,
    )

    # NOTE: the loader must use doc_tokenizer_config (bos_token_id=None) -- see above.
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=doc_tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,  # same stream as the mimicked run
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
                # A permanent (non-ephemeral) checkpoint at the end -- the ladder/xlong evals read
                # it, and ephemeral saves are the ones max_checkpoints prunes.
                save_interval=MAX_STEPS,
                ephemeral_save_interval=MAX_STEPS // 4,
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
