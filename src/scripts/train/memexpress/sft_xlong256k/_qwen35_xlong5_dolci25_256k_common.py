"""
Shared config for the 256k-context Qwen3.5-4B **DENSE** SFT arms on the xlong5 2k-256k 5-task
ladder + 25% ``allenai/Dolci-Instruct-SFT``.

THE CONTRAST
------------
Two arms, identical in every respect except which 5-task shard tree they read:

  * ``qboth``  -- ``xlong5_2k256k_qwen35/shards_full``, the original build (converter default
    ``--query-position both``: the task ask appears before *and* after the corpus).
  * ``qafter`` -- ``xlong5_2k256k_qwen35_qafter/shards_full``, a rebuild of the same pools with
    ``--query-position after`` (the ask appears only after the corpus).

Everything else is shared by construction rather than by convention -- both arms are built by
:func:`build_qwen35_xlong5_experiment` from the constants in this file, so the base checkpoint, the
75/25 blend, the within-mix weights, the window, the parallelism, the LR, the schedule, the seed and
the token budget cannot drift apart. **Do not fork this file to add an arm; add a row to _ARMS.**

READOUT
  The 5-task ladder evals at every rung, plus the xlong rungs and the OOD set, per the ``run-evals``
  skill's standing rules. The headline is the qafter-minus-qboth delta on long rungs.

  **Exclude outlier from that delta.** Per the qafter tree's README, outlier goes through the legacy
  ``grouping/outlier`` branch, which has no positioned query and was already query-after; its shards
  were rebuilt only so the root is self-contained. The two arms' outlier data are the same task
  format. Four of the five tasks carry the contrast, not five.

  **Eval must pass ``--query-position after`` for the qafter arm** and the default for qboth. There
  is no ``eval/`` directory on the qafter root: the xlong5 eval rungs are raw unified JSONL rendered
  at eval time, so query position is an eval-time flag rather than something baked into files. Both
  arms read ``xlong5_2k256k_qwen35/eval/``; only the flag differs. Getting this wrong makes a run
  read as a collapse.

WHAT IS *NOT* MATCHED BETWEEN THE ARMS, AND WHICH WAY IT BIASES
  The qafter rebuild also tightened the instance-length cap from 262,144 to 250,000, dropping the
  112 of 99,944 instances (0.11%) that sat above 250,000 -- concentrated in the 128-256k band, i.e.
  in exactly the rungs under test. The qboth arm keeps them. This is baked into the data pair and
  cannot be fixed in config: the window must be a power of two for ``PackingInstanceSource``, so
  there is no way to make the qboth arm drop the same documents. It is 0.11% of instances and
  ~1.8% of tokens; name it if a long-rung delta comes out small.

  Per-task, the token deltas are dominated by that cap rather than by the query move -- e.g. outlier
  is a documented query-position no-op yet still shows 20 fewer instances and 5.1M fewer tokens,
  which is the cap alone. The 112 dropped instances reconcile exactly: contra 27, nq 22, oolong 7,
  outlier 20, rerank 36.

RELATIONSHIP TO THE LEGACY 256k RUNS
  ``src/scripts/train/sft/amanda-landmark/Qwen3.5-4B-dense-xlong5-dolci25-256k-SFT.py`` (run
  ``q35-4b-dense-xlong5-dolci25-256k``) trained the qboth data from the same base checkpoint at
  560 steps x 4.19M tokens/step and LR 1e-5. The ``qboth`` arm here is NOT that run: it reproduces
  its data at this family's batch and LR (see "Optimization"). That is deliberate -- the legacy run
  cannot serve as the control for ``qafter``, because it differs in batch and LR as well as data.
  Use ``qboth`` as the control and treat the legacy run as a separate, differently-optimized point.
"""

from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional

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
from olmo_core.internal.experiment import CliContext, ExperimentConfig
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
# CP stays at 4, the value the 256k CPT run measured as fitting in 80GB. What sets this family's
# batch apart from the legacy 256k runs is the NODE COUNT: 2 nodes instead of 8, so DP = 16/4 = 4
# against their 16. Raising CP to 16 here (legal on both halves of the hybrid -- Ulysses needs only
# ``cp | n_heads`` since the 2026-08-01 KV-replication fix, and ``GatedDeltaNet.apply_cp`` asserts
# cp divides n_v_heads=32 / key_dim=2048 / value_dim=4096) would drive DP to 1 and cut the batch by
# a further 4x, which is not the batch these runs are designed around.
# ---------------------------------------------------------------------------
CP_DEGREE = 4
NUM_NODES = 2  # 2 x 8 = 16 GPUs -> DP = 16 / 4 = 4
GPUS_PER_NODE = 8
DP_DEGREE = NUM_NODES * GPUS_PER_NODE // CP_DEGREE  # 4

# Shard params+optim across all DP ranks (pure FSDP), same intent as the CPT run's shard_degree=16.
#
# ** THE MEMORY RISK IN THIS CONFIG LIVES HERE. ** shard_degree can be at most DP, and DP is 4, so
# the optimizer state is spread over 4 ranks where the legacy 256k runs spread it over 16 -- roughly
# 18 GiB/rank against ~4.5 GiB for a 4B model under AdamW with fp32 moments. Activations are NOT
# reduced to compensate: at CP=4 each rank still holds 262144/4 = 65,536 tokens, exactly as those
# runs did. So this is their measured-to-fit activation load plus ~13 GiB of extra state. It should
# still fit in 80GB with full AC and expandable_segments, but it is the untested corner of this
# config -- an OOM, if it comes, comes in the first few steps.
#
# If it does OOM, the fix that preserves the experiment is 4 nodes at CP=8: DP stays 4, so the
# global batch, LR and step count are all unchanged, while per-rank activations halve to 32,768
# tokens. Raising CP at 2 nodes does NOT work -- it drops DP and changes the batch.
SHARD_DEGREE = DP_DEGREE

# ---------------------------------------------------------------------------
# Per-arm data. THE ONLY THING THAT DIFFERS BETWEEN ARMS.
#
# Counts are measured from each task's metadata.json on 2026-08-11, not estimated. Both trees are
# Qwen3.5-tokenized (vocab 248320, EOS/BOS/pad 248044); these models do NOT share the Qwen3
# vocabulary, and the ``shards_chunked`` sibling of each root carries <|box_start|>/<|box_end|>
# markers and belongs to the chunked arm, NOT to either of these.
# ---------------------------------------------------------------------------
_XLONG_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"

_ARMS: Dict[str, Dict[str, Any]] = {
    "qboth": dict(
        data_root=f"{_XLONG_ROOT}/xlong5_2k256k_qwen35/shards_full",
        query_position="both",  # converter default; the field is absent from these metadata.json
        # contra 351.9M / nq 352.3M / oolong 342.7M / outlier 352.6M / rerank 364.2M
        tokens=1_763_719_249,
        instances=99_944,
        longest_example=262_072,
    ),
    "qafter": dict(
        data_root=f"{_XLONG_ROOT}/xlong5_2k256k_qwen35_qafter/shards_full",
        query_position="after",
        # contra 343.3M / nq 346.3M / oolong 339.9M / outlier 347.5M / rerank 354.8M
        tokens=1_731_810_480,
        instances=99_832,
        longest_example=249_950,
    ),
}

# Neither tree has an example that reaches the window, so LongDocStrategy.exclude drops nothing on
# either arm and the packer is not a source of asymmetry. Assert it rather than trusting the comment.
for _name, _spec in _ARMS.items():
    assert (
        _spec["longest_example"] < SEQUENCE_LENGTH
    ), f"arm {_name}: longest example {_spec['longest_example']} would be dropped at {SEQUENCE_LENGTH}"

DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen35"

# Dense 256k CPT base (model+optim), loaded weights-only. Same checkpoint the legacy 256k pair uses.
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "q35-4b-dense-256k-fix/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions WITHIN the 5-task group (sum to 1.0): contra 2x / rerank 1.5x / outlier 1.5x /
# nq 1x / oolong 1x -- the canonical 5-task weighting, matching both legacy 256k arms.
#
# These are deliberately NOT the 32k family's {contra: 2.9, oolong: 1.3} weights. That set is
# *compensation* for the 32768-window dense packer dropping ~31% of contradiction tokens and ~23% of
# oolong tokens to LongDocStrategy.exclude; 2.9 = 2.0/0.69 and 1.3 = 1.0/0.77 restore the intended
# share. At 262,144 nothing is dropped on either arm (asserted above), so requesting 2.9 here would
# *deliver* 2.9 -- a +6.8pp overweight of contradiction against a shortfall that does not exist.
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
# TOKEN-MATCHED to the legacy 256k runs: those did 560 steps x 16 DP windows x 262,144 = 2.35B model
# tokens. These arms reach the same 2.35B in 2,240 steps of 4 windows. Both arms get the identical
# budget, so the qafter-vs-qboth contrast is at equal compute AND equal tokens.
#
# That also lands at ~one epoch: the 5-task side is ~1.73-1.76B tokens and binds the blend, so 75/25
# puts the mixture at ~2.31-2.35B tokens -> ~8.8-9.0k windows of 262,144 -> ~2,200-2,240 steps at
# DP=4. Confirm against the packer's instance count in each arm's launch_prep log.
#
# LR IS RESCALED from the legacy runs, and this is deliberate. Those inherited 1e-5 from the 32k SFT
# family (65,536 tokens/step) while running a 4,194,304-token batch -- a 64x batch at an unchanged
# LR, which we treat as mis-set rather than as a precedent. The anchor here is the 32k family
# directly:
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
#     than the 32k recipe's. Expect these arms to sit CLOSER to the base checkpoint than the 32k
#     family does, and read a weak-fine-tuning signature as undertraining before reading it as data.
#   * The 1e-5 anchor was itself inherited, never swept, so this extrapolates from an untuned point.
#
# None of that threatens the qafter-vs-qboth contrast, which holds LR fixed across the two arms.
#
# Warmup stays at the family's 3%, which is 67 steps here against 17 in the legacy runs.
# ---------------------------------------------------------------------------
LR = 4e-5  # = 1e-5 * sqrt(GLOBAL_BATCH_SIZE / 65_536), anchored on the 32k SFT family

# The legacy runs' budget, written out rather than folded into a tautology.
LEGACY_STEPS = 560
LEGACY_GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 16  # DP=16 there
TARGET_TOKENS = LEGACY_STEPS * LEGACY_GLOBAL_BATCH_SIZE  # 2,348,810,240

GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * DP_DEGREE  # 1,048,576 -- grad-accum 1
MAX_STEPS = round(TARGET_TOKENS / GLOBAL_BATCH_SIZE)  # 2,240
# model tokens: GLOBAL_BATCH_SIZE * MAX_STEPS = 2,348,810,240 (legacy runs: 2,348,810,240)


def arm_data_root(arm: str) -> str:
    """
    Look up the 5-task shard root for an arm.

    :param arm: One of the keys of ``_ARMS`` (``"qboth"`` or ``"qafter"``).

    :returns: The absolute weka path to that arm's ``shards_full`` directory.

    :raises KeyError: If ``arm`` is not a known arm.
    """
    if arm not in _ARMS:
        raise KeyError(f"unknown arm '{arm}'; expected one of {sorted(_ARMS)}")
    return _ARMS[arm]["data_root"]


def build_qwen35_xlong5_experiment(cli_context: CliContext, *, arm: str) -> ExperimentConfig:
    """
    Build the full experiment config for one arm of the xlong5 256k dense SFT pair.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :param arm: Which arm to build -- ``"qboth"`` (original data) or ``"qafter"``.

    :returns: The full experiment config.

    :raises KeyError: If ``arm`` is not a known arm.
    """
    data_root = arm_data_root(arm)

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
    # cross-entropy upcasts them.
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

    # ---- Two-way mixed document source: 5-task group + Dolci-Instruct-SFT ----
    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        r = root.rstrip("/")
        return NumpyDocumentSourceConfig(
            source_paths=[f"{r}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            # Matches both these builds' labels_mask_part_*.npy and the Dolci converter's
            # labels_mask_*.npy; the loader pairs token/mask files by sorted position.
            label_mask_paths=[f"{r}/labels_mask_*.npy"],
            expand_glob=True,
        )

    five_task_specs = [
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{data_root}/contradiction_train"),
            ratio=CONTRA_FRAC,
            max_repetition_factor=8.0,
            label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            # Built from the p10 pool, not the banned 98%-hard-negative build.
            source=_sft_source(f"{data_root}/nq_train"),
            ratio=NQ_FRAC,
            max_repetition_factor=8.0,
            label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{data_root}/oolong_train"),
            ratio=OOLONG_FRAC,
            max_repetition_factor=8.0,
            label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{data_root}/rerank_train"),
            ratio=RERANK_FRAC,
            max_repetition_factor=8.0,
            label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{data_root}/outlier_train"),
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
    # answer and leaves a fully-masked, NaN-loss window. Nothing is dropped on either arm; the
    # assertion over _ARMS above enforces that.
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
        seed=34521,  # same stream on both arms, so they are paired rather than independent draws
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


__all__ = ["arm_data_root", "build_qwen35_xlong5_experiment"]
