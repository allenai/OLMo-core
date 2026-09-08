"""
Shared builder for the **Qwen3.5-4B interleaved sparse/regular landmark** SFT arms at 256k context,
one per CPT arm produced by ``cpt/interleaved/``.

Each arm SFTs its own interleaved CPT checkpoint on 75% the xlong5 2k->256k 5-task mix / 25%
``allenai/Dolci-Instruct-SFT`` at a 262,144 window -- i.e. the same data, blend, batch, LR, schedule
and seed as ``sft_xlong256k/``'s dense pair. Holding all of that fixed is the point: the dense
``qboth`` arm (``q35-4b-dense-xlong5-qboth-dolci25-256k``) is then the architecture control, and the
only thing separating it from any arm here is which attention variant sits in the 8 full-attention
layers.

WHY THIS IS A SEPARATE FAMILY FROM ``sft_xlong256k/``
  That family's stated invariant is that everything except the 5-task shard root is shared by
  construction -- it is a controlled pair on *query position*, over one fixed dense base. An
  interleaved arm varies the base checkpoint AND the model architecture, so folding it into that
  ``_ARMS`` table would falsify the invariant its README advertises. This file mirrors its constants
  instead, and imports them where they are literally shared.

THE ARM DEFINITIONS ARE NOT RESTATED HERE
  ``ARMS`` and :func:`build_layer_types` are imported from
  ``cpt/interleaved/_qwen35_interleaved_landmark_256k_common.py``, the same functions the CPT runs
  built their models with. This is a correctness requirement, not tidiness: the SFT model must place
  the identical attention variant at each of layers 3, 7, ..., 31, or the CPT weights load into the
  wrong kernels and the run silently trains a different model. Do not re-declare a pattern here.

READOUT
  The 5-task ladder evals at every rung, plus the xlong rungs and the OOD set, per the ``run-evals``
  skill's standing rules. The headline is each arm minus the dense ``qboth`` arm on the long rungs,
  which isolates the interleaving; ``sparse-reg`` minus ``reg-sparse`` separates "how many regular
  layers" from "where they sit" at 4/4, and ``reg-first`` minus ``reg-last`` does the same at 1/7.

  Eval these with the qboth rendering (the default, NOT ``--query-position after``): the data root
  below is the qboth build, and mismatching the flag makes a run read as a collapse.

=====================================================================================
THE ONE AXIS THAT IS **NOT** MATCHED TO THE DENSE CONTROL -- READ BEFORE INTERPRETING
=====================================================================================
Landmark training inserts one landmark token every ``MEM_FREQ = 63`` content tokens, so a 262,144
window holds ``262144 / 64 * 63 = 258,048`` tokens of original content against the dense arm's
262,144. Two consequences, neither fixable in config:

1. **~1.56% less content per window.** At the shared 2,240-step budget these arms therefore see
   ~1.56% less original data than the dense control for the same compute. The 32k family solved the
   equivalent problem by *widening* the landmark window (40,960 against dense 32,768) so content
   capacity met or exceeded the dense window. That option does not exist here: 262,144 is the length
   the CPT arms actually trained at, and widening past it would evaluate the model outside the
   context it was continued-pretrained on, which is a larger confound than 1.56% of data.

2. **The long tail gets dropped, and only on these arms.** ``sft_xlong256k``'s README records
   qboth's longest example at 262,072 tokens, which fits its dense 262,144 window -- its assertion
   that ``LongDocStrategy.exclude`` drops nothing holds for the dense arms. Against a *content*
   capacity of 258,048 that same example does not fit, so the landmark packer drops it and every
   other example over 258,048, concentrated in the 128-256k band that these long-context arms exist
   to measure. MEASURED 2026-09-08 (prep ``01M20FXCTYN403D1ARAS23JX5Q``): **41 of 1,030,564
   documents, 0.004%** -- the same order as the 112-instance asymmetry the dense pair already
   tolerates between its own arms. ``warn_drop_fraction`` below is set to 0.0 so this is always
   re-reported if the data changes; see the README's prep readout.

Both points push the same way -- the landmark arms see marginally less, and marginally shorter,
data than the dense control -- so an interleaving deficit on the longest rungs is confounded with
them, while an interleaving *advantage* there is not.

The budget is comfortable against the packed count: the same prep packed 9,195 windows, i.e.
2,298.8 steps per epoch at DP=4, so MAX_STEPS=2,240 is 0.974 of an epoch and no data repeats.
"""

import os
import sys
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
    LandmarkPackingStrategy,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
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

# The CPT arm definitions live one tree over, in cpt/interleaved. Import them rather than restating
# them -- see "THE ARM DEFINITIONS ARE NOT RESTATED HERE" above.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpt", "interleaved"),
)

from _qwen35_interleaved_landmark_256k_common import (  # noqa: E402
    ARMS,
    BLOCK_SIZE,
    LANDMARK_TOKEN_ID,
    MEM_FREQ,
    REG_LANDMARK_TYPE,
    build_layer_types,
    describe_arm,
)

# ---------------------------------------------------------------------------
# Geometry. SEQUENCE_LENGTH is in landmark-token space and is deliberately the CPT runs' window, not
# a widened one -- see the module docstring's "ONE AXIS THAT IS NOT MATCHED" section.
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 262144  # 4096 blocks of 64
CONTENT_CAPACITY = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 258,048 tokens of original content

assert SEQUENCE_LENGTH % BLOCK_SIZE == 0, "the landmark packer requires a block-aligned window"

# ---------------------------------------------------------------------------
# Parallelism. Identical to both the CPT arms and the dense SFT pair:
#   * TP=1     -- GatedDeltaNet.apply_tp raises NotImplementedError.
#   * Ulysses  -- ring/zigzag CP is rejected by GDN and by every landmark variant.
#   * CP <= 4  -- SparseLandmarkAttention.apply_cp() requires cp | n_kv_heads (=4), and every arm
#                 here contains sparse layers. This is a hard cap, not a tuning choice.
# ---------------------------------------------------------------------------
CP_DEGREE = 4
NUM_NODES = 2  # 2 x 8 = 16 GPUs -> DP = 16 / 4 = 4
GPUS_PER_NODE = 8
DP_DEGREE = NUM_NODES * GPUS_PER_NODE // CP_DEGREE  # 4
SHARD_DEGREE = DP_DEGREE

# ---------------------------------------------------------------------------
# Data. The qboth (query-position "both") build -- the dense family's CONTROL arm, and the only one
# of the two that ships an eval/ directory. Everything about the blend is copied from
# sft_xlong256k/_qwen35_xlong5_dolci25_256k_common.py so the two families stay comparable.
# ---------------------------------------------------------------------------
_XLONG_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
DATA_ROOT = f"{_XLONG_ROOT}/xlong5_2k256k_qwen35/shards_full"
DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen35"

# Where the CPT arms wrote their checkpoints: cpt/interleaved saves to
# f"{root_dir}/checkpoints/{run_name}" (no username segment), which is the path the existing
# ev-* eval jobs read for the regsparse arm.
_CPT_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints"
CPT_STEP = 2385  # MAX_TOKENS=10B / (262144*16) -> 2,384.2 -> the 2385 the finished arms wrote

#: Arm key (as used by ``cpt/interleaved``) -> the CPT Beaker run name that produced its checkpoint.
ARM_TO_CPT_RUN: Dict[str, str] = {
    "reg-first": "q35-4b-il-regfirst-256k",
    "reg-last": "q35-4b-il-reglast-256k",
    "sparse-reg": "q35-4b-il-sparsereg-256k",
    "reg-sparse": "q35-4b-il-regsparse-256k",
}

# Every CPT arm must have a base checkpoint mapping, and no extras. If cpt/interleaved gains or
# renames an arm, this fires at import rather than launching against a path that does not exist.
_ARM_MISMATCH = sorted(set(ARM_TO_CPT_RUN) ^ set(ARMS))
assert not _ARM_MISMATCH, f"arm mismatch against cpt/interleaved: {_ARM_MISMATCH}"

# ---------------------------------------------------------------------------
# Mixing fractions WITHIN the 5-task group, copied verbatim from the dense 256k family: contra 2x /
# rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x. These are NOT the 32k family's 2.9/1.3 weights --
# those compensate for a 32,768-window packer dropping long contradiction/oolong documents, a
# shortfall that is ~0 at this window.
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
# Optimization / budget. Held identical to the dense 256k pair so those arms are usable as the
# control: 2,240 steps x 1,048,576 window tokens = 2.35B, LR 4e-5 (= 1e-5 * sqrt(1048576/65536),
# anchored on the 32k SFT family), 3% warmup, seed 34521. The derivation and its caveats live in
# sft_xlong256k/_qwen35_xlong5_dolci25_256k_common.py; do not re-derive them here, and do not change
# one family's value without the other.
#
# NOTE the budget is matched in WINDOW tokens, which is the compute-matched reading. It is ~1.56%
# short in CONTENT tokens, for the landmark-geometry reason in the module docstring.
# ---------------------------------------------------------------------------
LR = 4e-5
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * DP_DEGREE  # 1,048,576 -- grad-accum 1
MAX_STEPS = 2240  # sft_xlong256k's MAX_STEPS; keep the two families equal by construction
SEED = 34521

#: Minimum guaranteed runtime before Beaker may preempt. At 2 nodes these arms take hours to reach
#: their first ephemeral checkpoint, so a preemption in the first minutes costs the whole warmup.
MIN_RUNTIME = "1h"


def cpt_checkpoint(arm: str) -> str:
    """
    The CPT checkpoint an SFT arm continues from.

    :param arm: One of the keys of ``ARMS`` (e.g. ``"reg-first"``).

    :returns: The absolute weka path to that arm's ``model_and_optim`` directory.

    :raises KeyError: If ``arm`` is not a known arm.
    """
    if arm not in ARM_TO_CPT_RUN:
        raise KeyError(f"unknown arm '{arm}'; expected one of {sorted(ARM_TO_CPT_RUN)}")
    return f"{_CPT_ROOT}/{ARM_TO_CPT_RUN[arm]}/step{CPT_STEP}/model_and_optim"


def build_qwen35_interleaved_sft_experiment(
    cli_context: CliContext, *, arm: str
) -> ExperimentConfig:
    """
    Build the full experiment config for one interleaved-landmark 256k SFT arm.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :param arm: One of the keys of ``ARMS``.

    :returns: The full experiment config.

    :raises KeyError: If ``arm`` is not a known arm.
    """
    # Raises for an unknown arm before anything else is built.
    base_checkpoint = cpt_checkpoint(arm)
    layer_types = build_layer_types(arm)

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
        beaker_launch_config.min_runtime = MIN_RUNTIME
        # Put the layer layout on the Beaker workload itself, so an arm is identifiable from the
        # UI without opening the config.
        beaker_launch_config.description = (
            f"interleaved-landmark 256k SFT, arm {arm!r}: {describe_arm(arm)}; "
            f"from {ARM_TO_CPT_RUN[arm]}/step{CPT_STEP}"
        )
        # Carried over from the CPT arms: at this context length the allocator fragments badly
        # enough to OOM with ~20% of the card stranded in reserved-but-unusable segments. Both
        # spellings -- torch 2.9 renamed the variable and warns on the old name, but older images
        # only honour the old one.
        for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            beaker_launch_config.env_vars.append(
                BeakerEnvVar(name=_var, value="expandable_segments:True")
            )

    tokenizer_config = TokenizerConfig.qwen3_5()
    # Qwen3.5 ties bos == eos == 248044, and EOS-based document splitting only fires on an EOS
    # *followed by* a BOS -- which never happens in single-EOS-separated SFT data. bos=None makes
    # every EOS a boundary. Used for the document sources; the loader takes the plain config
    # because the landmark packer, not EOS scanning, supplies document boundaries.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # The backend is inert on these arms: all 8 full-attention blocks resolve to a landmark variant,
    # each of which runs its own Triton kernel rather than routing through ``self.backend``, and the
    # 24 GDN blocks never touch a flash backend. Left at flash_2 exactly as the CPT arms had it, so
    # nothing implies an FA3 dependency that does not exist. (The dense 256k SFT pair uses flash_3
    # because on those arms the backend is live.)
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    # Swap ONLY the full-attention layers, per this arm's pattern, keeping their elementwise output
    # gate (both landmark variants apply it, so w_g loads straight from the CPT checkpoint).
    attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
    # ``name`` is only the fallback; ``layer_types`` overrides it per layer.
    attn_mixer.name = REG_LANDMARK_TYPE
    attn_mixer.layer_types = layer_types
    attn_mixer.mem_freq = MEM_FREQ
    attn_mixer.num_landmarks = 1  # matches the packer's 1-landmark-per-block data

    # Mandatory at 256k: dense logits over a 248,320 vocab would be tens of GB in bf16 before
    # cross-entropy upcasts them.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one full sequence per DP rank, split across CP
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,  # SFT, as in the dense 256k pair (the CPT arms used 0.1)
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
            label_mask_paths=[f"{r}/labels_mask_*.npy"],
            expand_glob=True,
        )

    five_task_specs = [
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{DATA_ROOT}/contradiction_train"),
            ratio=CONTRA_FRAC,
            max_repetition_factor=8.0,
            label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            # Built from the p10 pool, not the banned 98%-hard-negative build.
            source=_sft_source(f"{DATA_ROOT}/nq_train"),
            ratio=NQ_FRAC,
            max_repetition_factor=8.0,
            label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{DATA_ROOT}/oolong_train"),
            ratio=OOLONG_FRAC,
            max_repetition_factor=8.0,
            label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{DATA_ROOT}/rerank_train"),
            ratio=RERANK_FRAC,
            max_repetition_factor=8.0,
            label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(f"{DATA_ROOT}/outlier_train"),
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

    # Block-aligned packing with per-document landmarks; the document boundaries it emits are
    # multiples of BLOCK_SIZE, which is what the landmark kernels require for packed masking.
    #
    # best_fit_decreasing, not the next_fit default, to packing-match the dense 256k pair (which
    # uses PackingInstanceSource's BFD). Leaving it at next_fit would put a packer difference on top
    # of the architecture difference this family exists to measure.
    #
    # warn_drop_fraction is lowered from the 0.01 default to 0.0 so the log reports the long-document
    # drop even when it is a handful of examples: on these arms it is NOT expected to be zero (the
    # content capacity is 258,048 against a 262,072-token longest example) and the count has to be
    # read out of launch_prep. See the module docstring.
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=MixingDocumentSourceConfig(source_specs=specs),
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
        num_landmarks=1,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
        warn_drop_fraction=0.0,
    )

    data_loader_config = ComposableDataLoaderConfig(
        # The landmark packer supplies document boundaries, so the loader does not EOS-scan and
        # takes the plain tokenizer config (contrast the dense family, which needs bos=None here).
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,  # same stream on every arm, so the arms are paired rather than independent draws
        num_workers=4,
        generate_doc_lengths=False,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=base_checkpoint,
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


__all__ = [
    "ARM_TO_CPT_RUN",
    "build_qwen35_interleaved_sft_experiment",
    "cpt_checkpoint",
    "describe_arm",
]
