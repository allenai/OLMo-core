"""
Shared builder for the **SummTokenSFT** arms (Beaker/gantry).

Each context document in an SFT example is followed by a run of ``SUMMARY_TOKENS`` ``<|summ|>``
tokens. On a **masked** example a document reads only itself plus the summary runs of strictly
earlier documents, and the trailing query/answer reads the summary runs but no raw document content;
on a **causal** example the mask is plain causal. Which examples are which is decided per forward by
the mask-mixture schedule -- that is the axis the five arms vary.

See :mod:`olmo_core.nn.attention.summary_mask` for the mask and its levers, and
:class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention` for the kernel path.

⚠ **Scope of any claim from these runs.** Qwen3.5-4B is a hybrid
(``block_pattern=["gdn","gdn","gdn","attn"]``), so only **8 of 32** layers carry the mask; the 24
GatedDeltaNet layers ignore the roles and are an unrestricted cross-document channel. That is a
deliberate choice -- the intervention is on attention only -- but it means results describe what the
*attention* layers route, not the whole model. The realized split is logged at build and lands in the
saved ``config.json``; do not describe these runs as "documents communicate only through summary
tokens".

⚠ **Train only from a summary-repaired base.** ``<|summ|>`` is an untrained row in the embedding
matrix's padded region. Untrained rows are bit-identical *and* out-of-distribution in norm, and
RMSNorm amplifies a low-norm row into full-strength noise at every occurrence -- which flatlines
training at CE ~0.79 for **every** mask including plain causal, and reads as "the mask is too
restrictive" rather than as an embedding bug. Repair with::

    python src/scripts/data/fix_marker_embeddings.py --family qwen3_5 --model-size 4B \\
        --marker-set doc_start,doc_end,summary,pad --base <cpt>/model_and_optim \\
        --out <cpt>-summfix --audit-json audit.json

and gate the launch on ``audit_pass``.

Layout is PadToLength (one already-chunked example per window, padded) over a 5-task
MixingDocumentSource, exactly like the document-chunked family: roles are reconstructed from the
token stream, which needs one EOS-terminated example per instance, so no packing.
"""

import os
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.data.document_chunk_landmark import reserved_ids  # canonical ids -- never retype
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig
from olmo_core.launch.beaker import BeakerEnvVar, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
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

IDS = reserved_ids("qwen3_5")

# ---------------------------------------------------------------------------
# Geometry. Start at 32k: the mask machinery is exercised there by the document-chunked family, and
# padding waste at 256k is a real cost (one example per window over a 2k->256k ladder pads the short
# rungs almost entirely, and the chunked path cannot use PackingInstanceSource). Decide 256k from the
# launch_prep numbers -- see the README -- rather than launching blind.
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = int(os.environ.get("SUMMTOK_SEQ_LEN", "262144"))
SUMMARY_TOKENS = int(os.environ.get("SUMMTOK_N_SUMMARY", "5"))
NUM_NODES = int(os.environ.get("SUMMTOK_NUM_NODES", "2"))
GPUS_PER_NODE = 8
CP_DEGREE = int(os.environ.get("SUMMTOK_CP_DEGREE", "4"))
#: Instances are distributed across DP ranks only -- the CP ranks of one DP group all process the
#: SAME instance. This distinction is load-bearing for the curriculum; see derive_curriculum.
DP_DEGREE = NUM_NODES * GPUS_PER_NODE // CP_DEGREE

# Summary-token shards: the dense doc-chunked layout plus a <|summ|> run after each document, built by
#   convert_unified_to_document_landmark.py --emit summary --num-summary-tokens $SUMMTOK_N_SUMMARY
# ⚠ --num-summary-tokens MUST equal SUMMARY_TOKENS: roles are derived by counting summary RUNS, so a
# mismatch silently renumbers every document.
DATA_ROOT = os.environ.get(
    "SUMMTOK_DATA_ROOT",
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/summtoken_5task_xlong",
)

# The Qwen3.5 dense CPT base, AFTER the summary-row repair (see the module docstring).
BASE_CHECKPOINT = os.environ.get(
    "SUMMTOK_BASE",
    # q35-4b-dense-256k-fix/step2385 with the <|summ|> row repaired. NOTE: no user subdirectory.
    "/weka/oe-training-default/ai2-llm/checkpoints/q35-4b-dense-256k-summfix/model_and_optim",
)

# Mix weights -- identical to the document-chunked rows so the two families stay comparable.
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())

# LR anchored the same way as the 256k dense family: 1e-5 * sqrt(GLOBAL_BATCH_SIZE / 65_536).
LR = float(os.environ.get("SUMMTOK_LR", "4e-5"))
WORLD_SIZE = NUM_NODES * GPUS_PER_NODE
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * DP_DEGREE  # grad-accum 1
MAX_STEPS = int(os.environ.get("SUMMTOK_MAX_STEPS", "2240"))

#: Instances in one epoch of the realized mixture. Measured by ``launch_prep`` (read the
#: "MixingInstanceSource: N instances" line) and hardcoded, because the curriculum arms need it to
#: size their anneal and ``dry_run`` does NOT build the dataset.
N_INSTANCES = os.environ.get("SUMMTOK_N_INSTANCES")

#: One PadToLength window per DP rank per step, grad-accum 1.
MICRO_BATCH_INSTANCES = 1


def _arm_mixture(arm: str) -> dict:
    """The mask-mixture kwargs for one arm. ``p`` is the probability of the CAUSAL arm."""
    return {
        # Floor: every example masked. The first arm to run -- it is what surfaces any problem in a
        # mask that has never been trained.
        "summ-only": {},
        # Mode 1: a static random fraction of examples are causal.
        "summ-p25": {"standard_mix_prob": 0.25},
        # Mode 2: a hard phase switch -- everything masked, then everything causal.
        "summ-step50": {
            "mix_start_p": 0.0,
            "mix_end_p": 1.0,
            "mix_schedule": "step",
            "mix_step_frac": 0.5,
        },
        # Mode 3: the causal fraction rises smoothly through training.
        "summ-anneal": {"mix_start_p": 0.0, "mix_end_p": 0.5},
        # 50% masked / 50% causal, constant. At exactly 0.5 the two readings of "50% mask mixing"
        # coincide, so there is no direction to get wrong here.
        "summ-p50": {"standard_mix_prob": 0.5},
        # 100% masked -> 0% masked, linear: starts fully summary-only and ends fully causal.
        # p is P(CAUSAL), so "100% mask mixing decaying to 0%" is p: 0.0 -> 1.0.
        "summ-decay": {"mix_start_p": 0.0, "mix_end_p": 1.0},
        # THE CONTROL: same data, same summary tokens, same base -- only the mask differs.
        "summ-causal": {"standard_mix_prob": 1.0},
    }[arm]


ARMS = (
    "summ-only",
    "summ-p25",
    "summ-p50",
    "summ-step50",
    "summ-anneal",
    "summ-decay",
    "summ-causal",
)


def derive_curriculum(mixture: dict) -> dict:
    """
    Resolve ``mix_total_forwards`` and hard-fail if the anneal would not land on ``mix_end_p``.

    ``mix_total_forwards`` must be the number of forwards **one rank** performs, not the global
    count: the counter lives on the model and advances once per microbatch forward per rank. Getting
    this wrong leaves ``p`` short of its endpoint -- it has silently voided three prior arms
    (``records/contradiction-data-and-base-hygiene.md``), which is why this raises rather than warns.

    ⚠ Under context parallelism the divisor is **DP_DEGREE, not WORLD_SIZE**. The CP ranks of one DP
    group all process the *same* instance, so instances are spread over DP ranks only; dividing by
    the world size would make ``mix_total_forwards`` CP_DEGREE times too small and the anneal would
    finish a quarter of the way through training, pinned at ``mix_end_p`` for the rest. Every rank
    still advances its own counter once per forward, so the per-rank count is what this must be.
    """
    from olmo_core.nn.attention.chunked_mask import mask_mix_standard_prob

    if not any(k.startswith("mix_") for k in mixture):
        return {}
    if N_INSTANCES is None:
        raise OLMoConfigurationError(
            "The curriculum arms need the realized instance count to size their anneal, and "
            "'dry_run' does not build the dataset. Run 'launch_prep' on a CPU node, read the "
            "'MixingInstanceSource: N instances' line, and set SUMMTOK_N_INSTANCES."
        )
    n_instances = int(N_INSTANCES)
    forwards_per_rank = max(1, n_instances // (DP_DEGREE * MICRO_BATCH_INSTANCES))
    resolved = dict(mixture, mix_total_forwards=forwards_per_rank)

    final_p = mask_mix_standard_prob(forwards_per_rank, **resolved)
    if abs(final_p - mixture["mix_end_p"]) > 1e-6:
        raise OLMoConfigurationError(
            f"The mask-mix curriculum would not land: p ends at {final_p} rather than "
            f"{mixture['mix_end_p']} with mix_total_forwards={forwards_per_rank}. Check "
            f"N_INSTANCES={n_instances}, DP_DEGREE={DP_DEGREE} (NOT world size -- CP ranks share "
            f"an instance)."
        )
    return resolved


def _task_source(name: str, doc_tok) -> NumpyDocumentSourceConfig:
    root = f"{DATA_ROOT}/{name}_summary"
    return NumpyDocumentSourceConfig(
        source_paths=[f"{root}/token_ids_part_*.npy"],
        tokenizer=doc_tok,
        label_mask_paths=[f"{root}/labels_mask_*.npy"],
        expand_glob=True,
    )


def build_summtoken_experiment(cli_context: CliContext, arm: str) -> ExperimentConfig:
    if arm not in ARMS:
        raise OLMoConfigurationError(f"unknown arm {arm!r}; expected one of {ARMS}")
    mixture = derive_curriculum(_arm_mixture(arm)) or _arm_mixture(arm)

    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/amandab/{cli_context.run_name}"

    beaker_launch_config = build_launch_config(
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
        beaker_launch_config.allow_dirty = True
        for var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            beaker_launch_config.env_vars.append(
                BeakerEnvVar(name=var, value="expandable_segments:True")
            )
        # The Beaker job REBUILDS this config on the node, so anything resolved from the launch-host
        # environment must be propagated or the rebuild silently falls back to the defaults.
        for var in (
            "SUMMTOK_SEQ_LEN",
            "SUMMTOK_N_SUMMARY",
            "SUMMTOK_NUM_NODES",
            "SUMMTOK_DATA_ROOT",
            "SUMMTOK_BASE",
            "SUMMTOK_MAX_STEPS",
            "SUMMTOK_N_INSTANCES",
            "SUMMTOK_CP_DEGREE",
            "SUMMTOK_LR",
        ):
            if var in os.environ:
                beaker_launch_config.env_vars.append(BeakerEnvVar(name=var, value=os.environ[var]))

    tokenizer_config = TokenizerConfig.qwen3_5()
    # Qwen3.5 ties bos == eos, and the EOS split only fires on EOS-followed-by-BOS; bos=None is what
    # makes every EOS a document boundary for the source reader.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model: summary-token attention on the attention layers of the hybrid ----
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    # Only the "attn" block of the hybrid carries an AttentionConfig; the "gdn" blocks hold a
    # GatedDeltaNetConfig and are left alone (they ignore the roles -- see the module docstring).
    attn_mixer = model_config.block["attn"].sequence_mixer
    attn_mixer.name = AttentionType.summary_token
    attn_mixer.n_summary_tokens = SUMMARY_TOKENS
    # Defaults are the treatment: all summary tokens readable, relay on, query restricted to
    # summaries. Spelled out so the saved config states the experiment rather than implying it.
    attn_mixer.summary_visible_tokens = SUMMARY_TOKENS
    attn_mixer.summaries_read_own_document = True
    attn_mixer.summaries_read_earlier_summaries = True
    attn_mixer.query_reads_documents = False
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    model_config.summary_token_attention = {
        "doc_start_id": IDS.doc_start,
        "doc_end_id": IDS.doc_end,
        "summary_token_id": IDS.summary,
        "eos_id": IDS.eos,
        "pad_id": IDS.pad,
        "mix_seed": 42,
        **mixture,
    }

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(name="embeddings", opts=dict(weight_decay=0.0))],
        ),
        compile_model=False,  # GDN kernels
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=DP_DEGREE,
        ),
        # Ulysses ONLY. SummaryTokenAttention performs the all-to-all itself (it overrides sdpa and
        # so bypasses the backend); ring CP cannot express this mask and is rejected at runtime.
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    specs = [
        MixingDocumentSourceSpecConfig(
            source=_task_source(task, doc_tokenizer_config),
            ratio=_W[task] / _WSUM,
            max_repetition_factor=8.0,
            label=label,
        )
        for task, label in (
            ("contra", "contradiction"),
            ("nq", "nq_retrieval"),
            ("oolong", "oolong"),
            ("rerank", "rerank"),
            ("outlier", "outlier"),
        )
    ]

    instance_source_config = PadToLengthInstanceSourceConfig(
        sources=[MixingDocumentSourceConfig(source_specs=specs)],
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=doc_tokenizer_config,
    )

    # NOTE: no ``generate_doc_lengths``. Roles come from the token stream, and
    # SummaryTokenAttention REFUSES cu_doc_lens -- turning doc lengths on would raise at the first
    # forward, not degrade quietly.
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
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
                ephemeral_save_interval=max(1, MAX_STEPS // 4),
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
            "slack_notifier", SlackNotifierCallback(name=run_name_with_ts, enabled=False)
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    return ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=[instance_source_config],
        data_loader=data_loader_config,
    ).merge(cli_context.overrides)
