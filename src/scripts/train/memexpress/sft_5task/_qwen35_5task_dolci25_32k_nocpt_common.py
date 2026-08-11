"""
Shared builder for the **Qwen3.5-4B** 5-task + Dolci-25% 32k no-CPT SFT arms.

Three arms, one per finished Qwen3.5 CPT run on the dolma3_longmino sample (all at ``step2385``):

  * ``dense``           -> plain hybrid Qwen3.5-4B (GDN + full attention, 3:1)
  * ``fast-landmark``   -> full-attention layers swapped to ``AttentionType.fast_landmark``
  * ``sparse-landmark`` -> full-attention layers swapped to ``AttentionType.sparse_landmark``

plus two **data-vs-compute ablation** arms, both ``fast_landmark`` against the ``dense`` arm:

  * ``fast-landmark-datamatch``  -> same *original data* as dense; spends more tokens for it
  * ``fast-landmark-tokenmatch`` -> same *token budget* as dense; therefore sees less data

Landmark training inserts one landmark token per ``MEM_FREQ=63`` content tokens, so a document
occupies 64/63 = 1.0159x its original length and every token-matched landmark-vs-dense comparison
we have run saw *less original data* on the landmark side. These two arms bracket that: if they
agree, the landmark result is an architecture effect; if they disagree, it is a data effect. They
remove the two confounds catalogued in ``records/POSSIBLE_BUG_SFT_DATA.md``:

1. **Window width.** ``ABLATION_SEQ_LEN = 33344`` (521 blocks of 64) has a content capacity of
   521 x 63 = 32,823 >= the dense arm's 32,768, so one landmark window carries at least as much
   original content as one dense window.
2. **Packing algorithm.** Both use ``LandmarkPackingStrategy.best_fit_decreasing``, the same
   algorithm the dense arm's ``PackingInstanceSource`` uses, instead of the default next-fit that
   inflated the landmark instance count by ~13%. What remains is the genuine landmark cost: each
   document is ceil'd to a whole number of blocks.

They also take the dense arm's ``2.9/1.3`` sampling weights, not the landmark arms' ``2.0/1.0``:
those weights compensate for over-long documents the packer drops, and at 33,344 the landmark drop
threshold (content > 32,823) is within 0.17% of the dense one, so the same compensation applies.
The ``fast-landmark`` (40960) and ``sparse-landmark`` arms are untouched by all of this.

The data recipe is the canonical 32k one carried over from the Qwen3 runs (75% the 5 long-context
tasks / 25% ``allenai/Dolci-Instruct-SFT``, p10 NQ, no CPT text), **re-tokenized for the Qwen3.5
vocabulary from the same source JSONL in the same row order**, so each arm draws the same examples
in the same sequence as its Qwen3 counterpart (up to per-example token-count differences between the
two tokenizers). See ``records/POSSIBLE_BUG_SFT_DATA.md`` for the dense-vs-landmark packing/upsample
asymmetry that this recipe deliberately inherits unchanged.

Two things differ from the Qwen3 32k launchers, both forced by the architecture:

1. **Ulysses is the only parallelism axis besides DP.** ``GatedDeltaNet.apply_cp()`` rejects
   ring/zigzag CP and ``apply_tp()`` raises ``NotImplementedError``. Ulysses itself works fine --
   the "no Ulysses CP, incompatible with the GatedDeltaNet recurrence" comment in the
   ``Qwen3.5-4B-*-dolma3longmino.py`` CPT scripts is **wrong**; cp=16 was measured at 1M context on
   2026-08-01, and the old ``cp <= n_kv_heads`` cap was lifted by the KV replication in commit
   4cf4a38 (only ``cp | n_heads`` = 16 is required).
2. ``compile`` is off (GDN custom kernels), which also rules out ``budget`` activation
   checkpointing, so AC is ``full``. The LM head uses the fused-linear (Liger) loss to avoid
   materializing 248320-vocab logits for a whole window.

With CP=8 on 2 nodes the DP degree is 2 -- the same topology the Qwen3 32k launchers used -- so the
global batch (65,536 dense / 81,920 landmark) and the step counts (10,700 / 8,550) are identical to
the Qwen3 arms, not merely token-matched.
"""

from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
    LandmarkPackingStrategy,
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
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName, AttentionType
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

# ---------------------------------------------------------------------------
# Geometry / parallelism (shared)
# ---------------------------------------------------------------------------
MEM_FREQ = 63  # landmark arms: block_size = MEM_FREQ + 1 = 64, matching each CPT run
BLOCK_SIZE = MEM_FREQ + 1
LANDMARK_TOKEN_ID = 248200  # Qwen3.5 unused embedding row (vocab 248320); same id the CPT used

# Ulysses CP degree 8 on 2 nodes -> DP = 16/8 = 2, i.e. exactly the Qwen3 32k launchers' topology
# (2 nodes, CP=8, 2 DP replicas), so the global batch and step count match them token-for-token.
# Ulysses requires only `cp | n_heads` (=16 for Qwen3.5-4B) since the 2026-08-01 KV-replication fix.
CP_DEGREE = 8
NUM_NODES = 2
GPUS_PER_NODE = 8

# The sparse-landmark arm cannot run at CP=8. ``SparseLandmarkAttention.apply_cp()`` still calls the
# pre-2026-08-01 ``all_to_all_cp2hp``, whose ``h_out = h_in // world_size`` gives 0 when
# cp > n_kv_heads, so it (correctly) rejects any CP degree that does not divide n_kv_heads=4 --
# ``FastLandmarkAttention`` was migrated to the KV-replicating ``all_to_all_qkv_cp2hp`` in commit
# 4cf4a38 but ``landmark_sparse.py`` was missed. Until that migration lands, this arm runs CP=4 on
# ONE node, which still yields DP=2 and therefore the *same* global batch and step count as the
# other arms -- only the wall-clock differs.
_ARM_TOPOLOGY = {"sparse-landmark": dict(cp_degree=4, num_nodes=1)}

_CPT_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/amandab"

# ---------------------------------------------------------------------------
# Data (weka) -- the Qwen3.5 re-tokenization of the same ladders the Qwen3 32k runs used.
# Built by src/scripts/data/convert_unified_to_sft.py with --tokenizer Qwen/Qwen3.5-0.8B
# --eos-token-id 248044 --landmark-token-id 248200, reading the SAME source JSONL in the SAME order
# (see the per-task metadata.json in the Qwen3 shards for the exact inputs).
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2_qwen35"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
# nq: the p10 pipeline (hard-neg ~10% + cross-encoder gold filter) -- the only permitted NQ build.
NQ_DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10_qwen35/nq"
)
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"

# allenai/Dolci-Instruct-SFT tokenized with the Qwen3.5 chat template (already on weka).
DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen35"

# ---------------------------------------------------------------------------
# Top-level blend: 75% the 5-task mix / 25% Dolci-Instruct-SFT. No CPT source.
# ---------------------------------------------------------------------------
FIVE_TASK_FRAC = 0.75
DOLCI_FRAC = 0.25

LR = 1e-5

# ---------------------------------------------------------------------------
# Per-arm config. ``weights`` is the within-5-task upsampling; the dense arm keeps the
# contra-2.9 / oolong-1.3 compensation for the docs its 32768 packer drops, exactly as the Qwen3
# dense launchers do, while the landmark arms (which drop ~0% at 40960) keep the base 2.0/1.0.
# ``target_tokens`` matches the corresponding Qwen3 arm's budget; an arm may instead pin
# ``target_steps`` directly (used by the data-matched ablation arm, whose step count comes from a
# measured instance count rather than from a token budget).
# ---------------------------------------------------------------------------
_DENSE_WEIGHTS = {"contra": 2.9, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.3}
_LANDMARK_WEIGHTS = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}

# --- Data-vs-compute ablation geometry (the ``*-datamatch`` / ``*-tokenmatch`` arms) -------------
# 33344 = 521 blocks x 64: the smallest landmark window whose *content* capacity, 521 x 63 = 32823,
# is >= the dense arm's 32768. One landmark window therefore carries at least as much original
# content as one dense window, and the landmark-token overhead is paid in window width instead of
# in dropped content. It is not a power of 2 -- fine for LandmarkPackingInstanceSource (which only
# needs a multiple of the block size) but impossible for the dense PackingInstanceSource, whose
# SegmentTree requires one.
ABLATION_SEQ_LEN = 33344
# Instances the dense baseline's BFD packer produced from this exact mixture, measured in the
# q35-4b-dense-5task-dolci25-32k-nocpt run (Beaker 01KZAF94DPA971G2J7YCESFC74): 2,133,805,797
# content tokens -> 65,121 windows of 32768, averaging 1.2 padding tokens each. Its 10,700 steps x
# 2 windows = 21,400 instances = 32.86% of one epoch.
DENSE_BASELINE_INSTANCES = 65121
DENSE_BASELINE_STEPS = 10700

# Instances the *landmark* BFD packer produces from the same mixture at ABLATION_SEQ_LEN. Measure
# it with `launch_prep` and grep the log for "LandmarkPackingInstanceSource packed"; until it is
# set, the data-matched arm refuses to build rather than guess.
#
# Measured 2026-08-11 (Beaker 01KZS0DVQ3S5Q5GFBANRR34QKZ): 2,136,560,718 content tokens from
# 1,053,362/1,061,460 documents -> 66,083 windows, averaging 1012.5 non-content tokens each (3.04%
# of the window: 521 landmark tokens = 1.56%, the rest per-document block padding and the tail).
# That is only +1.48% instances over the dense arm's 65,121 -- the ~13% inflation reported in
# records/POSSIBLE_BUG_SFT_DATA.md was the next-fit packer, and BFD removes it. Content tokens are
# +0.13% because the landmark cut (content > 32,823) keeps a few documents the dense one (> 32,768)
# drops. The data-matched step count therefore comes out at 10,858, i.e. 3.26% more data *and*
# compute than the token-matched arm's 10,515 -- that 3.26% is the whole span of this ablation.
LANDMARK_ABLATION_INSTANCES: Optional[int] = 66083
_ARMS: Dict[str, Dict[str, Any]] = {
    "dense": dict(
        attn_type=None,
        sequence_length=32768,
        weights=_DENSE_WEIGHTS,
        target_tokens=10700 * 65536,  # the Qwen3 dense-dolci25 budget: 701.2M
        checkpoint=f"{_CPT_ROOT}/qwen35-4b-dense-longmino/step2385/model_and_optim",
    ),
    "fast-landmark": dict(
        attn_type=AttentionType.fast_landmark,
        sequence_length=40960,  # landmark-token space; 640 blocks of 64
        weights=_LANDMARK_WEIGHTS,
        target_tokens=8550 * 81920,  # the Qwen3 landmark budget: 700.4M
        checkpoint=f"{_CPT_ROOT}/qwen35-4b-fast-landmark-dolma3longmino/step2385/model_and_optim",
    ),
    "sparse-landmark": dict(
        attn_type=AttentionType.sparse_landmark,
        sequence_length=40960,
        weights=_LANDMARK_WEIGHTS,
        target_tokens=8550 * 81920,
        checkpoint=f"{_CPT_ROOT}/qwen35-4b-sparse-landmark-dolma3longmino/step2385/model_and_optim",
    ),
    # --- Data-vs-compute ablation, both vs. the `dense` arm above -------------------------------
    # Same architecture, same CPT checkpoint, same packing algorithm (BFD) and the same sampling
    # weights as `dense`; they differ from each other *only* in how long they run.
    "fast-landmark-datamatch": dict(
        attn_type=AttentionType.fast_landmark,
        sequence_length=ABLATION_SEQ_LEN,
        weights=_DENSE_WEIGHTS,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
        # Setting 1: consume the same *original data* as the baseline, i.e. the same fraction of
        # the same document stream -- the baseline's 21,400/65,121 = 32.86% of an epoch. The
        # landmark arm then spends slightly more tokens/compute for that same data.
        target_steps=(
            None
            if LANDMARK_ABLATION_INSTANCES is None
            else round(
                DENSE_BASELINE_STEPS * LANDMARK_ABLATION_INSTANCES / DENSE_BASELINE_INSTANCES
            )
        ),
        checkpoint=f"{_CPT_ROOT}/qwen35-4b-fast-landmark-dolma3longmino/step2385/model_and_optim",
    ),
    "fast-landmark-tokenmatch": dict(
        attn_type=AttentionType.fast_landmark,
        sequence_length=ABLATION_SEQ_LEN,
        weights=_DENSE_WEIGHTS,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
        # Setting 2: spend the same token budget as the baseline (701.2M window tokens), which at
        # a 66,688-token global batch is 10,515 steps -- and therefore see less original data.
        target_tokens=DENSE_BASELINE_STEPS * 65536,
        checkpoint=f"{_CPT_ROOT}/qwen35-4b-fast-landmark-dolma3longmino/step2385/model_and_optim",
    ),
}


def arm_geometry(arm: str) -> Dict[str, Any]:
    """
    Resolve an arm name to its full geometry, mix fractions, budget and CPT checkpoint.

    :param arm: One of ``"dense"``, ``"fast-landmark"``, ``"sparse-landmark"``.

    :returns: The resolved settings for the arm, including the derived ``global_batch_size``,
        ``max_steps`` and per-task top-level fractions.

    :raises KeyError: If ``arm`` is not a known arm.
    :raises ValueError: If the arm pins ``target_steps`` but it hasn't been measured yet.
    """
    spec = _ARMS[arm]
    seq_len = int(spec["sequence_length"])
    topo = _ARM_TOPOLOGY.get(arm, dict(cp_degree=CP_DEGREE, num_nodes=NUM_NODES))
    cp_degree = int(topo["cp_degree"])
    num_nodes = int(topo["num_nodes"])
    dp_degree = num_nodes * GPUS_PER_NODE // cp_degree
    global_batch_size = dp_degree * seq_len  # one window per DP replica per step, grad-accum 1
    if "target_steps" in spec:
        if spec["target_steps"] is None:
            raise ValueError(
                f"Arm '{arm}' pins its step count to a measured instance count, but "
                f"LANDMARK_ABLATION_INSTANCES is still None. Run\n"
                f"    python <launcher> launch_prep <run-name> ai2/jupiter "
                f"--launch.follow=false --launch.step_soft_timeout=null\n"
                f"and set it from the 'LandmarkPackingInstanceSource packed ...' log line."
            )
        max_steps = max(1, int(spec["target_steps"]))
    else:
        max_steps = max(1, round(int(spec["target_tokens"]) / global_batch_size))

    w = dict(spec["weights"])
    wsum = sum(w.values())
    nq_frac = w["nq"] / wsum
    oolong_frac = w["oolong"] / wsum
    rerank_frac = w["rerank"] / wsum
    outlier_frac = w["outlier"] / wsum
    contra_frac = max(0.0, 1.0 - (nq_frac + oolong_frac + rerank_frac + outlier_frac))

    return dict(
        arm=arm,
        attn_type=spec["attn_type"],
        is_landmark=spec["attn_type"] is not None,
        sequence_length=seq_len,
        cp_degree=cp_degree,
        num_nodes=num_nodes,
        dp_degree=dp_degree,
        shard_degree=dp_degree,  # shard params+grads+optim across all DP ranks
        global_batch_size=global_batch_size,
        max_steps=max_steps,
        packing_strategy=spec.get("packing_strategy", LandmarkPackingStrategy.next_fit),
        checkpoint=spec["checkpoint"],
        contra_frac=contra_frac,
        nq_frac=nq_frac,
        oolong_frac=oolong_frac,
        rerank_frac=rerank_frac,
        outlier_frac=outlier_frac,
    )


def build_qwen35_sft_experiment(cli_context: CliContext, *, arm: str) -> ExperimentConfig:
    """
    Build the SFT config for one Qwen3.5-4B arm.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :param arm: Which arm to build; see :func:`arm_geometry`.

    :returns: The full experiment config.
    """
    geom = arm_geometry(arm)
    seq_len = geom["sequence_length"]

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
        num_nodes=geom["num_nodes"],
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3_5()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3.5-4B hybrid (GDN + full attention, 3:1). No YaRN -- rejected for the hybrid, as in CPT.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    if geom["is_landmark"]:
        # Swap ONLY the full-attention layers, keeping their elementwise output gate (the landmark
        # attentions apply it, so w_g loads from the checkpoint) -- identical to the CPT scripts.
        attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
        attn_mixer.name = geom["attn_type"]
        attn_mixer.mem_freq = MEM_FREQ

    # Fused linear cross-entropy (Liger): without CP each rank computes the loss over its whole
    # window at once, so the 248320-vocab logits -- not params or activations -- are the OOM risk.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq_len,  # one full window per rank per micro-step
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        # GatedDeltaNet layers use custom kernels; compile stays off (as in CPT).
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=geom["shard_degree"],
        ),
        # Ulysses ONLY: GatedDeltaNet.apply_cp() rejects ring/zigzag CP, and GatedDeltaNet.apply_tp()
        # raises NotImplementedError, so this is the only parallelism axis besides DP.
        cp_config=TransformerContextParallelConfig.ulysses(degree=geom["cp_degree"]),
        # FULL activation checkpointing -- budget mode requires torch.compile, which is off here.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Two-way mixed document source: 5-task group + Dolci-Instruct-SFT (no CPT) ----
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
            source=_sft_source(CONTRA_DATA_ROOT),
            ratio=geom["contra_frac"],
            max_repetition_factor=8.0,
            label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(NQ_DATA_ROOT),
            ratio=geom["nq_frac"],
            max_repetition_factor=8.0,
            label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OOLONG_DATA_ROOT),
            ratio=geom["oolong_frac"],
            max_repetition_factor=8.0,
            label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(RERANK_DATA_ROOT),
            ratio=geom["rerank_frac"],
            max_repetition_factor=8.0,
            label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OUTLIER_DATA_ROOT),
            ratio=geom["outlier_frac"],
            max_repetition_factor=8.0,
            label="outlier",
        ),
    ]

    top_level_specs = [
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
    mixed_source = MixingDocumentSourceConfig(source_specs=top_level_specs)

    if geom["is_landmark"]:
        # Block-aligned greedy packing with per-document landmarks; the document boundaries it emits
        # are multiples of BLOCK_SIZE, which is what the landmark kernels require for packed masking.
        instance_source_config: Any = LandmarkPackingInstanceSourceConfig(
            source=mixed_source,
            sequence_length=seq_len,
            mem_freq=MEM_FREQ,
            mem_id=LANDMARK_TOKEN_ID,
            pad_id=tokenizer_config.pad_token_id,
            packing_strategy=geom["packing_strategy"],
        )
        # Doc boundaries come from the landmark packer, not from EOS scanning.
        generate_doc_lengths = False
        loader_tokenizer = tokenizer_config
    else:
        # Best-Fit-Decreasing packing of whole documents; documents longer than the window are
        # DROPPED rather than truncated (truncation would cut off the answer and leave a
        # fully-masked, NaN-loss window for the long-context tasks).
        instance_source_config = PackingInstanceSourceConfig(
            sources=[mixed_source],
            sequence_length=seq_len,
            tokenizer=doc_tokenizer_config,
            long_doc_strategy=LongDocStrategy.exclude,
        )
        generate_doc_lengths = True
        loader_tokenizer = doc_tokenizer_config

    # NOTE: Qwen3.5 ties bos == eos == 248044, so the loader must see bos_token_id=None for the
    # EOS-based document splitting to fire on every separator (see the Qwen3 launchers).
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=loader_tokenizer,
        work_dir=str(work_dir),
        global_batch_size=geom["global_batch_size"],
        seed=34521,
        num_workers=4,
        generate_doc_lengths=generate_doc_lengths,
    )

    max_steps = geom["max_steps"]
    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=geom["checkpoint"],
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(max_steps),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100000,
                ephemeral_save_interval=max_steps,
                max_checkpoints=2,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",  # prasanns-allen-institute-for-ai 403s for amandab's launches
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
