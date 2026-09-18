"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B **SHARED-VECTOR
LANDMARK** CPT model on the 5 long-context tasks (contradiction, nq, oolong, rerank, outlier).
No-CPT variant: the SFT budget is the whole mix, no continued-pretraining text.

**The contrast.** This is a PILOT, not a controlled arm. It is the first SFT of the shared-vector
landmark mixer, taken from the finished ``qwen3-4b-shared-vec-landmark-d3lm15b`` CPT run (10B tokens
on the dolma3_longmino 15B sample, Beaker ``01KWJGE5MKACTD5524S76GX0XM``, ``step2385``). It runs at
the *corrected* window/packing/weighting geometry (§2, §3, §5 of the create-sft-config skill), which
no Qwen3 arm has been trained at yet.

**It therefore has no valid Qwen3 counterpart.** The existing Qwen3 5-task no-CPT arms --
``Qwen3-4B-dense-5task-32k-nocpt-SFT.py``, ``Qwen3-4B-fast-landmark-5task-32k-nocpt-SFT.py``,
``Qwen3-4B-compressive-5task-32k-nocpt-SFT.py`` -- all ran at ``sequence_length=40960`` with
next-fit packing and the ``2.0/1.0`` landmark weights. Plotting this run against them would confound
the mixer against three data-side axes at once (window width, packing algorithm, sampling weights).
To turn this pilot into a comparison, a ``fast-landmark`` arm at this same 33344/BFD/dense-weights
geometry (and, for the dense side, a 32768/BFD/dense-weights arm) has to be run alongside it.

**The readout.** Per-task f1 on the 5-task ladder evals (contradiction, nq, oolong, rerank, outlier)
at rungs up to 32k, from the final checkpoint. Contradiction's held-out set is 488 examples (the
whole file), so quote it as ``eval_size=488`` with its binomial SE (+/-0.021 at f1~0.70).

**What is deliberately not matched.**

* *Token-matched, not data-matched.* The budget is the canonical 5-task 701.2M window tokens
  (10,700 x 65,536, the dense Qwen3 budget), which at this arm's 66,688-token global batch is 10,515
  steps -- the same setting as the Qwen3.5 ``fast-landmark-tokenmatch`` arm. A landmark window's
  content capacity is 32,823 of 33,344 slots, so a token-matched landmark arm consumes slightly
  *less original data* than a dense arm on the same budget. On the Qwen3.5 measurement that gap was
  3.26%, biasing this arm **downward** relative to a data-matched one. A data-matched sibling would
  need ``launch_prep`` on this (Qwen3-tokenized) mixture to measure the packed instance count; the
  Qwen3.5 number does not transfer, since the two tokenizers give different per-document lengths.
* *No Dolci.* The Qwen3 5-task no-CPT family is the 5 tasks only; the 75/25 Dolci blend belongs to
  the Qwen3.5 ``dolci25`` family. Kept as-is so the data recipe stays in the Qwen3 lineage.

Geometry notes (see the create-sft-config skill for the full rationale):

* ``SEQUENCE_LENGTH = 33344`` = 521 blocks x 64. Content capacity 521 x 63 = 32,823 >= a dense arm's
  32,768, so one landmark window carries at least as much original content as one dense window and
  the landmark-token overhead is paid in window width rather than in dropped documents. Not a power
  of two -- fine for ``LandmarkPackingInstanceSource`` (needs only a multiple of the block size),
  impossible for the dense ``PackingInstanceSource`` (its ``SegmentTree`` requires one).
* ``packing_strategy=best_fit_decreasing``, the same algorithm a dense arm's packer uses. The
  default next-fit inflated the landmark instance count by ~13%; BFD leaves ~1.5%, which is the
  genuine landmark cost (each document ceil'd to a whole number of blocks).
* ``_DENSE_WEIGHTS`` (contra 2.9 / oolong 1.3). Those numbers compensate for the over-long documents
  the packer drops at a ~32.8k threshold; at 33,344 the landmark cut (content > 32,823) is within
  0.17% of the dense one (32,768), so the same compensation applies. The ``2.0/1.0`` set in the
  older Qwen3 landmark scripts belongs to the abandoned 40960 recipe and is NOT carried forward.

This is a standalone file rather than an arm in ``_qwen35_5task_dolci25_32k_nocpt_common.py``: that
builder is Qwen3.5-specific (248320 vocab, GDN hybrid, Qwen3.5-tokenized ladders, Dolci 25%) and
this checkpoint is Qwen3-4B. If a second Qwen3 arm lands at this geometry, factor the two into a
``_qwen3_5task_33344_common.py`` builder rather than forking again.

    PYTHONPATH=src python src/scripts/train/memexpress/sft_5task/Qwen3-4B-sharedvec-5task-33344-tokenmatch-SFT.py \\
        dry_run q4b-sharedvec-5task-33344-tokenmatch ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/memexpress/sft_5task/Qwen3-4B-sharedvec-5task-33344-tokenmatch-SFT.py \\
        launch  q4b-sharedvec-5task-33344-tokenmatch ai2/jupiter-cirrascale-2 \\
        --launch.follow=false --launch.step_soft_timeout=null
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional

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
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig, OLMoCoreBeakerImage
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
# Landmark geometry -- must match the CPT run that produced BASE_CHECKPOINT.
# Qwen3-4B-base-shared-vector-landmark-dolma3longmino.py: mem_freq=63, block 64, vec_dim=32,
# landmark token id 151860. A mismatch mis-places landmarks against the kernel's block tiling and
# does not raise.
# ---------------------------------------------------------------------------
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token
VEC_DIM = 32  # length of the learned per-block vector appended to each value

# 521 blocks x 64 = 33344 landmark-space slots; 521 x 63 = 32823 content slots >= a dense 32768.
N_BLOCKS = 521
SEQUENCE_LENGTH = N_BLOCKS * BLOCK_SIZE  # 33344
CONTENT_CAPACITY = N_BLOCKS * MEM_FREQ  # 32823
assert SEQUENCE_LENGTH % BLOCK_SIZE == 0
assert CONTENT_CAPACITY >= 32768, "landmark content capacity must clear the dense 32768 window"

# Ulysses CP degree 8. Qwen3-4B: n_heads=32, n_kv_heads=8 -> CP=8 divides both.
CP_DEGREE = 8
NUM_NODES = 2
GPUS_PER_NODE = 8
DP_DEGREE = NUM_NODES * GPUS_PER_NODE // CP_DEGREE  # 2

# ---------------------------------------------------------------------------
# Data (weka) -- Qwen3-tokenized single-task ladders.
# NQ comes from the p10 build (hard-neg ~10% + cross-encoder gold filter), the only permitted NQ;
# the `*/nq` under single_task_ladders_v2 is the old 98%-hard-negative build.
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
NQ_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"

# Shared-vector-landmark CPT base (Beaker 01KWJGE5MKACTD5524S76GX0XM, exit 0, final step 2385).
# The checkpoint already contains the shared-vector params (w_out_vec, weight_landmark, base), so
# this is a normal STRICT load -- no state_dict_load_opts override. Weights only.
#
# NOTE the ``amandab/`` namespace. The CPT script wrote to ``{root}/checkpoints/{run_name}`` (no
# user segment) and the run directory was moved under ``amandab/`` afterwards, so the path in the
# CPT job log is stale -- the same correction ``Qwen3-4B-fast-landmark-5task-32k-nocpt-SFT.py``
# records as its fix (b). Verified 2026-09-18 (Beaker 01M2TQ7PXNN3GD3AN6W3MH2NAF): step1000,
# step2000 and step2385 are present, each with config.json / model_and_optim / train.
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/qwen3-4b-shared-vec-landmark-d3lm15b/"
    "step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Sampling weights: the DENSE compensation set (contra 2.9 / oolong 1.3), not the 2.0/1.0 that the
# older 40960 landmark scripts used. See the module docstring.
# ---------------------------------------------------------------------------
_DENSE_WEIGHTS = {"contra": 2.9, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.3}
_WSUM = sum(_DENSE_WEIGHTS.values())
NQ_FRAC = _DENSE_WEIGHTS["nq"] / _WSUM
OOLONG_FRAC = _DENSE_WEIGHTS["oolong"] / _WSUM
RERANK_FRAC = _DENSE_WEIGHTS["rerank"] / _WSUM
OUTLIER_FRAC = _DENSE_WEIGHTS["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, 1.0 - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

# ---------------------------------------------------------------------------
# Optimization / budget -- TOKEN-matched to the canonical 5-task budget.
# ---------------------------------------------------------------------------
LR = 1e-5
GLOBAL_BATCH_SIZE = DP_DEGREE * SEQUENCE_LENGTH  # 66,688 window tokens/step (grad-accum 1)
# The canonical 5-task no-CPT budget: the dense arms' 10,700 steps x 65,536 = 701,235,200 window
# tokens (the 40960 landmark arms' 8,550 x 81,920 = 700,416,000 is the same budget to 0.12%).
TARGET_TOKENS = 10_700 * 65_536
MAX_STEPS = max(1, round(TARGET_TOKENS / GLOBAL_BATCH_SIZE))  # 10,515
# Content tokens actually seen: CONTENT_CAPACITY * DP_DEGREE * MAX_STEPS
#   = 32,823 x 2 x 10,515 = 690,267,690  (a dense 32768/BFD arm on the same budget: 701,235,200)
# i.e. this token-matched arm sees ~1.56% fewer content tokens, the landmark-token overhead.
CONTENT_TOKENS = CONTENT_CAPACITY * DP_DEGREE * MAX_STEPS


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the shared-vector-landmark 5-task SFT experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
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
        beaker_launch_config.priority = "high"
        # The shared-vector tail materializes several large fp32 (T, nb) tensors; reduce allocator
        # fragmentation so those transient spikes can reuse freed segments (same mitigation as the
        # CPT script -- see the sharedvec-tail-64k-oom note).
        beaker_launch_config.env_vars.append(
            BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True")
        )

    tokenizer_config = TokenizerConfig.qwen3()
    # Qwen3's bos == eos, and the EOS-followed-by-BOS boundary detector never fires on EOS-separated
    # SFT shards; LandmarkPackingInstanceSource owns the boundaries via cu_doc_lens instead.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3-4B with SHARED-VECTOR LANDMARK attention: the fast-landmark weights plus a learned
    # per-block positional vector of length VEC_DIM appended to each value before aggregation (see
    # olmo_core/nn/attention/landmark_shared_vector.py). landmark_use_kernel=True selects the fused
    # Triton kernel for the head_dim output; the factory default of False falls back to the eager
    # O(T^2) dense path, which OOMs at this window.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        shared_vector_landmark=True,
        mem_freq=MEM_FREQ,
        vec_dim=VEC_DIM,
        landmark_use_kernel=True,
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
            shard_degree=DP_DEGREE,  # shard params+grads+optim across all DP ranks
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

    # ---- 5-way mixed document source (no CPT text, no Dolci) ----
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

    # Block-aligned best-fit-decreasing packing with per-document landmarks; doc boundaries reach
    # the fused landmark kernel as cu_doc_lens -> DOC_MASK.
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=MixingDocumentSourceConfig(source_specs=specs),
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,  # the 5-task family seed, so arms draw the same stream
        num_workers=4,
        generate_doc_lengths=False,  # the landmark packing source owns cu_doc_lens
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
                entity="ai2-llm",  # prasanns-... 403s for amandab's launches
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
