"""Shared builder for Qwen3 shared-vector SFT at matched 32k content capacity.

Contrast: sharedvec-dolci25-tokenmatch versus the saved dense Dolci25 step10700
recipe, matching 75/25 mixture, task sources/weights, optimizer, YaRN2, and 701.2M
window-token budget. The sharedvec-tokenmatch arm retains the earlier pure-five-task
pilot with native RoPE.

Readout: v3 five-task and OOD ladders, task-native metrics (F1, NDCG@10, OOLONG score).

Not matched: attention architecture and architecture-specific CPT weights; CPT budget
provenance is audited before launch. Landmark packing includes landmark and block-padding
slots, so equal window-token budgets consume less original content, potentially biasing
shared-vector downward. Budget matching is not a claim of equal FLOPs. The Dolci25 arm uses four nodes at CP8 / DP4 per user request (min runtime zero):
133376 tokens/update and 5258 updates, versus dense 65536/update and 10700 updates.
Thus the optimizer batch is deliberately larger; its effect is not controlled.
HSDP shards across all four DP replicas for memory; dense used shard_degree=1. The 33344 landmark window has 32823
content capacity versus dense 32768, a 0.17% threshold difference. Do not call this data-matched.
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
from olmo_core.internal.experiment import CliContext, ExperimentConfig
from olmo_core.launch.beaker import (
    BeakerEnvVar,
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
)
from olmo_core.nn.rope import YaRNRoPEScalingConfig
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
DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen3"
_ARMS = {
    "sharedvec-tokenmatch": {"dolci_fraction": 0.0, "yarn_factor": None, "num_nodes": 2},
    "sharedvec-dolci25-tokenmatch": {"dolci_fraction": 0.25, "yarn_factor": 2.0, "num_nodes": 4},
}

LR = 1e-5
GLOBAL_BATCH_SIZE = DP_DEGREE * SEQUENCE_LENGTH  # 66,688 window tokens/step (grad-accum 1)
# The canonical 5-task no-CPT budget: the dense arms' 10,700 steps x 65,536 = 701,235,200 window
# tokens (the 40960 landmark arms' 8,550 x 81,920 = 700,416,000 is the same budget to 0.12%).
TARGET_TOKENS = 10_700 * 65_536
MAX_STEPS = max(1, round(TARGET_TOKENS / GLOBAL_BATCH_SIZE))  # 10,515
# Upper bound on original content slots (before document block padding): CONTENT_CAPACITY * DP_DEGREE * MAX_STEPS
#   = 32,823 x 2 x 10,515 = 690,267,690  (a dense 32768/BFD arm on the same budget: 701,235,200)
# The upper bound is ~1.56% below dense; realized non-padding content must be measured.
CONTENT_TOKENS = CONTENT_CAPACITY * DP_DEGREE * MAX_STEPS


def build_experiment_config(cli_context: CliContext, *, arm: str) -> ExperimentConfig:
    """
    Build the shared-vector-landmark 5-task SFT experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
    arm_config = _ARMS[arm]
    num_nodes = arm_config["num_nodes"]
    dp_degree = num_nodes * GPUS_PER_NODE // CP_DEGREE
    global_batch_size = dp_degree * SEQUENCE_LENGTH
    max_steps = round(TARGET_TOKENS / global_batch_size)
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
        num_nodes=num_nodes,
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"
        beaker_launch_config.post_setup = (
            "python -m pip install --quiet --no-deps "
            "flash-linear-attention==0.4.2 fla-core==0.4.2 && "
            "python src/scripts/ctc_eval/preflight/verify_fla.py"
        )
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

    if arm_config["yarn_factor"] is not None:
        model_config = model_config.with_rope_scaling(
            YaRNRoPEScalingConfig(
                factor=arm_config["yarn_factor"],
                beta_fast=32,
                beta_slow=1,
                old_context_len=32768,
            )
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
            shard_degree=dp_degree,  # shard params+grads+optim across all DP ranks
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

    # Five-task leaf sources; optional Dolci blend is applied below.
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

    mixed_source = MixingDocumentSourceConfig(source_specs=specs)
    dolci_fraction = arm_config["dolci_fraction"]
    if dolci_fraction:
        mixed_source = MixingDocumentSourceConfig(
            source_specs=[
                MixingDocumentSourceSpecConfig(
                    source=mixed_source,
                    ratio=1.0 - dolci_fraction,
                    label="five_task_mix",
                ),
                MixingDocumentSourceSpecConfig(
                    source=_sft_source(DOLCI_DATA_ROOT),
                    ratio=dolci_fraction,
                    max_repetition_factor=8.0,
                    label="dolci_instruct_sft",
                ),
            ]
        )

    # Block-aligned best-fit-decreasing packing with per-document landmarks; doc boundaries reach
    # the fused landmark kernel as cu_doc_lens -> DOC_MASK.
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=mixed_source,
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=global_batch_size,
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
            max_duration=Duration.steps(max_steps),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=max_steps,
                ephemeral_save_interval=1000,
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
