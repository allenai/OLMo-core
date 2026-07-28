"""
256k-context Beaker/gantry SFT of the Qwen3.5-4B **FAST-COMPRESSIVE-LANDMARK** 256k CPT model
(block 64, ``mem_freq=63``) on 75% the 2k-256k 5-task mix / 25% ``allenai/Dolci-Instruct-SFT``.

Paired with ``Qwen3.5-4B-dense-xlong5-dolci25-256k-SFT.py``: same data, same budget, same
parallelism, so the landmark-vs-dense comparison isolates the architecture.

Data (all Qwen3.5-tokenized -- these models do NOT share the Qwen3 vocabulary):

  * 5-task mix, 75%, from ``xlong5_2k256k_qwen35/shards_full`` (the ``--no-doc-markers`` "standard"
    arm; the ``shards_chunked`` sibling carries ``<|box_start|>``/``<|box_end|>`` boundaries and is
    for the chunked arm, NOT this one). ~1.76B tokens, ~20k instances/task on a 2k-256k ladder
    skewed short. Within-mix weighting is the canonical 5-task one, unchanged from the 32k runs:
    contra 2x / rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x.
  * ``allenai/Dolci-Instruct-SFT``, 25%, re-tokenized to Qwen3.5 by
    ``src/scripts/data/convert_dolci_instruct_sft.py --tokenizer Qwen/Qwen3.5-0.8B
    --eos-token-id 248044 --landmark-token-id 248200``. The pre-existing
    ``dolci-instruct-sft/qwen3`` tree is a *different vocabulary* and must not be used here.

LONG-DOC ASYMMETRY vs. the dense arm -- worth knowing before reading the eval deltas. Landmark
packing spends one slot in 64 on a landmark token, so a 262,144-token window holds 4096 x 63 =
258,048 *content* tokens, while the dense arm's window holds all 262,144. Documents above the
content capacity are dropped. The 5-task shards were built against a 262,144 cap and their longest
examples sit just under it (max 262,072), so the landmark arm loses the handful of instances in
(258,048, 262,072] -- well under 0.2% of instances, but concentrated in exactly the 128-256k bucket
the runs are meant to test. Left as a drop rather than truncation because truncating a long-context
example cuts off the trailing answer and leaves a fully-masked, NaN-loss window.

Model/geometry: ``AttentionType.fast_compressive_landmark`` swapped into the full-attention blocks
of the GDN hybrid, ``mem_freq=63``, no ``nonselected_landmark_mass`` override -- exactly as the CPT
script left it, so weight loading stays strict.

    PYTHONPATH=src python src/scripts/train/sft/Qwen3.5-4B-fast-compressive-landmark-xlong5-dolci25-256k-SFT.py \\
        dry_run q35-4b-fastcomplm-xlong5-dolci25-256k ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/sft/Qwen3.5-4B-fast-compressive-landmark-xlong5-dolci25-256k-SFT.py \\
        launch  q35-4b-fastcomplm-xlong5-dolci25-256k ai2/jupiter-cirrascale-2
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig, OLMoCoreBeakerImage
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
# Landmark geometry -- identical to the CPT run this initializes from.
# ---------------------------------------------------------------------------
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 262144  # 256k in landmark-token space; 262144 / 64 = 4096 blocks
CONTENT_CAPACITY = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 258048 content tokens per window
LANDMARK_TOKEN_ID = 248200  # Qwen3.5 unused embedding row (vocab 248320)

# ---------------------------------------------------------------------------
# Parallelism -- architecture-forced, and deliberately byte-for-byte the CPT run's configuration
# (the only setting at 256k that is *measured* to fit in 80GB):
#   * TP=1     -- GatedDeltaNet.apply_tp raises NotImplementedError.
#   * Ulysses  -- ring CP is rejected by both GatedDeltaNet and landmark attention.
#   * CP<=4    -- Ulysses needs cp_degree to divide n_kv_heads, and qwen3_5_4B has n_kv_heads=4.
# ---------------------------------------------------------------------------
CP_DEGREE = 4
NUM_NODES = 8  # 8 x 8 = 64 GPUs -> DP = 64 / 4 = 16
SHARD_DEGREE = 16  # shard params+optim across all DP ranks, as the CPT run did

# ---------------------------------------------------------------------------
# Data (weka). All Qwen3.5-tokenized; EOS/BOS/pad 248044.
# ---------------------------------------------------------------------------
DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/shards_full"
)
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction_train"
NQ_DATA_ROOT = f"{DATA_ROOT}/nq_train"  # built from the p10 pool, not the banned 98%-hard build
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong_train"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank_train"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier_train"

DOLCI_DATA_ROOT = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen35"

# Fast-compressive-landmark 256k CPT base (model+optim), loaded weights-only. The '-fix' run is the
# one that reached the full 2385 steps; the earlier q35-4b-fastcomplm-256k stalled at step1250.
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "q35-4b-fastcomplm-256k-fix/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions WITHIN the 5-task group (sum to 1.0): contra 2x / rerank 1.5x / outlier 1.5x /
# nq 1x / oolong 1x -- the canonical 5-task weighting, carried over unchanged from the 32k runs.
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
# Optimization / budget. LR and weight decay follow the 32k SFT runs (1e-5, wd 0), not the CPT's
# 3.2e-4 / 0.1.
#
# ~1 epoch: the 5-task shards hold 1.764B tokens, so at FIVE_TASK_FRAC the whole mix is
# 1.764B / 0.75 = 2.35B tokens. 560 steps x 16 DP windows x 262144 = 2.35B. The 5-task side is what
# binds; the Qwen3.5 Dolci build is larger than its 588M share, so it is subsampled rather than
# repeated and max_repetition_factor never comes into play.
# ---------------------------------------------------------------------------
LR = 1e-5
TARGET_STEPS = 560
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * (NUM_NODES * 8 // CP_DEGREE)  # 16 windows/step, grad-accum 1
MAX_STEPS = TARGET_STEPS


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
    # every EOS a boundary.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # The backend is inert here: the full-attention blocks are swapped to the landmark mixer, which
    # runs its own fused Triton kernel and never touches self.backend, and the GDN blocks never use
    # a flash backend either. Left at flash_2 to match the CPT script.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
    attn_mixer.name = AttentionType.fast_compressive_landmark
    attn_mixer.mem_freq = MEM_FREQ
    # Keep attn_mixer.gate (the elementwise gate from qwen3_5_4B): landmark attention applies it, so
    # w_g loads straight from the CPT checkpoint.

    # Mandatory at 256k: with CP=4 each rank holds 65,536 tokens, and dense logits over a 248,320
    # vocab would be tens of GB in bf16 before cross-entropy upcasts them.
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
        # Landmark attention performs its own cp2hp/hp2cp all-to-all inside forward(); it requires
        # Ulysses and rejects ring CP outright.
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

    # PACKED with intra-document masking: block-aligned greedy packing + per-doc landmarks. The
    # compressive fused kernel supports cu_doc_lens (DOC_MASK), so packing buys efficiency without
    # giving up proper intra-doc masking.
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=MixingDocumentSourceConfig(source_specs=specs),
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # Block-aligned doc boundaries come from LandmarkPackingInstanceSource (-> cu_doc_lens ->
        # DOC_MASK), so the loader must not re-derive them from EOS positions.
        generate_doc_lengths=False,
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
