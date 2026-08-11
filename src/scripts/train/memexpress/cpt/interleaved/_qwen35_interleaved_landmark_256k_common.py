"""
Shared builder for the **Qwen3.5-4B interleaved sparse/regular landmark** CPT arms at 256k context
on the longmino-512k mix.

Qwen3.5-4B is a hybrid: ``block_pattern = [gdn, gdn, gdn, attn]`` over 32 layers, so exactly 8
layers are full attention (global indices 3, 7, 11, 15, 19, 23, 27, 31). Every arm here replaces
*all* of those 8 with a landmark variant; the arms differ only in **which** variant each one gets:

  * ``reg`` (:data:`REG_LANDMARK_TYPE`, ``fast_compressive_landmark``) -- the compressive landmark
    grouped softmax. Every query still sees every past key, the landmark token supplies a per-block
    gate, and each block's landmark *value* is folded in as a compressed summary. Cost is quadratic:
    dense attention plus ~T^2/64 of landmark scoring. This is the same variant as the 8-node
    ``Qwen3.5-4B-fast-compressive-landmark-longmino512k.py`` run, which is therefore the
    all-regular control for this sweep.
  * ``sparse`` (:data:`AttentionType.sparse_landmark`) -- a query attends fully inside its own
    64-token chunk and sees *all* past chunks only through their landmark tokens. Sub-quadratic:
    ~2,080 keys per query at 256k versus ~131,072 for the dense/regular path.

The four arms:

===============  =============================================================================
Arm              Per-full-attention-layer assignment (8 entries, layers 3/7/.../31)
===============  =============================================================================
``reg-first``    ``reg`` on the FIRST full-attention layer, ``sparse`` on the other 7
``reg-last``     ``sparse`` on the first 7, ``reg`` on the LAST full-attention layer
``sparse-reg``   alternating starting sparse:  sparse, reg, sparse, reg, ... (ends ``reg``)
``reg-sparse``   alternating starting reg:     reg, sparse, reg, sparse, ... (ends ``sparse``)
===============  =============================================================================

``sparse-reg`` and ``reg-sparse`` are the same 4/4 alternation in opposite phase, so together they
separate "how many regular-landmark layers" from "where in the stack they sit"; ``reg-first`` /
``reg-last`` do the same at the 1/7 extreme.

Mechanism
---------
The assignment is expressed as an :class:`AttentionTypePatternConfig` whose ``pattern`` is exactly
:data:`N_LAYERS` long, so ``get_type(layer_idx, n_layers)`` indexes it directly with the **global**
layer index (``Transformer.__init__`` passes ``block_idx`` straight through to
``AttentionConfig.build``). The 24 GDN slots are filled with :data:`AttentionType.default` purely as
a placeholder -- those layers are :class:`GatedDeltaNetConfig` blocks and never consult this pattern.
A shorter, period-4 pattern would *not* work: every full-attention layer sits at ``idx % 4 == 3``
and would therefore receive the same entry.

Constraints, all forced by the architecture rather than chosen
--------------------------------------------------------------
* **TP = 1** -- ``GatedDeltaNet.apply_tp()`` raises ``NotImplementedError``.
* **Ulysses only** -- ring/zigzag CP is rejected by GDN *and* by every landmark variant.
* **CP <= 4** -- ``FastLandmarkAttention`` accepts any CP degree dividing ``n_heads`` (=16) since the
  KV-replication fix in 4cf4a38, but ``SparseLandmarkAttention.apply_cp()`` still requires
  ``cp | n_kv_heads`` (=4). Every arm here contains sparse layers, so the *model* caps at CP=4.
* **compile off** -- GDN custom kernels; this also rules out ``budget`` activation checkpointing, so
  AC is ``full``.
* **fused-linear (Liger) LM loss** -- mandatory, not an optimization: at CP=4 each rank holds 65,536
  tokens and dense logits would be 65,536 x 248,320 x 2B = ~32.5 GB in bf16 before the fp32 upcast.

On 2 nodes that gives cp=4, tp=1, **dp=4**. The global batch is held at 16 sequences (~4.2M model
tokens) to match the 8-node dense/landmark 256k runs token-for-token, which costs 4 gradient
accumulation micro-steps per rank instead of 1.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig
from olmo_core.launch.beaker import (
    BeakerEnvVar,
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
)
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionType,
    AttentionTypePatternConfig,
)
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

# ``longmino_512k_mix`` lives in src/scripts/data; this file is four levels below src/scripts.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data"),
)

from longmino_512k_mix import build_longmino_512k_mix  # noqa: E402

# ---------------------------------------------------------------------------
# Landmark geometry (shared with every other Qwen3.5 landmark run)
# ---------------------------------------------------------------------------
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 262144  # 256k model tokens (262144 / 64 = 4096 blocks)
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 258048
LANDMARK_TOKEN_ID = 248200  # Qwen3.5 unused embedding row (vocab 248320)

#: Which landmark variant plays the "regular landmark" role. ``fast_compressive_landmark`` is the one
#: used by ``Qwen3.5-4B-fast-compressive-landmark-longmino512k.py``, so that 8-node run is the
#: token- and step-matched all-regular control for this sweep. Switch to
#: :data:`AttentionType.fast_landmark` for the plain (non-compressive) landmark grouped softmax.
REG_LANDMARK_TYPE = AttentionType.fast_compressive_landmark
SPARSE_LANDMARK_TYPE = AttentionType.sparse_landmark

# ---------------------------------------------------------------------------
# Model layout: qwen3_5_4B is 32 layers of [gdn, gdn, gdn, attn]
# ---------------------------------------------------------------------------
N_LAYERS = 32
BLOCK_PATTERN_PERIOD = 4
#: Global layer indices of the full-attention blocks: [3, 7, 11, 15, 19, 23, 27, 31].
ATTN_LAYER_INDICES = list(range(BLOCK_PATTERN_PERIOD - 1, N_LAYERS, BLOCK_PATTERN_PERIOD))
N_ATTN_LAYERS = len(ATTN_LAYER_INDICES)  # 8

# ---------------------------------------------------------------------------
# Parallelism / budget (shared)
# ---------------------------------------------------------------------------
CP_DEGREE = 4  # capped by SparseLandmarkAttention: cp must divide n_kv_heads=4
NUM_NODES = 2
GPUS_PER_NODE = 8
DP_DEGREE = NUM_NODES * GPUS_PER_NODE // CP_DEGREE  # 4

#: 16 sequences per optimizer step (~4.2M model tokens), matching the 8-node 256k runs. At DP=4 that
#: is 4 gradient-accumulation micro-steps per rank.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 16
MAX_TOKENS = 10_000_000_000  # 10B -> ~2384 steps
LR = 3.2e-4
WARMUP_STEPS = 400

DATA_ROOT = "/weka/oe-training-default/amandab/longmino_512k"

#: The converted Qwen3.5-4B olmo-core base. Override with ``--trainer.load_path=``.
CHECKPOINT_PATH = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3.5-4B-olmocore/model_and_optim"
)


def _alternating(first: AttentionType, second: AttentionType) -> List[AttentionType]:
    return [first if i % 2 == 0 else second for i in range(N_ATTN_LAYERS)]


#: Per-arm assignment over the 8 full-attention layers, in stack order.
ARMS: Dict[str, List[AttentionType]] = {
    "reg-first": [REG_LANDMARK_TYPE] + [SPARSE_LANDMARK_TYPE] * (N_ATTN_LAYERS - 1),
    "reg-last": [SPARSE_LANDMARK_TYPE] * (N_ATTN_LAYERS - 1) + [REG_LANDMARK_TYPE],
    "sparse-reg": _alternating(SPARSE_LANDMARK_TYPE, REG_LANDMARK_TYPE),
    "reg-sparse": _alternating(REG_LANDMARK_TYPE, SPARSE_LANDMARK_TYPE),
}


def build_layer_types(arm: str) -> AttentionTypePatternConfig:
    """
    Expand an arm's per-full-attention-layer assignment into the global length-32 pattern.

    :param arm: One of the keys of :data:`ARMS`.

    :returns: A pattern config whose ``pattern`` is exactly :data:`N_LAYERS` entries long, so that
        :meth:`AttentionTypePatternConfig.get_type` indexes it with the global layer index.

    :raises OLMoConfigurationError: If ``arm`` is not a known arm.
    """
    if arm not in ARMS:
        raise OLMoConfigurationError(f"Unknown arm {arm!r} (expected one of {sorted(ARMS)})")
    per_attn_layer = ARMS[arm]
    assert len(per_attn_layer) == N_ATTN_LAYERS
    # GDN slots are placeholders: those blocks are GatedDeltaNet and never read this pattern.
    pattern: List[AttentionType] = [AttentionType.default] * N_LAYERS
    for layer_idx, attn_type in zip(ATTN_LAYER_INDICES, per_attn_layer):
        pattern[layer_idx] = attn_type
    return AttentionTypePatternConfig(pattern=pattern)


def describe_arm(arm: str) -> str:
    """
    A one-line human-readable summary of an arm's layer assignment, for the run log.

    :param arm: One of the keys of :data:`ARMS`.

    :returns: e.g. ``"layers 3,7,...,31 -> reg,sparse,sparse,..."``.
    """
    short = {REG_LANDMARK_TYPE: "reg", SPARSE_LANDMARK_TYPE: "sparse"}
    return "attn layers {} -> {}".format(
        ",".join(str(i) for i in ATTN_LAYER_INDICES),
        ",".join(short.get(t, str(t)) for t in ARMS[arm]),
    )


def build_qwen35_interleaved_experiment(cli_context: CliContext, *, arm: str) -> ExperimentConfig:
    """
    Build the experiment config for one interleaved sparse/regular landmark CPT arm.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :param arm: One of the keys of :data:`ARMS`.

    :returns: The full experiment config.
    """
    layer_types = build_layer_types(arm)

    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

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
        # At this context length the allocator fragments badly (the 256k dense run first OOMed with
        # 58.2 GiB allocated but 15.5 GiB reserved-and-unusable). Expandable segments lets it grow
        # existing segments instead of stranding them. Both spellings: torch 2.9 renamed this to
        # PYTORCH_ALLOC_CONF and warns on the old name, but older images only honour the old one.
        for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            beaker_launch_config.env_vars.append(
                BeakerEnvVar(name=_var, value="expandable_segments:True")
            )

    tokenizer_config = TokenizerConfig.qwen3_5()

    # The backend is inert here: every full-attention block resolves to a landmark variant, and each
    # of those runs its own Triton kernel rather than routing through ``self.backend``. The GDN
    # blocks never touch a flash backend either. Left at flash_2 so nothing implies an FA3
    # dependency that does not exist.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    if model_config.n_layers != N_LAYERS:
        raise OLMoConfigurationError(
            f"the layer-type pattern is built for exactly {N_LAYERS} layers but the model has "
            f"{model_config.n_layers}"
        )
    attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
    # ``name`` is only the fallback; ``layer_types`` overrides it per layer.
    attn_mixer.name = REG_LANDMARK_TYPE
    attn_mixer.layer_types = layer_types
    attn_mixer.mem_freq = MEM_FREQ
    attn_mixer.num_landmarks = 1  # matches LandmarkInstanceSource's 1-landmark-per-block data
    # Keep attn_mixer.gate (the elementwise gate from qwen3_5_4B): both landmark variants apply it,
    # so gated attention is preserved and w_g loads straight from the converted checkpoint.

    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one full sequence per DP rank, split across CP
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=WARMUP_STEPS, alpha_f=0.0),
        # GatedDeltaNet custom kernels; compile off, which also rules out 'budget' AC.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            # Plain FSDP rather than HSDP: the DP mesh is only 4 ranks wide here, so full sharding
            # across it is the maximum available and an HSDP replicate dim of 1 buys nothing.
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        # Landmark attention performs its own cp2hp/hp2cp all-to-all inside forward(); it requires
        # Ulysses and rejects ring CP outright.
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Per-stratum longmino-512k mix (qwen35 tree) at the *content* length, then landmark insertion
    # brings each instance up to SEQUENCE_LENGTH:
    #   MixingInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=258048)
    #     -> LandmarkInstanceSource (one landmark every MEM_FREQ tokens -> 262144)
    # Both landmark variants consume them positionally; the GDN layers see them as ordinary tokens,
    # and the label mask keeps them out of the loss.
    instance_source_config = LandmarkInstanceSourceConfig(
        source=build_longmino_512k_mix(
            tokenizer=tokenizer_config,
            sequence_length=CONTENT_SEQUENCE_LENGTH,
            tree="qwen35",
            root=DATA_ROOT,
            seed=1234,
        ),
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
    )

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
            load_path=CHECKPOINT_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=250,  # ~2384 steps total
                ephemeral_save_interval=None,
                max_checkpoints=3,
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
