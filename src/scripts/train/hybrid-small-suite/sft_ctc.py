"""
CTC-suite SFT for the hybrid-ratio ladder: 4:1 vs 7:1, at matched size.

Runs entirely on this repo's standard SFT machinery (``sft_common.run_sft``) -- same optimizer,
schedule, FSDP, activation checkpointing, packing and Beaker plumbing as ``sft_think.py``. The only
additions are (a) architectures for the 7:1 ratio and (b) Scalable-Softmax, which both released
arms of this comparison were trained with.

Usage::

    python src/scripts/train/hybrid-small-suite/sft_ctc.py dry_run  ctc-sft-1.4b-4to1 ai2/jupiter
    python src/scripts/train/hybrid-small-suite/sft_ctc.py launch   ctc-sft-1.4b-7to1 ai2/jupiter \\
        --launch.num_nodes=1 --launch.priority=urgent --launch.budget=ai2/oe-other

The run name selects the arm: it must contain a size (``275m``/``450m``/``810m``/``1.4b``) and a
ratio (``4to1`` or ``7to1``).

Data
----
``DATASET_PATH`` holds pre-tokenized olmo-core SFT shards, so nothing about CTC prompt construction
is needed here -- the training side only reads ``token_ids_part_*.npy`` + ``labels_mask_*.npy``.
The mix is 8 CTC tasks (absence, contradiction, nq, oolong, outlier, qdmatch_nq, strmatch,
xabsence) at rungs 2k-32k, dolma2 tokenizer, answer-only loss with the terminating EOS included.
See the README beside the shards for provenance and caveats.
"""

import math
import os
import sys
from functools import partial
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch import MODEL_CONFIGS  # noqa: E402
from sft_common import (  # noqa: E402
    SEQUENCE_LENGTH,
    build_data_components,
    build_trainer_config,
)

from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.float8 import Float8Config  # noqa: E402
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig  # noqa: E402
from olmo_core.train.train_module import (  # noqa: E402
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode  # noqa: E402
from olmo_core.internal.experiment import CommonComponents, build_config, main  # noqa: E402
from olmo_core.nn.attention import (  # noqa: E402
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig  # noqa: E402
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType  # noqa: E402
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation  # noqa: E402
from olmo_core.nn.transformer import (  # noqa: E402
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

DATASET_PATH = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft/shards_long32k"
)

MAINLINE = "/weka/oe-training-default/ai2-llm/scaling-ladders/mainline"

#: The two arms hold the number of FULL-ATTENTION layers fixed and vary how many gated-delta-net
#: layers sit between them: 4:1 is `LLLLF` repeated, 7:1 is `LLLLLLLF`. Both use Scalable-Softmax.
#: ``interval`` reproduces the checkpoints' own override indices exactly -- 5 -> {4,9,14,19},
#: 8 -> {7,15,23,31} -- because ``build_model_config`` overrides where
#: ``layer_idx % interval == interval - 1``.
#:
#: ⚠ The arms are NOT parameter-matched (1.4b: 1.42B vs 2.15B, +51%). That is inherent to the
#: design, not a flaw in this script, and must be quoted with any result.
ARMS: Dict[str, dict] = {
    "1.4b_4to1": dict(
        n_layers=20, interval=5, d_model=1280, n_heads=16,
        load_path=f"{MAINLINE}/yashasbls/v0.0.1-ssmax-a04f0e8e7236/1.4B-Cx8/long-context/step34156/",
    ),
    "1.4b_7to1": dict(
        n_layers=32, interval=8, d_model=1280, n_heads=16,
        load_path=f"{MAINLINE}/tanushy/v0.0.1-seven_to_one_hybrid_ratio-c6e480e336d5/1.4B-Cx8/long-context/step44124/",
    ),
    "275m_4to1": dict(
        n_layers=10, interval=5, d_model=640, n_heads=8,
        load_path=f"{MAINLINE}/yashasbls/v0.0.1-ssmax-a04f0e8e7236/275M-Cx8/long-context/step53839/",
    ),
    "275m_7to1": dict(
        n_layers=16, interval=8, d_model=640, n_heads=8,
        load_path=f"{MAINLINE}/tanushy/v0.0.1-seven_to_one_hybrid_ratio-c6e480e336d5/275M-Cx8/long-context/step72023/",
    ),
    "450m_4to1": dict(
        n_layers=15, interval=5, d_model=768, n_heads=8,
        load_path=f"{MAINLINE}/yashasbls/v0.0.1-ssmax-a04f0e8e7236/450M-Cx8/long-context/step42919/",
    ),
    "450m_7to1": dict(
        n_layers=24, interval=8, d_model=768, n_heads=8,
        load_path=f"{MAINLINE}/tanushy/v0.0.1-seven_to_one_hybrid_ratio-c6e480e336d5/450M-Cx8/long-context/step58015/",
    ),
}

# ── optimization: matched to amandab's validated SFT runs ────────────────────────────────────────
# Read off `amandab/q35-dense-contra-3ep-256k-min1h-20260914/step1005/config.json`. Those runs are
# known-good on this stack, so where they disagree with `sft_common`'s defaults we follow them:
# compile_model off (compile has been a recurring source of flakiness), activation checkpointing on
# `full` rather than `budget`, and a weight-decay override pinning the embeddings to 0.0.
#
# NOT copied: her context-parallel config (Ulysses degree 4). That exists because she trains at
# 262144; at 32768 CP only adds communication for no memory benefit.
LR = 4e-5
GLOBAL_BATCH_SIZE = 128 * SEQUENCE_LENGTH   # 4,194,304 tokens -- 2x sft_common, for utilisation
RANK_MICROBATCH = 2 * SEQUENCE_LENGTH       # 2 sequences per rank per microbatch


def build_ctc_train_module_config(common, **_) -> TransformerTrainModuleConfig:
    """Train-module config mirroring amandab's validated SFT setup."""
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            compile=False,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
    )


def parse_arm(run_name: str) -> str:
    """
    Resolve the arm key from a run name.

    :param run_name: e.g. ``ctc-sft-1.4b-7to1``.

    :returns: A key of :data:`ARMS`.

    :raises SystemExit: If the name does not name exactly one size and one ratio. Guessing here
        would silently train the wrong architecture against the right checkpoint.
    """
    name = run_name.lower().replace("-", "_")
    sizes = [s for s in ("1.4b", "810m", "450m", "275m") if s in name]
    ratios = [r for r in ("4to1", "7to1") if r in name]
    if len(sizes) != 1 or len(ratios) != 1:
        raise SystemExit(
            f"run name {run_name!r} must contain exactly one size and one ratio; "
            f"valid arms: {sorted(ARMS)}"
        )
    key = f"{sizes[0]}_{ratios[0]}"
    if key not in ARMS:
        raise SystemExit(f"no checkpoint registered for {key}; have {sorted(ARMS)}")
    return key


def build_ctc_model_config(
    common: CommonComponents,
    arm: str,
    attn_backend: AttentionBackendName = AttentionBackendName.flash_3,
) -> TransformerConfig:
    """
    Build the arm's architecture.

    Mirrors :func:`arch.build_model_config` but takes ``n_layers``/``interval`` from :data:`ARMS`
    rather than the per-size table (which encodes 4:1 only), and enables ``scalable_softmax`` on the
    attention layers. SSMax adds a learned per-head ``ssmax_scale`` parameter; these checkpoints
    carry it, so omitting it would leave those weights unloaded and run a different model.
    """
    cfg = ARMS[arm]
    d_model, n_heads = cfg["d_model"], cfg["n_heads"]
    n_layers, interval = cfg["n_layers"], cfg["interval"]
    n_kv_heads, head_dim, dtype, expand_v = 8, 128, DType.float32, 2.0

    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    feed_forward = FeedForwardConfig(
        hidden_size=d_model * 8, bias=False, dtype=dtype, activation=ActivationFunction.silu
    )

    block = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=n_heads, n_v_heads=n_heads, head_dim=head_dim, expand_v=expand_v, dtype=dtype
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    block_overrides: Dict[int, TransformerBlockConfig] = {}
    for layer_idx in range(n_layers):
        if layer_idx % interval == (interval - 1):
            block_overrides[layer_idx] = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=AttentionConfig(
                    name=AttentionType.default,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    bias=False,
                    rope=None,  # NoPE on the global layers
                    gate=GateConfig(
                        granularity=GateGranularity.elementwise, full_precision=True
                    ),
                    qk_norm=layer_norm,
                    use_head_qk_norm=True,
                    scalable_softmax=True,
                    backend=attn_backend,
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )

    return TransformerConfig(
        d_model=d_model,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=n_layers,
        block=block,
        lm_head=LMHeadConfig(
            loss_implementation=LMLossImplementation.default,
            layer_norm=layer_norm,
            bias=False,
            dtype=dtype,
        ),
        dtype=dtype,
        block_overrides=block_overrides or None,
        embed_scale=math.sqrt(d_model),
        embedding_norm=LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False),
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <dry_run|launch|train> <run_name> <cluster> [overrides...]\n"
            f"Run name must name a size and a ratio, e.g. ctc-sft-1.4b-7to1. Arms: {sorted(ARMS)}"
        )

    arm = parse_arm(sys.argv[2])
    size = arm.split("_")[0]
    sft_cfg = {size: dict(lr=LR, global_batch_size=GLOBAL_BATCH_SIZE, load_path=ARMS[arm]["load_path"])}

    CLUSTER_ATTN_BACKENDS = {
        "saturn": AttentionBackendName.flash_2,
        "jupiter": AttentionBackendName.flash_3,
        "titan": AttentionBackendName.flash_4,
    }
    cluster_arg = " ".join(sys.argv[2:4]).lower()
    attn_backend = AttentionBackendName.flash_3
    for cluster, backend in CLUSTER_ATTN_BACKENDS.items():
        if cluster in cluster_arg:
            attn_backend = backend
            break

    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=MODEL_CONFIGS[size]["num_nodes"] if size in MODEL_CONFIGS else 1,
        data_config_builder=partial(build_data_components, dataset_path=DATASET_PATH),
        model_config_builder=partial(build_ctc_model_config, arm=arm, attn_backend=attn_backend),
        train_module_config_builder=build_ctc_train_module_config,
        trainer_config_builder=partial(
            build_trainer_config, model_size=size, sft_configs=sft_cfg, tags=["ctc-sft", arm]
        ),
        include_default_evals=False,
        beaker_workspace="ai2/flex2",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
