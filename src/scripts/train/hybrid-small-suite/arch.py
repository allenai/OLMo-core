"""
Shared architecture and per-model-size compute settings for the OLMo hybrid small suite
(275M, 810M, 1.4B).

Imported by all per-stage training scripts in this folder (pretraining.py, midtraining.py, …).
Training-specific settings (lr, load_path, scheduler) are kept in each stage script.
"""

import math
from typing import Dict

from olmo_core.config import DType
from olmo_core.internal.experiment import CommonComponents
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

SEQUENCE_LENGTH = 8192

# Per-size architecture and compute settings shared across all training stages.
# Keys: d_model, hidden_size, n_layers, n_heads, num_nodes,
#       global_batch_size, rank_microbatch_size.
MODEL_CONFIGS: Dict[str, dict] = {
    "275m": dict(
        # 275,493,760 total / 211,268,480 non-embedding params
        d_model=640,
        hidden_size=640 * 8,
        n_layers=10,
        n_heads=8,
        num_nodes=4,
        global_batch_size=2_621_440,
        rank_microbatch_size=5 * SEQUENCE_LENGTH,
    ),
    "810m": dict(
        # 810,354,816 total / 707,594,368 non-embedding params
        d_model=1024,
        hidden_size=1024 * 8,
        n_layers=15,
        n_heads=16,
        num_nodes=8,
        global_batch_size=5_242_880,
        rank_microbatch_size=2 * SEQUENCE_LENGTH,
    ),
    "1.4b": dict(
        # 1,422,110,720 total / 1,293,660,160 non-embedding params
        d_model=1280,
        hidden_size=1280 * 8,
        n_layers=20,
        n_heads=16,
        num_nodes=32,
        global_batch_size=8_388_608,
        rank_microbatch_size=2 * SEQUENCE_LENGTH,
    ),
}


def build_model_config(common: CommonComponents, model_size: str) -> TransformerConfig:
    """
    Build the TransformerConfig for the given model size.

    Architecture: Gemma3-like hybrid — 4 GDN layers + 1 global attention layer (repeating).
    GDN uses expand_v=2 and no positional embeddings on the global attention layers (NoPE).
    """
    cfg = MODEL_CONFIGS[model_size]

    d_model = cfg["d_model"]
    hidden_size = cfg["hidden_size"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]

    n_kv_heads = 8
    head_dim = 128
    global_layer_interval = 5
    layer_norm_eps = 1e-6
    dtype = DType.float32
    expand_v = 2.0

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=layer_norm_eps,
        bias=False,
        dtype=dtype,
    )

    feed_forward = FeedForwardConfig(
        hidden_size=hidden_size,
        bias=False,
        dtype=dtype,
        activation=ActivationFunction.silu,
    )

    # Default block: GDN.
    block = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=n_heads,
            n_v_heads=n_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    # Override every global_layer_interval-th layer with full global attention (NoPE).
    block_overrides: Dict[int, TransformerBlockConfig] = {}
    for layer_idx in range(n_layers):
        if layer_idx % global_layer_interval == (global_layer_interval - 1):
            global_block = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=AttentionConfig(
                    name=AttentionType.default,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    bias=False,
                    rope=None,
                    gate=GateConfig(
                        granularity=GateGranularity.elementwise,
                        full_precision=True,
                    ),
                    qk_norm=layer_norm,
                    use_head_qk_norm=True,
                    backend=AttentionBackendName.flash_3,
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )
            block_overrides[layer_idx] = global_block

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
        block_overrides=block_overrides if block_overrides else None,
        embed_scale=math.sqrt(d_model),
        embedding_norm=LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
        ),
    )


def parse_model_size(run_name: str) -> str:
    """
    Extract model size key from a run name string.

    Examples::

        "hybrid-small-275M"            -> "275m"
        "hybrid-small-midtraining-1.4B" -> "1.4b"

    :raises SystemExit: If no recognized size is found in ``run_name``.
    """
    run_name_lower = run_name.lower()
    # Try longest keys first so "1.4b" matches before any partial overlap.
    for key in sorted(MODEL_CONFIGS.keys(), key=len, reverse=True):
        if key in run_name_lower:
            return key
    raise SystemExit(
        f"Error: could not parse model size from run name '{run_name}'. "
        f"Run name must contain one of: {list(MODEL_CONFIGS.keys())}"
    )
