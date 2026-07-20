"""Architecture for the 2.7B hybrid-small FLA reproduction run.

Adapted from ``YashasSamaga/OLMo-core`` at
``f9893fe62ab11aef89699df0307171a053616c30``.  The architecture is kept
identical to W&B run ``ai2-llm/hybrid-small-suite/gpii1bqu``.
"""

import math

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
MODEL_SIZE = "2.7b"

# Resolved compute settings from W&B run gpii1bqu. The source suite's defaults
# were overridden at launch to four nodes and a 4 Mi-token global batch.
MODEL_CONFIG = dict(
    d_model=1536,
    hidden_size=1536 * 8,
    n_layers=30,
    n_heads=16,
    head_dim=128,
    num_nodes=4,
    global_batch_size=4 * 1024 * 1024,
    rank_microbatch_size=4 * SEQUENCE_LENGTH,
)


def build_model_config(
    common: CommonComponents,
    attn_backend: AttentionBackendName = AttentionBackendName.flash_4,
) -> TransformerConfig:
    """Build the exact 4-GDN/1-global-attention repeating architecture."""
    d_model = MODEL_CONFIG["d_model"]
    hidden_size = MODEL_CONFIG["hidden_size"]
    n_layers = MODEL_CONFIG["n_layers"]
    n_heads = MODEL_CONFIG["n_heads"]
    head_dim = MODEL_CONFIG["head_dim"]
    dtype = DType.float32

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )
    feed_forward = FeedForwardConfig(
        hidden_size=hidden_size,
        bias=False,
        dtype=dtype,
        activation=ActivationFunction.silu,
    )

    block = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=n_heads,
            n_v_heads=n_heads,
            head_dim=head_dim,
            expand_v=2.0,
            allow_neg_eigval=True,
            conv_size=4,
            conv_bias=False,
            norm_eps=1e-5,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    block_overrides = {}
    for layer_idx in range(n_layers):
        if layer_idx % 5 == 4:
            block_overrides[layer_idx] = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=AttentionConfig(
                    name=AttentionType.default,
                    n_heads=n_heads,
                    n_kv_heads=8,
                    head_dim=head_dim,
                    bias=False,
                    rope=None,
                    gate=GateConfig(
                        granularity=GateGranularity.elementwise,
                        full_precision=True,
                    ),
                    qk_norm=layer_norm,
                    use_head_qk_norm=True,
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
        block_overrides=block_overrides,
        lm_head=LMHeadConfig(
            loss_implementation=LMLossImplementation.default,
            layer_norm=layer_norm,
            bias=False,
            dtype=dtype,
        ),
        dtype=dtype,
        embed_scale=math.sqrt(d_model),
        embedding_norm=LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
        ),
    )
