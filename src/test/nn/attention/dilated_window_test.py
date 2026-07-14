"""Tests for the (DEPRECATED) positional ``DilatedSlidingWindowAttention`` -- a dilated causal
sliding-window mask whose dilation stride rotates with transformer depth ("Hierarchical K").

The attention *module* and its config selection are deprecated and now raise
:class:`NotImplementedError`; use the document-chunked ``cross_doc_mode="hierarchical_dilated"``
pattern instead (see ``hierarchical_dilated_test.py``). The pure positional mask helpers
(:func:`build_dilated_window_allowed_mask`, :func:`layer_dilation`) are retained for reference and
still tested below."""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    DilatedSlidingWindowAttention,
)
from olmo_core.nn.attention.dilated_window import (
    build_dilated_window_allowed_mask,
    layer_dilation,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerConfig

# ---------------------------------------------------------------------------
# Deprecation: constructing the module / selecting it via config now raises.
# ---------------------------------------------------------------------------


def test_direct_construction_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DEPRECATED"):
        DilatedSlidingWindowAttention(n_heads=8, head_dim=8)


def test_attention_config_build_raises_not_implemented():
    config = AttentionConfig(
        name=AttentionType.dilated_sliding_window,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
    )
    with pytest.raises(NotImplementedError, match="DEPRECATED"):
        config.build(64, layer_idx=0, n_layers=12)


def test_llama_like_dilated_sliding_window_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DEPRECATED"):
        TransformerConfig.llama_like(
            d_model=64,
            vocab_size=1024,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            rope_type=RoPEType.default,
            dilated_sliding_window=True,
            dilated_window_k=3,
            dilated_window_num_configs=3,
            dilated_window_base=2,
            attn_backend=AttentionBackendName.torch,
            dtype=DType.float32,
        )


# ---------------------------------------------------------------------------
# Pure positional mask helpers (retained for reference; not deprecated).
# ---------------------------------------------------------------------------


def _allowed_offsets(dilation: int, window: int, *, t: int = 20) -> set:
    """The set of key *offsets* (t - k) a query at position ``t`` may attend, from the mask."""
    positions = torch.arange(t + 1)
    allowed = build_dilated_window_allowed_mask(
        positions, positions, dilation=dilation, window=window
    )
    row = allowed[t]  # (t+1,) over keys 0..t
    keys = row.nonzero(as_tuple=True)[0].tolist()
    return {t - k for k in keys}


def test_layer_dilation_rotation():
    # base=2, num_configs=3 -> strides rotate 1, 2, 4, 1, 2, 4, 1 across layers 0..6.
    strides = [layer_dilation(ell, num_configs=3, base=2) for ell in range(7)]
    assert strides == [1, 2, 4, 1, 2, 4, 1]


@pytest.mark.parametrize(
    "cycle_pos, dilation, expected_offsets",
    [
        (0, 1, {0, 1, 2}),
        (1, 2, {0, 2, 4}),
        (2, 4, {0, 4, 8}),
    ],
)
def test_dilated_window_offsets_per_cycle_position(cycle_pos, dilation, expected_offsets):
    assert layer_dilation(cycle_pos, num_configs=3, base=2) == dilation
    assert _allowed_offsets(dilation, window=3) == expected_offsets


def test_dilated_window_is_causal():
    positions = torch.arange(6)
    allowed = build_dilated_window_allowed_mask(positions, positions, dilation=2, window=3)
    for q in range(6):
        for k in range(6):
            if k > q:
                assert not bool(allowed[q, k]), (q, k)


def test_dilated_window_truncates_at_sequence_start():
    # A query near the start has fewer than ``window`` valid strided predecessors.
    assert _allowed_offsets(2, window=3, t=1) == {0}  # only offsets 0 (and 2 would be position -1)


def test_dilated_window_reset_at_cycle_length():
    # Layer 0 and layer 3 (num_configs=3) share the same stride -> identical masks.
    positions = torch.arange(12)
    d0 = layer_dilation(0, num_configs=3, base=2)
    d3 = layer_dilation(3, num_configs=3, base=2)
    assert d0 == d3
    m0 = build_dilated_window_allowed_mask(positions, positions, dilation=d0, window=3)
    m3 = build_dilated_window_allowed_mask(positions, positions, dilation=d3, window=3)
    assert torch.equal(m0, m3)
