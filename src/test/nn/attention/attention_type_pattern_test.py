import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    AttentionTypePatternConfig,
    FastLandmarkAttention,
    SlidingWindowAttentionConfig,
    SparseLandmarkAttention,
)
from olmo_core.nn.transformer import TransformerConfig


@pytest.mark.parametrize(
    "force_first, force_last, layer_idx, expected_type",
    [
        # No forcing: pattern cycles directly over [fast, sparse].
        (False, False, 0, AttentionType.fast_landmark),
        (False, False, 1, AttentionType.sparse_landmark),
        (False, False, 2, AttentionType.fast_landmark),
        (False, False, 5, AttentionType.sparse_landmark),
        # Force full attention on the first layer (pattern shifts by one).
        (True, False, 0, AttentionType.default),
        (True, False, 1, AttentionType.fast_landmark),  # effective idx 0
        (True, False, 2, AttentionType.sparse_landmark),  # effective idx 1
        # Force full attention on the last layer.
        (False, True, 0, AttentionType.fast_landmark),
        (False, True, 5, AttentionType.default),
        # Force on both ends.
        (True, True, 0, AttentionType.default),
        (True, True, 1, AttentionType.fast_landmark),  # effective idx 0
        (True, True, 5, AttentionType.default),
    ],
)
def test_attention_type_pattern_get_type(
    force_first: bool,
    force_last: bool,
    layer_idx: int,
    expected_type: AttentionType,
):
    n_layers = 6
    config = AttentionTypePatternConfig(
        pattern=[AttentionType.fast_landmark, AttentionType.sparse_landmark],
        force_full_attention_on_first_layer=force_first,
        force_full_attention_on_last_layer=force_last,
    )
    assert config.get_type(layer_idx, n_layers) == expected_type


def test_attention_type_pattern_landmark_types():
    config = AttentionTypePatternConfig(
        pattern=[AttentionType.default, AttentionType.fast_landmark, AttentionType.sparse_landmark]
    )
    assert config.landmark_types() == {
        AttentionType.fast_landmark,
        AttentionType.sparse_landmark,
    }

    # A pattern of only full attention has no landmark types.
    assert AttentionTypePatternConfig(pattern=[AttentionType.default]).landmark_types() == set()


def test_attention_config_build_resolves_per_layer_type():
    n_layers = 6
    config = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        n_kv_heads=2,
        mem_freq=15,
        num_landmarks=2,
        layer_types=AttentionTypePatternConfig(
            pattern=[AttentionType.fast_landmark, AttentionType.sparse_landmark],
            force_full_attention_on_first_layer=True,
            force_full_attention_on_last_layer=True,
        ),
    )

    expected = [
        Attention,
        FastLandmarkAttention,
        SparseLandmarkAttention,
        FastLandmarkAttention,
        SparseLandmarkAttention,
        Attention,
    ]
    for layer_idx, expected_cls in enumerate(expected):
        module = config.build(128, layer_idx=layer_idx, n_layers=n_layers)
        assert isinstance(module, expected_cls), f"layer {layer_idx}: {type(module)}"


def test_attention_config_build_landmark_requires_mem_freq():
    config = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        layer_types=AttentionTypePatternConfig(pattern=[AttentionType.fast_landmark]),
    )
    with pytest.raises(OLMoConfigurationError):
        config.build(128, layer_idx=0, n_layers=4)


def test_attention_config_build_landmark_sliding_window_conflict():
    config = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        mem_freq=15,
        layer_types=AttentionTypePatternConfig(pattern=[AttentionType.fast_landmark]),
        sliding_window=SlidingWindowAttentionConfig(
            pattern=[64],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        ),
    )
    with pytest.raises(OLMoConfigurationError):
        config.build(128, layer_idx=0, n_layers=4)


def _mixed_pattern_config(**kwargs) -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=128,
        vocab_size=1024,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        mem_freq=15,
        num_landmarks=2,
        layer_types=AttentionTypePatternConfig(
            pattern=[AttentionType.fast_landmark, AttentionType.sparse_landmark],
        ),
        **kwargs,
    )


def test_llama_like_layer_types_builds_mixed_model():
    model = _mixed_pattern_config().build(init_device="cpu")
    classes = [type(block.attention).__name__ for block in model.blocks.values()]
    assert classes == [
        "FastLandmarkAttention",
        "SparseLandmarkAttention",
        "FastLandmarkAttention",
        "SparseLandmarkAttention",
        "FastLandmarkAttention",
        "SparseLandmarkAttention",
    ]


def test_llama_like_layer_types_round_trips():
    config = _mixed_pattern_config()
    rebuilt = TransformerConfig.from_dict(config.as_config_dict())
    layer_types = rebuilt.block.sequence_mixer.layer_types
    assert layer_types is not None
    assert [AttentionType(t) for t in layer_types.pattern] == [
        AttentionType.fast_landmark,
        AttentionType.sparse_landmark,
    ]
    # The deserialized config (pattern stored as strings) still builds.
    model = rebuilt.build(init_device="cpu")
    assert isinstance(next(iter(model.blocks.values())).attention, FastLandmarkAttention)


def test_llama_like_layer_types_rejects_uniform_flag():
    with pytest.raises(OLMoConfigurationError):
        _mixed_pattern_config(fast_landmark=True)


def test_llama_like_layer_types_rejects_num_landmarks_without_sparse():
    with pytest.raises(OLMoConfigurationError):
        TransformerConfig.llama_like(
            d_model=128,
            vocab_size=1024,
            n_layers=4,
            n_heads=4,
            mem_freq=15,
            num_landmarks=2,
            layer_types=AttentionTypePatternConfig(pattern=[AttentionType.fast_landmark]),
        )
