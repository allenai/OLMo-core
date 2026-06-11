"""Tests for VisionEncoderConfig factory methods and build()."""

import pytest

from olmo_core.nn.vision.config import VisionEncoderConfig, VisionEncoderType
from olmo_core.nn.vision.image_vit import VisionTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIGLIP_FACTORIES = [
    ("siglip_b16_224", VisionEncoderConfig.siglip_b16_224, 768, 12, 196),
    ("siglip_l16_384", VisionEncoderConfig.siglip_l16_384, 1024, 24, 576),
    (
        "siglip_so400m_patch14_224",
        VisionEncoderConfig.siglip_so400m_patch14_224,
        1152,
        27,
        256,
    ),
    ("siglip_so400m", VisionEncoderConfig.siglip_so400m, 1152, 27, 729),
]

_SIGLIP2_FACTORIES = [
    ("siglip2_b16_256", VisionEncoderConfig.siglip2_b16_256, 768, 12, 256),
    ("siglip2_l16_256", VisionEncoderConfig.siglip2_l16_256, 1024, 24, 256),
    (
        "siglip2_so400m_patch14_378",
        VisionEncoderConfig.siglip2_so400m_patch14_378,
        1152,
        27,
        729,
    ),
    (
        "siglip2_so400m_patch16_256",
        VisionEncoderConfig.siglip2_so400m_patch16_256,
        1152,
        27,
        256,
    ),
]


# ---------------------------------------------------------------------------
# Field-value tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,factory,emb_dim,n_layers,num_pos",
    _SIGLIP_FACTORIES + _SIGLIP2_FACTORIES,
    ids=[row[0] for row in _SIGLIP_FACTORIES + _SIGLIP2_FACTORIES],
)
def test_factory_fields(name, factory, emb_dim, n_layers, num_pos):
    cfg = factory()
    assert cfg.image_emb_dim == emb_dim, f"{name}: emb_dim"
    assert cfg.image_num_layers == n_layers, f"{name}: n_layers"
    assert cfg.image_num_pos == num_pos, f"{name}: num_pos"
    # head_dim must satisfy emb_dim == heads * head_dim
    assert (
        cfg.image_emb_dim == cfg.image_num_heads * cfg.image_head_dim
    ), f"{name}: head_dim consistency"


def test_clip_default_fields():
    cfg = VisionEncoderConfig()
    assert cfg.name == VisionEncoderType.openai
    assert cfg.image_emb_dim == 1024
    assert cfg.image_num_layers == 23
    assert cfg.image_num_pos == 577
    assert cfg.image_emb_dim == cfg.image_num_heads * cfg.image_head_dim


# ---------------------------------------------------------------------------
# build() returns the correct class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expect_cls_token",
    [
        (VisionEncoderConfig, True),
        (VisionEncoderConfig.siglip_so400m, False),
        (VisionEncoderConfig.siglip_b16_224, False),
        (VisionEncoderConfig.siglip2_b16_256, False),
        (VisionEncoderConfig.siglip2_so400m_patch14_378, False),
    ],
    ids=["clip", "siglip_so400m", "siglip_b16", "siglip2_b16", "siglip2_so400m"],
)
def test_build_returns_configured_variant(factory, expect_cls_token):
    model = factory().build(init_device="meta")
    assert isinstance(model, VisionTransformer)
    # CLIP prepends a CLS token, applies a pre-LN, and uses a bias-free patch
    # projection; the SigLIP families are the inverse on all three switches.
    assert (model.class_embedding is not None) == expect_cls_token
    assert (model.pre_ln is not None) == expect_cls_token
    assert (model.patch_embedding.bias is not None) == (not expect_cls_token)


# ---------------------------------------------------------------------------
# CLIP has CLS token, SigLIP / SigLIP2 do not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expect_cls_token",
    [
        (VisionEncoderConfig, True),
        (VisionEncoderConfig.siglip_so400m, False),
        (VisionEncoderConfig.siglip2_b16_256, False),
    ],
    ids=["clip", "siglip", "siglip2"],
)
def test_cls_token_presence(factory, expect_cls_token):
    model = factory().build(init_device="meta")
    if expect_cls_token:
        assert model.class_embedding is not None
        assert model.num_prefix_tokens == 1
    else:
        assert model.class_embedding is None
        assert model.num_prefix_tokens == 0


# ---------------------------------------------------------------------------
# image_num_patch property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_patch_num",
    [
        (VisionEncoderConfig, (24, 24)),  # 336/14
        (VisionEncoderConfig.siglip_b16_224, (14, 14)),  # 224/16
        (VisionEncoderConfig.siglip_l16_384, (24, 24)),  # 384/16
        (VisionEncoderConfig.siglip2_b16_256, (16, 16)),  # 256/16
    ],
    ids=["clip", "siglip_b16", "siglip_l16", "siglip2_b16"],
)
def test_image_num_patch(factory, expected_patch_num):
    cfg = factory()
    assert cfg.image_num_patch == expected_patch_num


# ---------------------------------------------------------------------------
# SigLIP2 and SigLIP SO400M share architecture; differ only by type
# ---------------------------------------------------------------------------


def test_siglip2_so400m_shares_arch_with_siglip_so400m():
    s1 = VisionEncoderConfig.siglip_so400m()
    s2 = VisionEncoderConfig.siglip2_so400m_patch14_378()
    assert s1.image_emb_dim == s2.image_emb_dim
    assert s1.image_num_heads == s2.image_num_heads
    assert s1.image_num_layers == s2.image_num_layers
    assert s1.name != s2.name
    assert s1.name == VisionEncoderType.siglip
    assert s2.name == VisionEncoderType.siglip2


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_config_round_trip():
    # A SigLIP config exercises the non-default variant switches.
    cfg = VisionEncoderConfig.siglip_b16_224()
    dumped = cfg.as_config_dict()
    assert dumped["name"] == "siglip"

    restored = VisionEncoderConfig.from_dict(dumped)
    assert restored.name == VisionEncoderType.siglip
    # The variant switches must survive the round-trip.
    assert restored.use_cls_token is False
    assert restored.patch_embedding_bias is True
    assert restored.use_pre_ln is False
    assert restored.image_emb_dim == cfg.image_emb_dim
