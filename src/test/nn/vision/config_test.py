"""Tests for VisionBackboneConfig factory methods and build()."""

import pytest

from olmo_core.nn.vision.config import VisionBackboneConfig, VisionBackboneType
from olmo_core.nn.vision.image_vit import SiglipVisionTransformer, VisionTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIGLIP_FACTORIES = [
    ("siglip_b16_224", VisionBackboneConfig.siglip_b16_224, 768, 12, 196),
    ("siglip_l16_384", VisionBackboneConfig.siglip_l16_384, 1024, 24, 576),
    (
        "siglip_so400m_patch14_224",
        VisionBackboneConfig.siglip_so400m_patch14_224,
        1152,
        27,
        256,
    ),
    ("siglip_so400m", VisionBackboneConfig.siglip_so400m, 1152, 27, 729),
]

_SIGLIP2_FACTORIES = [
    ("siglip2_b16_256", VisionBackboneConfig.siglip2_b16_256, 768, 12, 256),
    ("siglip2_l16_256", VisionBackboneConfig.siglip2_l16_256, 1024, 24, 256),
    (
        "siglip2_so400m_patch14_378",
        VisionBackboneConfig.siglip2_so400m_patch14_378,
        1152,
        27,
        729,
    ),
    (
        "siglip2_so400m_patch16_256",
        VisionBackboneConfig.siglip2_so400m_patch16_256,
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
    cfg = VisionBackboneConfig()
    assert cfg.name == VisionBackboneType.openai
    assert cfg.image_emb_dim == 1024
    assert cfg.image_num_layers == 23
    assert cfg.image_num_pos == 577
    assert cfg.image_emb_dim == cfg.image_num_heads * cfg.image_head_dim


# ---------------------------------------------------------------------------
# build() returns the correct class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_cls",
    [
        (VisionBackboneConfig, VisionTransformer),
        (VisionBackboneConfig.siglip_so400m, SiglipVisionTransformer),
        (VisionBackboneConfig.siglip_b16_224, SiglipVisionTransformer),
        (VisionBackboneConfig.siglip2_b16_256, SiglipVisionTransformer),
        (VisionBackboneConfig.siglip2_so400m_patch14_378, SiglipVisionTransformer),
    ],
    ids=["clip", "siglip_so400m", "siglip_b16", "siglip2_b16", "siglip2_so400m"],
)
def test_build_returns_correct_class(factory, expected_cls):
    cfg = factory()
    model = cfg.build(init_device="meta")
    assert isinstance(model, expected_cls)


# ---------------------------------------------------------------------------
# CLIP has CLS token, SigLIP / SigLIP2 do not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expect_cls_token",
    [
        (VisionBackboneConfig, True),
        (VisionBackboneConfig.siglip_so400m, False),
        (VisionBackboneConfig.siglip2_b16_256, False),
    ],
    ids=["clip", "siglip", "siglip2"],
)
def test_cls_token_presence(factory, expect_cls_token):
    model = factory().build(init_device="meta")
    if expect_cls_token:
        assert hasattr(model, "class_embedding")
        assert model.num_prefix_tokens == 1
    else:
        assert not hasattr(model, "class_embedding")
        assert model.num_prefix_tokens == 0


# ---------------------------------------------------------------------------
# image_num_patch property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_patch_num",
    [
        (VisionBackboneConfig, (24, 24)),  # 336/14
        (VisionBackboneConfig.siglip_b16_224, (14, 14)),  # 224/16
        (VisionBackboneConfig.siglip_l16_384, (24, 24)),  # 384/16
        (VisionBackboneConfig.siglip2_b16_256, (16, 16)),  # 256/16
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
    s1 = VisionBackboneConfig.siglip_so400m()
    s2 = VisionBackboneConfig.siglip2_so400m_patch14_378()
    assert s1.image_emb_dim == s2.image_emb_dim
    assert s1.image_num_heads == s2.image_num_heads
    assert s1.image_num_layers == s2.image_num_layers
    assert s1.name != s2.name
    assert s1.name == VisionBackboneType.siglip
    assert s2.name == VisionBackboneType.siglip2
