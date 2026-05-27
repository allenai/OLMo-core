import pytest
import torch

from olmo_core.nn.vision import (
    SiglipVisionTransformer,
    VisionBackboneConfig,
    VisionBackboneType,
    VisionTransformer,
)

# ---------------------------------------------------------------------------
# Tiny configs for fast CPU tests
# ---------------------------------------------------------------------------


def _tiny_clip_cfg(**kwargs) -> VisionBackboneConfig:
    """Minimal CLIP-style config: 2 layers, small dims."""
    return VisionBackboneConfig(
        name=VisionBackboneType.openai,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=128,
        image_mlp_activations="quick_gelu",
        image_num_pos=5,  # 2*2 patches + 1 CLS
        **kwargs,
    )


def _tiny_siglip_cfg(**kwargs) -> VisionBackboneConfig:
    """Minimal SigLIP-style config: 2 layers, small dims."""
    return VisionBackboneConfig(
        name=VisionBackboneType.siglip,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=128,
        image_mlp_activations="gelu_pytorch_tanh",
        image_num_pos=4,  # 2*2 patches, no CLS
        **kwargs,
    )


# ---------------------------------------------------------------------------
# VisionBackboneConfig.build()
# ---------------------------------------------------------------------------


def test_clip_build_returns_correct_type():
    cfg = _tiny_clip_cfg()
    vit = cfg.build(init_device="cpu")
    assert isinstance(vit, VisionTransformer)


def test_siglip_build_returns_correct_type():
    cfg = _tiny_siglip_cfg()
    vit = cfg.build(init_device="cpu")
    assert isinstance(vit, SiglipVisionTransformer)


# ---------------------------------------------------------------------------
# VisionTransformer (CLIP-style)
# ---------------------------------------------------------------------------


class TestVisionTransformer:
    def setup_method(self):
        self.cfg = _tiny_clip_cfg()
        self.vit = VisionTransformer(self.cfg, init_device="cpu")

    def _make_input(self, batch: int = 2) -> torch.Tensor:
        n_patches = 2 * 2  # 28px / 14px = 2 per side
        patch_pixels = 14 * 14 * 3
        return torch.randn(batch, n_patches, patch_pixels)

    def test_forward_returns_list_of_length_num_layers(self):
        x = self._make_input()
        out = self.vit(x)
        assert len(out) == self.cfg.image_num_layers

    def test_output_shape_includes_cls_token(self):
        batch = 3
        x = self._make_input(batch)
        out = self.vit(x)
        n_patches = 2 * 2
        expected = (batch, 1 + n_patches, self.cfg.image_emb_dim)
        for layer_out in out:
            assert layer_out.shape == expected, f"Got {layer_out.shape}, expected {expected}"

    def test_forward_deterministic(self):
        x = self._make_input()
        self.vit.eval()
        out1 = self.vit(x)
        out2 = self.vit(x)
        assert torch.allclose(out1[-1], out2[-1])

    def test_patch_num_override(self):
        """Positional embeddings should interpolate to a different resolution."""
        cfg = _tiny_clip_cfg()
        cfg.image_default_input_size = (42, 42)  # 3x3 patches
        cfg.image_num_pos = 5  # still sized for 2x2 — must interpolate
        vit = VisionTransformer(cfg, init_device="cpu")

        n_patches = 3 * 3
        patch_pixels = 14 * 14 * 3
        x = torch.randn(1, n_patches, patch_pixels)
        out = vit(x, patch_num=(3, 3))
        assert out[-1].shape == (1, 1 + n_patches, cfg.image_emb_dim)

    def test_num_prefix_tokens(self):
        assert self.vit.num_prefix_tokens == 1

    def test_meta_device_build(self):
        vit = VisionTransformer(_tiny_clip_cfg(), init_device="meta")
        assert next(iter(vit.parameters())).device.type == "meta"

    @pytest.mark.parametrize("batch", [1, 4])
    def test_various_batch_sizes(self, batch: int):
        x = self._make_input(batch)
        out = self.vit(x)
        assert out[-1].shape[0] == batch


# ---------------------------------------------------------------------------
# SiglipVisionTransformer
# ---------------------------------------------------------------------------


class TestSiglipVisionTransformer:
    def setup_method(self):
        self.cfg = _tiny_siglip_cfg()
        self.vit = SiglipVisionTransformer(self.cfg, init_device="cpu")

    def _make_input(self, batch: int = 2) -> torch.Tensor:
        n_patches = 2 * 2
        patch_pixels = 14 * 14 * 3
        return torch.randn(batch, n_patches, patch_pixels)

    def test_forward_returns_list_of_length_num_layers(self):
        x = self._make_input()
        out = self.vit(x)
        assert len(out) == self.cfg.image_num_layers

    def test_output_shape_no_cls_token(self):
        batch = 2
        x = self._make_input(batch)
        out = self.vit(x)
        n_patches = 2 * 2
        expected = (batch, n_patches, self.cfg.image_emb_dim)
        for layer_out in out:
            assert layer_out.shape == expected

    def test_num_prefix_tokens(self):
        assert self.vit.num_prefix_tokens == 0

    def test_meta_device_build(self):
        vit = SiglipVisionTransformer(_tiny_siglip_cfg(), init_device="meta")
        assert next(iter(vit.parameters())).device.type == "meta"


# ---------------------------------------------------------------------------
# VisionBackboneConfig helpers
# ---------------------------------------------------------------------------


def test_siglip_so400m_factory():
    cfg = VisionBackboneConfig.siglip_so400m()
    assert cfg.name == VisionBackboneType.siglip
    assert cfg.image_default_input_size == (378, 378)
    assert cfg.image_emb_dim == 1152
    assert cfg.image_num_layers == 27
    assert cfg.image_num_pos == 729


def test_image_num_patch_property():
    cfg = _tiny_clip_cfg()
    assert cfg.image_num_patch == (2, 2)


def test_config_round_trip_yaml():
    from olmo_core.config import Config

    cfg = _tiny_clip_cfg()
    assert isinstance(cfg, Config)
    dumped = cfg.as_config_dict()
    assert dumped["name"] == "openai"
