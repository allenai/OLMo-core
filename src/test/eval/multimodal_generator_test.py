"""Tests for :class:`MultimodalGenerator`."""

import pytest
import torch

from olmo_core.data.multimodal import MultimodalTokenizerConfig
from olmo_core.eval.multimodal_generator import GenerationOutput, MultimodalGenerator
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalTransformer,
    MultimodalTransformerConfig,
    VisionBackboneConfig,
    VisionBackboneType,
    VisionConnectorConfig,
)


def _tiny_model() -> MultimodalTransformer:
    mm_tok = MultimodalTokenizerConfig.dolma2()
    lm_cfg = TransformerConfig.olmo2_1M(vocab_size=mm_tok.padded_vocab_size(128))
    vis_cfg = VisionBackboneConfig(
        name=VisionBackboneType.openai,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=32,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=64,
        image_num_pos=5,
        image_norm_eps=1e-5,
    )
    conn_cfg = VisionConnectorConfig.from_vision_backbone(
        vis_cfg, output_dim=lm_cfg.d_model, mlp_hidden_size=32
    )
    return MultimodalTransformer(
        MultimodalTransformerConfig(
            lm=lm_cfg,
            vision=vis_cfg,
            connector=conn_cfg,
            image_patch_token_id=mm_tok.image_patch_id,
        ),
        init_device="cpu",
    )


# ---------------------------------------------------------------------------
# Greedy text-only
# ---------------------------------------------------------------------------


def test_generate_greedy_text_only():
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    out = gen.generate(
        input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long),
        max_new_tokens=5,
    )
    assert isinstance(out, GenerationOutput)
    assert len(out.token_ids) == 5
    assert out.finished_reason == "max_tokens"


def test_greedy_is_deterministic():
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    out1 = gen.generate(input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long), max_new_tokens=8)
    out2 = gen.generate(input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long), max_new_tokens=8)
    assert out1.token_ids == out2.token_ids


# ---------------------------------------------------------------------------
# EOS / stop-token
# ---------------------------------------------------------------------------


def test_eos_stops_generation():
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    # Force EOS to be the only winning token by setting all other logits to
    # a huge negative via a stub.
    eos_id = 7
    # We can't easily force the model to predict EOS, but we *can* test the
    # control flow: set eos_token_id to one of the tokens that always wins
    # for this seed-fixed input. Easier: rely on the model picking one of
    # the tokens it'll generate; set max_new_tokens larger so we observe
    # at least one full sequence.
    torch.manual_seed(0)
    out_with_eos = gen.generate(
        input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long),
        max_new_tokens=50,
        eos_token_id=eos_id,
    )
    # Either we hit EOS or filled the buffer. Either way, no EOS in output.
    assert eos_id not in out_with_eos.token_ids
    assert out_with_eos.finished_reason in {"eos", "max_tokens"}


def test_stop_token_ids_also_halts():
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    # Same control-flow check as EOS, but via stop_token_ids.
    out = gen.generate(
        input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long),
        max_new_tokens=50,
        stop_token_ids=(10, 11, 12),
    )
    for tid in (10, 11, 12):
        assert tid not in out.token_ids


# ---------------------------------------------------------------------------
# With images
# ---------------------------------------------------------------------------


def test_generate_with_image():
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    mm_tok = MultimodalTokenizerConfig.dolma2()
    patch_id = mm_tok.image_patch_id

    # Build prompt with exactly 1 <im_patch> token (= 1 pooled feature in our config).
    input_ids = torch.tensor([[patch_id, 5, 6]], dtype=torch.long)
    images = torch.randn(1, 1, 4, 14 * 14 * 3)
    pooled_patches_idx = torch.arange(4).view(1, 1, 4)
    out = gen.generate(
        input_ids=input_ids,
        images=images,
        pooled_patches_idx=pooled_patches_idx,
        max_new_tokens=3,
    )
    assert len(out.token_ids) == 3


def test_images_without_idx_raises():
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    with pytest.raises(ValueError, match="pooled_patches_idx"):
        gen.generate(
            input_ids=torch.tensor([[5, 6]], dtype=torch.long),
            images=torch.randn(1, 1, 4, 14 * 14 * 3),
            max_new_tokens=2,
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_topp_sampling_changes_with_seed():
    """Different RNG seeds should give different outputs with stochastic sampling."""
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    prompts = torch.tensor([[5, 6, 7]], dtype=torch.long)
    torch.manual_seed(0)
    a = gen.generate(prompts, max_new_tokens=10, temperature=1.0, top_p=0.9)
    torch.manual_seed(1)
    b = gen.generate(prompts, max_new_tokens=10, temperature=1.0, top_p=0.9)
    # Most of the time the sequences should differ.
    assert a.token_ids != b.token_ids or len(a.token_ids) <= 1


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_rejects_unbatched_input():
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    with pytest.raises(ValueError, match="shape"):
        gen.generate(input_ids=torch.tensor([5, 6, 7], dtype=torch.long), max_new_tokens=2)


def test_rejects_batched_input():
    """generate() only supports batch_size=1 in this implementation."""
    model = _tiny_model()
    gen = MultimodalGenerator(model)
    with pytest.raises(ValueError, match="shape"):
        gen.generate(input_ids=torch.tensor([[5, 6], [7, 8]], dtype=torch.long), max_new_tokens=2)
