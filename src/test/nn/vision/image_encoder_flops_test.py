"""FLOP accounting for the vision half, which feeds MFU.

Stage 1 freezes the ViT and Stage 2 trains it. ``image_encoder_flops`` used to assume
"frozen" unconditionally, charging the encoder forward-only and so under-reporting
Stage-2 MFU by roughly the backward pass. These tests pin that the charge now follows
``requires_grad``.
"""

import pytest

from olmo_core.nn.vision import MultimodalLM

from .multimodal_test import _tiny_multimodal_cfg

_CROPS = 8
_PATCHES = 4
_POOLED = 8


def _build() -> MultimodalLM:
    return _tiny_multimodal_cfg().build()


def _freeze_vision(model: MultimodalLM) -> None:
    for p in model.vision.parameters():
        p.requires_grad_(False)


def test_frozen_encoder_is_charged_forward_only():
    model = _build()
    _freeze_vision(model)

    assert not model.vision_is_trainable()
    frozen = model.image_encoder_flops(_CROPS, _PATCHES, _POOLED)

    for p in model.vision.parameters():
        p.requires_grad_(True)
    trainable = model.image_encoder_flops(_CROPS, _PATCHES, _POOLED)

    # Only the ViT term triples; the connector is trained in both cases, so the ratio
    # sits strictly between 1 and 3 rather than landing exactly on either.
    assert trainable > frozen
    assert 1.0 < trainable / frozen < 3.0


def test_vit_term_triples_when_trainable():
    """Isolate the ViT term by differencing two crop counts (connector held fixed)."""
    model = _build()

    _freeze_vision(model)
    fwd_only = model.image_encoder_flops(2 * _CROPS, _PATCHES, _POOLED) - model.image_encoder_flops(
        _CROPS, _PATCHES, _POOLED
    )

    for p in model.vision.parameters():
        p.requires_grad_(True)
    with_bwd = model.image_encoder_flops(2 * _CROPS, _PATCHES, _POOLED) - model.image_encoder_flops(
        _CROPS, _PATCHES, _POOLED
    )

    assert with_bwd == 3 * fwd_only


def test_vision_is_trainable_follows_requires_grad():
    model = _build()
    assert model.vision_is_trainable()

    _freeze_vision(model)
    assert not model.vision_is_trainable()

    # A single unfrozen parameter is enough -- the encoder then has a backward pass.
    next(iter(model.vision.parameters())).requires_grad_(True)
    assert model.vision_is_trainable()


@pytest.mark.parametrize("n_crops", [1, 8, 64])
def test_flops_scale_linearly_in_crops(n_crops):
    """Padded crops cost real FLOPs, so the charge must track the padded crop count."""
    model = _build()
    per_crop = model.image_encoder_flops(1, _PATCHES, 0)
    assert model.image_encoder_flops(n_crops, _PATCHES, 0) == n_crops * per_crop
