"""Tests that :class:`MultimodalLM` exposes one consistent parameter key space.

The vision encoder and connector are registered under ``vision_backbone.``. Back-compat
for checkpoints predating that rename used to be a ``state_dict()`` / ``load_state_dict()``
override pair on the model that rewrote those keys to the legacy ``vision.*`` /
``connector.*`` names.

That was unsound: optimizer FQNs come from ``named_parameters()``, which the override
never touched. A saved checkpoint therefore carried model key ``vision.class_embedding``
alongside optimizer key ``param_groups.vision_backbone.vision.class_embedding.lr`` — two
different key spaces in one checkpoint. Back-compat now lives at the checkpoint layer
(``load_key_mapping`` / :func:`swap_param_keys`) instead, and the converters normalize
their output with :func:`canonicalize_vision_keys`.
"""

from __future__ import annotations

from test.nn.vision.multimodal_test import _tiny_multimodal_cfg

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.distributed.checkpoint import swap_param_keys
from olmo_core.nn.vision.molmo2_loader import (
    VISION_BACKBONE_PREFIX,
    canonicalize_vision_keys,
    strip_vision_backbone_prefix,
)

LEGACY_PREFIXES = ("vision.", "connector.")


def _model_and_optim_state_dicts():
    model = _tiny_multimodal_cfg().build(init_device="cpu")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Take one step so the optimizer actually has state to key.
    model.lm.embeddings.weight.sum().backward()
    optim.step()

    opts = dist_cp_sd.StateDictOptions(flatten_optimizer_state_dict=True)
    return (
        model,
        dist_cp_sd.get_model_state_dict(model, options=opts),
        dist_cp_sd.get_optimizer_state_dict(model, optim, options=opts),
    )


def test_model_and_optimizer_state_dicts_share_one_key_space():
    _, model_sd, optim_sd = _model_and_optim_state_dicts()

    assert not [k for k in model_sd if k.startswith(LEGACY_PREFIXES)]
    vision_model_keys = [k for k in model_sd if k.startswith(VISION_BACKBONE_PREFIX)]
    assert vision_model_keys

    # The optimizer side is flattened (`param_groups.<fqn>.<hyperparam>`), so match on a
    # substring rather than a prefix.
    assert [k for k in optim_sd if VISION_BACKBONE_PREFIX in k]
    assert not [k for k in optim_sd if ".connector." in k and VISION_BACKBONE_PREFIX not in k]


def test_state_dict_keys_match_named_parameters():
    """The remap used to make these disagree, which is what broke optimizer keys."""
    model, model_sd, _ = _model_and_optim_state_dicts()
    for name, _ in model.named_parameters():
        assert name in model_sd, name


def test_state_dict_destination_arg_is_not_bypassed():
    """``nn.Module.state_dict`` fills ``destination`` in place; parents use that, not the
    return value. The override returned a *new* remapped dict, so ``destination`` kept the
    internal names while the return value had legacy ones — the remap was silently skipped
    for any nested use. With no override the two agree by construction."""
    model, _, _ = _model_and_optim_state_dicts()
    destination: dict = {}
    returned = model.state_dict(destination=destination)
    assert set(destination.keys()) == set(returned.keys())


def test_legacy_vision_key_mapping_covers_exactly_the_vision_subtree():
    model, model_sd, _ = _model_and_optim_state_dicts()
    mapping = model.legacy_vision_key_mapping()

    assert set(mapping) == {k for k in model_sd if k.startswith(VISION_BACKBONE_PREFIX)}
    for current, legacy in mapping.items():
        assert current.startswith(VISION_BACKBONE_PREFIX)
        assert current == VISION_BACKBONE_PREFIX + legacy
        assert legacy.startswith(LEGACY_PREFIXES)


def test_legacy_vision_key_mapping_round_trips_through_swap_param_keys():
    """A pre-rename checkpoint loads, and re-keying is lossless in both directions."""
    model, model_sd, _ = _model_and_optim_state_dicts()
    mapping = model.legacy_vision_key_mapping()

    state_dict: dict = {"model": dict(model_sd)}
    swap_param_keys(state_dict, mapping, quiet=True)
    # Now keyed the way a pre-rename checkpoint on disk is.
    assert len([k for k in state_dict["model"] if k.startswith(LEGACY_PREFIXES)]) == len(mapping)
    assert not [k for k in state_dict["model"] if k.startswith(VISION_BACKBONE_PREFIX)]

    swap_param_keys(state_dict, mapping, reverse=True, quiet=True)
    assert state_dict["model"].keys() == model_sd.keys()


def test_canonicalize_vision_keys_is_idempotent_and_invertible():
    _, model_sd, _ = _model_and_optim_state_dicts()

    legacy = strip_vision_backbone_prefix(model_sd)
    assert [k for k in legacy if k.startswith(LEGACY_PREFIXES)]

    assert canonicalize_vision_keys(legacy).keys() == model_sd.keys()
    # Already-current keys pass straight through.
    assert canonicalize_vision_keys(model_sd).keys() == model_sd.keys()
    assert strip_vision_backbone_prefix(canonicalize_vision_keys(legacy)).keys() == legacy.keys()


def test_converter_output_layout_actually_loads_the_vision_tower():
    """Guards the silent failure that removing the model-side override would have caused.

    The converters in ``molmo2_loader`` build ``vision.*`` / ``connector.*`` keys, and both
    stage scripts apply them with ``load_state_dict(..., strict=False)``. Without
    normalization those 48 keys become *unexpected*, ``strict=False`` swallows it, and
    training silently starts from a randomly initialized ViT and connector.
    """
    _, model_sd, _ = _model_and_optim_state_dicts()

    marker = 7.0
    legacy_converted = {
        k: (torch.full_like(v, marker) if v.is_floating_point() else v)
        for k, v in strip_vision_backbone_prefix(model_sd).items()
    }

    # Unnormalized: the vision tower is not covered.
    bad = _tiny_multimodal_cfg().build(init_device="cpu")
    missing, unexpected = bad.load_state_dict(legacy_converted, strict=False)
    assert [k for k in missing if k.startswith(VISION_BACKBONE_PREFIX)]
    assert unexpected
    assert not torch.allclose(bad.vision.class_embedding, torch.tensor(marker))

    # Normalized, as the converters now return: full coverage.
    good = _tiny_multimodal_cfg().build(init_device="cpu")
    missing, unexpected = good.load_state_dict(
        canonicalize_vision_keys(legacy_converted), strict=False
    )
    assert not missing
    assert not unexpected
    assert torch.allclose(good.vision.class_embedding, torch.tensor(marker))
    pooling = good.connector.pooling
    assert pooling is not None
    assert torch.allclose(pooling.wq.weight, torch.tensor(marker))
