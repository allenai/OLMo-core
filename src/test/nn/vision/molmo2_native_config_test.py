"""``MultimodalLMConfig.molmo2_4B`` must stay identical to the HF-derived config.

The native factory exists so training from base checkpoints needs nothing from the
released ``allenai/Molmo2-4B`` repo. That is only safe while it produces exactly what
:func:`molmo2_config_from_hf_config` produces, so this test pins the two together.
"""

import pytest

from olmo_core.nn.vision import MultimodalLMConfig

from ._molmo2_common import _hf_cache_has

transformers = pytest.importorskip("transformers")

MODEL_ID = "allenai/Molmo2-4B"


def _flatten(d, prefix=""):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


@pytest.mark.skipif(not _hf_cache_has(MODEL_ID), reason=f"{MODEL_ID} not in local HF cache")
def test_molmo2_4b_native_config_matches_hf_derived():
    from transformers import AutoConfig

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_config_from_hf_config,
    )

    ensure_default_rope_registered()
    hf_derived = molmo2_config_from_hf_config(
        AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    )
    native = MultimodalLMConfig.molmo2_4B()

    expected, actual = _flatten(hf_derived.as_config_dict()), _flatten(native.as_config_dict())
    differences = {
        key: (expected.get(key), actual.get(key))
        for key in expected.keys() | actual.keys()
        if expected.get(key) != actual.get(key)
    }
    assert not differences, f"native config diverged from the HF-derived one: {differences}"


def test_molmo2_4b_rope_theta_is_selectable():
    """Base Qwen3-4B was trained at 1e6; the released Molmo2-4B checkpoint uses 5e6."""
    assert MultimodalLMConfig.molmo2_4B().lm.block.sequence_mixer.rope.theta == 5_000_000
    scratch = MultimodalLMConfig.molmo2_4B(rope_theta=1_000_000)
    assert scratch.lm.block.sequence_mixer.rope.theta == 1_000_000


def test_molmo2_4b_native_config_shape():
    """Layout invariants that do not need the HF checkpoint cached."""
    cfg = MultimodalLMConfig.molmo2_4B()
    assert cfg.lm.d_model == 2560
    assert cfg.lm.n_layers == 36
    assert cfg.lm.vocab_size == 152_064  # 151,936 base + 128 image-special tokens
    assert cfg.lm.tie_word_embeddings is True
    assert cfg.output_vocab_size == 151_936
    assert cfg.vision.image_num_layers == 25  # SigLIP2 blocks 0..24 of 27
    assert cfg.vit_layers == (24, 18)
    assert cfg.connector.num_input_layers == len(cfg.vit_layers)
    assert cfg.connector.output_dim == cfg.lm.d_model
    assert max(cfg.vit_layers) == cfg.vision.image_num_layers - 1
