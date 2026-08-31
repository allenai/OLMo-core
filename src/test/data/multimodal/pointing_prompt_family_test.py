"""The pointing prompt form must follow the checkpoint's family, not the dataset.

Stage 1 trains with the released ``Molmo2-4B-Pretrain`` family (``prompt_templates="none"`` +
``system_prompt="style_and_length_v2"``): the question is the bare lowercased label behind a
``"<style>:"`` prefix. Stage 2 (``image_only_v9``) uses ``uber_model_v2`` templates with no
prefix. Training on one and evaluating on the other is out of distribution: a stage-1 4B
checkpoint trained on the stage-2 form scored 0.706 f1 on ``pixmo_point_eval_v3_mp`` against
0.815 for the released Pretrain checkpoint, losing most of it on abstention (zero-slice f1
0.718 vs 0.913) because the "Please say 'There are none.'" instruction exists only in the
stage-2 template.
"""

import numpy as np
import pytest

from olmo_core.data.multimodal.sft_formatter import SftFormatter

STAGE1 = {"prompt_templates": "none", "system_prompt": "style_and_length_v2"}
POINTS = [{"x": 10.5, "y": 20.5}, {"x": 30.0, "y": 40.0}]


def _turn(style: str, points, **family):
    fmt = SftFormatter(seed=0, **family)
    example = {"style": style, "label": "Leather Earmuff", "points": points, "point_scale": 100}
    return fmt.format_turns(example, index=3, rng=np.random.RandomState(0))[0]


@pytest.mark.parametrize("style", ["pointing", "point_count"])
def test_stage1_family_is_the_terse_prefixed_form(style: str):
    """mm_olmo ``data_formatter.py:1770-1779`` plus the ``"<style>:"`` prefix."""
    user, _ = _turn(style, POINTS, **STAGE1)
    assert user == f"{style}: leather earmuff"


@pytest.mark.parametrize("style", ["pointing", "point_count"])
def test_stage2_family_is_unchanged(style: str):
    """The default must stay the templated, unprefixed stage-2 form."""
    user, _ = _turn(style, POINTS)
    assert not user.startswith(f"{style}:")
    assert "leather earmuff" in user.lower()
    assert len(user) > len(f"{style}: leather earmuff")


def test_target_never_carries_the_prefix():
    """Only the prompt is prefixed; the answer's label stays bare in both families.

    The 08/14 checkpoint echoed ``pointing:`` back inside ``<points>...</points>`` at eval,
    which is what a model does when the prefix is unfamiliar -- not something the targets ever
    contained.
    """
    for family in ({}, STAGE1):
        _, target = _turn("pointing", POINTS, **family)
        assert target == '<points coords="1 1 105 205 2 300 400">leather earmuff</points>'


def test_absent_label_abstains_in_both_families():
    """Zero points must yield the abstention string regardless of prompt family."""
    for family in ({}, STAGE1):
        _, target = _turn("pointing", [], **family)
        assert target == "There are none."


def test_stage1_weights_pointing_tokens_like_the_released_run():
    """Stage 1 must not leave the pointing datasets on `root_subsegments`.

    The dataset classes default to `root_subsegments`, which scales an example's loss by
    1/sqrt(n_labels). The released ``Molmo2-4B-Pretrain`` sets no weighting anywhere
    (``mm_preprocessor.loss_token_weighting: None``, ``message_weight: None`` on all four
    pointing entries), so every response token counts equally. Our pointing rows average ~12.5
    labels and counting ~2.8, so the default gave that data ~3.5x / ~1.7x less gradient weight
    per token than captions -- and the factor does not cancel out of the global
    ``sum(CE*w)/sum(w)`` divisor when branch counts differ across examples.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_s1", "src/scripts/train/Molmo2-Stage1.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s1"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    assert mod.POINTING_DATASET_KWARGS["loss_token_weighting"] == "none"
    assert mod.POINTING_DATASET_KWARGS["prompt_templates"] == "none"
    assert mod.POINTING_DATASET_KWARGS["system_prompt"] == "style_and_length_v2"


def test_root_subsegments_downweights_multi_label_examples():
    """Pin the mechanism: 1/sqrt(n) per token, so a 12-label example is weighted ~0.29."""
    import numpy as np

    from olmo_core.data.multimodal.message_weight import (
        MessageWeight,
        apply_message_weight_to_loss_masks,
    )

    for n, expected in ((1, 1.0), (4, 0.5), (12, 1 / 12**0.5)):
        sub = np.repeat(np.arange(1, n + 1), 10).astype(np.int64)
        masks = np.ones(10 * n, dtype=np.float32)
        out = apply_message_weight_to_loss_masks(
            masks, sub, MessageWeight(root_subsegments=True), branch_scaling_already_applied=False
        )
        assert abs(float(out[0]) - expected) < 1e-4, f"{n} labels -> {out[0]}"
