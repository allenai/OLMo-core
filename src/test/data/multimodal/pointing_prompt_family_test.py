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
