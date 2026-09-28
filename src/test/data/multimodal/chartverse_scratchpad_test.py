"""ChartVerse derivation targets: scratchpad form, length gate, and the implied-flag rule."""

import pytest

from olmo_core.data.multimodal.chartverse import (
    ChartVerseDatasetConfig,
    build_supervision_target,
)

RAW = "<think>Read the two smallest segments, sum them per year, first year over 30%.</think>" \
      "<answer>2011</answer>"


def test_prose_form_ends_in_final_answer():
    t = build_supervision_target(RAW, "2011", scratchpad=False)
    assert "<think>" not in t and "</think>" not in t
    assert t.rstrip().endswith("2011")


def test_scratchpad_form_keeps_envelope_and_bare_answer():
    t = build_supervision_target(RAW, "2011", scratchpad=True)
    assert t.startswith("<think>") and "</think>" in t
    # the graded string after the strip is exactly the short answer
    assert t.split("</think>", 1)[1] == "2011"


def test_over_length_derivation_is_skipped_not_downgraded():
    with pytest.raises(ValueError, match="max_cot_chars"):
        build_supervision_target(RAW, "2011", scratchpad=True, max_cot_chars=10)
    # under the cap it is unchanged
    assert build_supervision_target(RAW, "2011", scratchpad=True, max_cot_chars=len(RAW))


@pytest.mark.parametrize("raw", [None, "", "<think></think><answer>2011</answer>"])
def test_empty_or_bodyless_derivation_is_skipped(raw):
    with pytest.raises(ValueError):
        build_supervision_target(raw, "2011", scratchpad=True)


def test_scratchpad_implies_supervise_cot_and_skip_overlong():
    cfg = ChartVerseDatasetConfig(cot_scratchpad=True)
    assert cfg.supervise_cot and cfg.skip_overlong
    plain = ChartVerseDatasetConfig()
    assert not plain.supervise_cot and not plain.skip_overlong
