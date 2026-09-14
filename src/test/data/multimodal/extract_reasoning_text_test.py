"""Tests for :func:`~olmo_core.data.multimodal.sft_common.extract_reasoning_text`.

ChartVerse ``cot_solution`` and MMFineReason ``original_answer`` share the
``<think>…</think> … <answer>…</answer>`` markup, so one helper covers both. The cases that
matter are the ones that would otherwise corrupt supervision silently: markup leaking into
the target (which the CharXiv judge would then grade), and a derivation that never states
the graded answer.
"""

import pytest

from olmo_core.data.multimodal.sft_common import extract_reasoning_text

CHARTVERSE_SHAPE = (
    "<think>Okay, let's tackle this. Company A 2015 is ~50, G is ~5.\n"
    "The ratio is 50/5 = 10.</think>\n\n"
    "Therefore, the final answer is <answer>4.0</answer>."
)


def test_keeps_derivation_and_drops_markup():
    out = extract_reasoning_text(CHARTVERSE_SHAPE, final_answer="4.0")
    assert "Company A 2015 is ~50" in out
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        assert tag not in out


def test_appends_final_answer_when_the_trace_omits_it():
    out = extract_reasoning_text("<think>Read the bars, then subtract.</think>", final_answer="4.0")
    assert out.endswith("Final answer: 4.0")


def test_does_not_duplicate_an_answer_already_in_the_tail():
    raw = "<think>Sum the two segments, giving 30%, first crossed in 2013.</think>"
    out = extract_reasoning_text(raw, final_answer="2013")
    assert out.count("2013") == 1


def test_recovers_the_answer_from_the_tag_when_none_is_passed():
    out = extract_reasoning_text("<think>Work.</think><answer>7</answer>", final_answer="")
    assert out == "Work.\n\nFinal answer: 7"


def test_untagged_trace_passes_through():
    out = extract_reasoning_text("Step 1. Step 2.", final_answer="9")
    assert out == "Step 1. Step 2.\n\nFinal answer: 9"


@pytest.mark.parametrize("raw", [None, "", "   ", "<think></think>"])
def test_empty_trace_yields_the_answer_or_nothing(raw):
    assert extract_reasoning_text(raw, final_answer="") == ""
    assert extract_reasoning_text(raw, final_answer="4.0") in ("", "4.0")


def test_stray_unpaired_tags_are_removed():
    out = extract_reasoning_text("Partial </think> trace <answer>", final_answer="1")
    assert "<" not in out and ">" not in out
