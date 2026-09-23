"""Tests for the delimited-scratchpad trace renderer.

The invariant that matters is the round trip: whatever this writes, olmo-eval's
``strip_reasoning_trace`` must recover exactly the committed answer. If the two ever drift
apart the trace reaches the GPT judge and the arm silently measures the opposite of what it
intends -- which is the failure mode this option exists to avoid.
"""

import re

from olmo_core.data.multimodal.sft_common import (
    extract_reasoning_scratchpad,
    extract_reasoning_text,
)

TRACE = "<think>Step one. Step two.</think><answer>42</answer>"


def _strip(text: str) -> str:
    """Mirror of olmo_eval.evals.extract.reasoning.strip_reasoning_trace."""
    if not text or not ("<think>" in text or "</think>" in text or "<answer>" in text):
        return text
    if "</think>" in text:
        answer = re.sub(r"(?s).*</think>", "", text)
    elif "<think>" in text:
        answer = ""
    else:
        answer = text
    answer = re.sub(r"(?s)^\s*<answer>\s*", "", answer.strip())
    answer = re.sub(r"(?s)\s*</answer>\s*$", "", answer)
    return answer.strip()


def test_keeps_the_trace_delimited_and_the_answer_bare():
    out = extract_reasoning_scratchpad(TRACE, final_answer="42")
    assert out == "<think>Step one. Step two.</think>42"
    assert _strip(out) == "42"


def test_prefers_the_supplied_short_answer_over_the_embedded_one():
    # The point of the option: grade a 1-character answer, hide the derivation.
    out = extract_reasoning_scratchpad(TRACE, final_answer="7")
    assert out == "<think>Step one. Step two.</think>7"
    assert _strip(out) == "7"


def test_falls_back_to_the_embedded_answer_when_none_supplied():
    assert extract_reasoning_scratchpad(TRACE) == "<think>Step one. Step two.</think>42"


def test_returns_empty_without_a_trace_so_the_caller_uses_the_answer_only_target():
    # ~51% of MMFineReason rows have no <think> block and must behave exactly as today.
    assert extract_reasoning_scratchpad("just an answer", final_answer="42") == ""
    assert extract_reasoning_scratchpad("", final_answer="42") == ""
    assert extract_reasoning_scratchpad("<think>  </think><answer>42</answer>") == ""
    assert extract_reasoning_scratchpad(TRACE, final_answer="") != ""


def test_no_stray_tags_survive_into_the_loss():
    messy = "<think>a <answer> b</think><answer>4<think>2</answer>"
    out = extract_reasoning_scratchpad(messy)
    body, _, answer = out.partition("</think>")
    assert "<answer>" not in body and "<think>" not in body[len("<think>") :]
    assert "<" not in answer


def test_differs_from_the_prose_renderer_exactly_in_the_delimiters():
    prose = extract_reasoning_text(TRACE, final_answer="42")
    pad = extract_reasoning_scratchpad(TRACE, final_answer="42")
    assert "<think>" not in prose and "<think>" in pad
    # The prose form is graded in full; the scratchpad form grades only the answer.
    assert _strip(prose) == prose
    assert _strip(pad) == "42"
