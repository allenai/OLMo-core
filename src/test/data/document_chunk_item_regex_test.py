"""Regression test for the OOLONG ``--item-regex`` leak (``debug/ctc_vllm_validation/
CHUNK_LEAK_AUDIT.md``, 2026-07-23; fixed 2026-07-26).

Several converters passed ``--item-regex ||`` to
``src/scripts/data/convert_unified_to_document_landmark.py``. As a *regex* ``||`` is an alternation
of empty branches, so it matches **every** line. In ``--chunk-by line`` mode that wrapped the
instruction, question and data-header lines as their own chunks and left the blank lines between
them FREE -- a global bridge between otherwise-isolated chunks, and a train/eval layout mismatch
(the native eval keeps the preamble FREE and wraps only data items). The audit measured 2019
inter-chunk FREE tokens in ``oolong_train``, ~5 per example, while every other task measured 0.

The converter's default has always been the correctly-escaped ``r"\\|\\|"``; the fix was to stop
overriding it and to reject any ``--item-regex`` that matches the empty string.

Does not require a tokenizer: exercises ``_wrap_item_lines`` directly on a prompt string.
"""

import re

from olmo_core.data.document_chunk_landmark import DOC_END_STR, DOC_START_STR, _wrap_item_lines

#: The converter's default ``--item-regex`` -- a literal ``||``, properly escaped.
GOOD_ITEM_REGEX = r"\|\|"
#: What four launchers passed before the fix. Matches every line.
BAD_ITEM_REGEX = "||"

OOLONG_PROMPT = """Read the data below and answer the question.

Question: how many entries mention rain?

The following lines are the data:
Date: 2021-03-04 || Temp: 15C || Note: rain
Date: 2021-03-05 || Temp: 18C || Note: clear
Date: 2021-03-06 || Temp: 12C || Note: rain

Question: how many entries mention rain?"""

N_DATA_ITEMS = 3


def _inter_chunk_free_chars(wrapped: str) -> int:
    """Count FREE characters strictly between two consecutive chunks -- the audit's leak metric.

    Text before the first chunk and after the last is the legitimate free prefix/suffix.
    """
    gaps = re.findall(
        re.escape(DOC_END_STR) + "(.*?)" + re.escape(DOC_START_STR), wrapped, flags=re.S
    )
    return sum(len(g) for g in gaps)


def test_default_item_regex_wraps_only_data_items_with_no_inter_chunk_free_tokens():
    wrapped = _wrap_item_lines(
        OOLONG_PROMPT, re.compile(GOOD_ITEM_REGEX), DOC_START_STR, DOC_END_STR
    )
    assert wrapped.count(DOC_START_STR) == N_DATA_ITEMS, (
        "only the data-item lines should be chunk-wrapped; the instruction, question and "
        "data-header lines must stay FREE so train matches the native eval layout"
    )
    assert _inter_chunk_free_chars(wrapped) == 0, (
        "FREE tokens between consecutive chunks bridge otherwise-isolated documents"
    )


def test_bare_pipe_item_regex_reproduces_the_leak():
    """Sanity-check the regression itself: the pre-fix ``'||'`` wraps the preamble too."""
    wrapped = _wrap_item_lines(
        OOLONG_PROMPT, re.compile(BAD_ITEM_REGEX), DOC_START_STR, DOC_END_STR
    )
    assert wrapped.count(DOC_START_STR) > N_DATA_ITEMS, "expected the preamble to be wrapped too"
    assert _inter_chunk_free_chars(wrapped) > 0, "expected FREE gaps bridging the chunks"


def test_bare_pipe_item_regex_matches_the_empty_string():
    """The exact condition the converter now rejects at startup."""
    assert re.compile(BAD_ITEM_REGEX).search("") is not None
    assert re.compile(GOOD_ITEM_REGEX).search("") is None
