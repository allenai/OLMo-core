"""
Stopping rules.

Every test here is a bug that produced a real bad number. They are written as reproductions rather
than as general property checks, because the general property ("stop at the right place") is not
what anyone got wrong -- the specific interactions were.
"""

from __future__ import annotations

import pytest

from ctc.eval.stopping import (
    STOP_PRESETS,
    StopCondition,
    apply,
    in_unclosed_think,
    should_stop,
    strip_think,
)

# ── the newline-inside-think collapse ───────────────────────────────────────────────────────────


def test_newline_stop_does_not_fire_inside_an_unclosed_think():
    """The single most important rule.

    A Qwen3.5 checkpoint opens with <think>, reasons across several lines, closes, then answers.
    Stopping at the first newline cuts it mid-reasoning and yields no answer at all -- which read
    as a total task collapse rather than as a truncation bug.
    """
    text = "<think>\nLet me check each claim.\nClaim 1 says X."
    assert should_stop(text, STOP_PRESETS["newline"]) is None


def test_newline_stop_fires_once_think_is_closed():
    text = "<think>\nreasoning\n</think>Answer: Paris\nand then rambling"
    out = apply(text, STOP_PRESETS["newline"])
    assert out == "Answer: Paris"


def test_newline_stop_fires_normally_without_any_think():
    assert apply("Answer: Paris\nrambling", STOP_PRESETS["newline"]) == "Answer: Paris"


def test_a_second_think_block_reopens_suppression():
    """rfind, not find: only the LAST <think> decides whether we are currently inside one."""
    text = "<think>a</think>ok\n<think>reconsidering\nmore"
    assert should_stop(text, STOP_PRESETS["newline"]) is None


# ── the rambling no-cot checkpoint ──────────────────────────────────────────────────────────────


def test_pairs_stop_terminates_a_rambling_generation():
    """No-cot checkpoints frequently never emit EOS: they answer, then keep talking."""
    text = "[[1, 4], [3, 7]] and here are some further thoughts about the claims"
    assert apply(text, STOP_PRESETS["pairs"]) == "[[1, 4], [3, 7]]"


def test_pairs_stop_keeps_the_closing_bracket():
    """The parser needs it; dropping it turns a valid answer into a parse failure."""
    assert apply("[[1, 4]]", STOP_PRESETS["pairs"]).endswith("]]")


def test_newline_stop_drops_its_own_newline():
    assert "\n" not in apply("Answer: Paris\nmore", STOP_PRESETS["newline"])


def test_earliest_stop_wins_when_several_match():
    cond = StopCondition(text_stops=("]]", "\n"), keep_stop=False)
    assert apply("abc\ndef]]ghi", cond) == "abc"


# ── the leading formatting newline ──────────────────────────────────────────────────────────────


def test_leading_newline_does_not_end_generation():
    """Models clear their throat with a newline. Stopping there emptied EVERY generation, and
    obliq and retrieval both scored around chance until it was found."""
    assert apply("\nAnswer: Paris\nrambling", STOP_PRESETS["newline"]) == "\nAnswer: Paris"


def test_several_leading_blank_lines_are_skipped():
    assert apply("\n\n  \nAnswer: Paris\nmore", STOP_PRESETS["newline"]).strip() == "Answer: Paris"


def test_a_generation_of_only_whitespace_does_not_stop_early():
    """Nothing was said, so there is nothing to terminate -- let the budget end it."""
    assert should_stop("\n\n  ", STOP_PRESETS["newline"]) is None


def test_require_content_can_be_disabled():
    cond = StopCondition(text_stops=("\n",), keep_stop=False, require_content=False)
    assert apply("\nAnswer", cond) == ""


# ── oolong's templated answer line ──────────────────────────────────────────────────────────────


def test_oolong_ignores_newlines_before_the_answer_line():
    text = "Counting the items.\nStill counting.\nanswer: 42\ntrailing"
    assert apply(text, STOP_PRESETS["oolong"]).endswith("answer: 42")


def test_oolong_marker_is_case_insensitive():
    assert apply("x\nAnswer: 42\nmore", STOP_PRESETS["oolong"]).endswith("Answer: 42")


def test_oolong_does_not_stop_before_the_marker_appears():
    assert should_stop("thinking\nmore thinking\n", STOP_PRESETS["oolong"]) is None


def test_oolong_regression_examples_recover_their_answer():
    """Generations a reviewer found scoring 0.0, run through the same public entry point the
    harness uses (:func:`apply` with the ``oolong`` preset) plus the oolong spec's own
    :func:`~ctc.tasks.oolong.spec.parse`.

    The first two: a no-cot checkpoint answers and then keeps going, echoing a corpus line or a
    second question. Every oolong corpus line carries ``User:``, so a single whole-string
    should_stop() anchored on the echo as the LAST marker and the echo was graded. The decode loop
    never sees the echo -- it stopped at the newline after the answer -- and apply() replays that
    loop, so neither does the grader. The third: a newline immediately after the marker is
    formatting, not the end of the answer.
    """
    from ctc.tasks.oolong import spec

    cond = STOP_PRESETS["oolong"]
    for text, want in [
        ("Answer: 1\nDate: Apr 27, 2025 || User: 28461 || Instance: a corpus line", "1"),
        ("Answer: 1\n\nFor the following question, give a number.\nAnswer: 7\n", "1"),
        ("Answer:\n1", "1"),
    ]:
        assert spec.parse(apply(text, cond)) == want, text


def test_a_preamble_naming_the_marker_word_stops_where_the_decode_loop_would():
    """The price of replaying the decode loop. A preamble sentence that happens to contain the
    marker word anchors the stop search, and the newline closing that sentence fires -- before the
    real answer line. Incremental decoding does exactly the same, since it cannot know an answer
    line is still coming. Agreement between the two paths is worth more than this case: without it
    the whole-string path grades a different span than the one the decode loop produced, and
    cross-backend parity means nothing. Pinned so that a future change here is a deliberate one.
    """
    cond = STOP_PRESETS["oolong"]
    text = "Counting each user: there are several.\nAnswer: 5"
    assert apply(text, cond) == "Counting each user: there are several."


# ── think stripping ─────────────────────────────────────────────────────────────────────────────


def test_think_is_stripped_before_parsing():
    """Otherwise a parser finds the ids the model CONSIDERED, not the ones it concluded with."""
    text = "<think>maybe [[9, 9]]?</think>[[1, 4]]"
    assert apply(text, STOP_PRESETS["pairs"]) == "[[1, 4]]"


def test_unclosed_think_is_kept_not_emptied():
    """Returning '' would record a truncation as a confident empty answer."""
    assert strip_think("<think>cut off mid-thought") == "<think>cut off mid-thought"


def test_in_unclosed_think_is_the_public_name_for_the_internal_predicate():
    """A downstream harness (allenai/olmo-eval vendors this module) needs to ask the same question

    should_stop asks internally -- is a token budget about to hit while reasoning is still open? --
    so it can treat a truncated <think> as a parse failure rather than scoring the ids the model was
    merely considering. Exposed as a thin public wrapper rather than renaming the private helper, so
    no existing call site's behaviour changes.
    """
    assert in_unclosed_think("<think>still reasoning") is True
    assert in_unclosed_think("<think>done</think>the answer") is False
    assert in_unclosed_think("no think block at all") is False


def test_strip_think_keeps_only_what_follows_the_close():
    assert strip_think("<think>reasoning</think> the answer") == " the answer"


def test_strip_can_be_disabled():
    cond = StopCondition(text_stops=("]]",), strip_think=False)
    assert apply("<think>x</think>[[1, 4]]", cond).startswith("<think>")


# ── validation ──────────────────────────────────────────────────────────────────────────────────


def test_a_condition_that_can_only_hit_the_budget_is_rejected():
    """That combination is exactly how a no-cot checkpoint rambles past a correct answer."""
    with pytest.raises(ValueError, match="ramble"):
        StopCondition(eos=False, text_stops=())


def test_zero_budget_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        StopCondition(max_new_tokens=0)


# ── incremental and whole-string paths agree ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("preset", "text"),
    [
        ("pairs", "[[1, 4], [3, 7]] trailing"),
        ("pairs", "<think>r\ne\na</think>[[1, 4]] trailing"),
        ("pairs", "no stop here at all"),
        ("pairs", ""),
        ("pairs", "<think>unclosed and rambling"),
        ("pairs", "```json\n[[1,\n 4]]\n```\nmore"),
        ("newline", "\n\nanswer\nmore"),
        # require_before: the marker that anchors the search must be the one the loop had seen,
        # not the last one in the finished string.
        ("oolong", "Answer: 1\nDate: Apr 27, 2025 || User: 28461 || Instance: x"),
        ("oolong", "Answer: 1\n\nFor the following question, give a number.\nAnswer: 7\n"),
        ("oolong", "Counting each user: there are several.\nAnswer: 5"),
        ("oolong", "Answer:\n1"),
        ("oolong", "The most common label: spam, ahead of ham.\nLabel: spam"),
        ("outliers", "Outliers: the minority topic is X.\nOutliers: [7], [10], [12]"),
        ("outliers", "Most are about X; the odd ones are about Y.\nOutliers: [3]\nand more"),
    ],
)
def test_apply_matches_incremental_truncation(preset, text):
    """A batched or remote backend cannot check mid-stream; it must still get the same string.

    Without this equivalence, cross-backend score parity would not mean anything.
    """
    assert apply(text, STOP_PRESETS[preset]) == _char_by_char(text, STOP_PRESETS[preset])


def _char_by_char(text: str, cond: StopCondition) -> str:
    """The reference: a loop that re-checks after every character and stops the moment the rule
    fires. Slow, obviously right, and what :func:`apply` has to agree with."""
    stripped = strip_think(text) if cond.strip_think else text
    for i in range(len(stripped) + 1):
        at = should_stop(stripped[:i], cond)
        if at is not None:
            return stripped[:i][:at]
    return stripped


#: The substrings the stop rules react to. Random strings over exactly these reach the awkward
#: interleavings -- a stop inside a reopened think block, a marker after a stop, ``]]]`` -- that
#: hand-written fixtures do not.
_STOP_RULE_TOKENS = (
    "\n", "]]", "[", "]", "```", "<think>", "</think>",
    "answer:", "label:", "user:", "outliers:", "a", "b", " ",
)  # fmt: skip


@pytest.mark.parametrize("name", sorted(STOP_PRESETS))
def test_apply_matches_a_character_by_character_replay_on_random_text(name):
    """Hand-picked fixtures proved too polite: the first replay implementation passed every one of
    them while a randomized search found hundreds of disagreements -- overlapping ``]]]`` counted
    once, and a second ``<think>`` block lifting suppression from an earlier stop at a prefix that
    ends at no stop substring. Seeded, so a failure reproduces.
    """
    import random

    rng = random.Random(0)
    cond = STOP_PRESETS[name]
    for _ in range(1500):
        length = rng.choice((0, 1, 2, 3, 5, 8, 12, 20, 30))
        text = "".join(rng.choice(_STOP_RULE_TOKENS) for _ in range(length))
        assert apply(text, cond) == _char_by_char(text, cond), repr(text)


# ── presets are sane ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(STOP_PRESETS))
def test_presets_have_a_positive_budget(name):
    assert STOP_PRESETS[name].max_new_tokens > 0


def test_eos_preset_has_no_text_stop():
    """grouping/reorder answers are legitimately multi-line; any text stop would cut them."""
    assert STOP_PRESETS["eos"].text_stops == ()
    assert STOP_PRESETS["eos"].eos
