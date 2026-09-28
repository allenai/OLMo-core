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


def test_oolong_regression_examples_all_recover_their_answer():
    """Four generations a reviewer found scoring 0.0 because require_before anchored on the

    EARLIEST marker occurrence instead of the LAST (oolong's own parser reads the last), and because
    a newline immediately after the marker was not skipped before the stop search began. Run
    through the same public entry point the harness uses (:func:`apply` with the ``oolong`` preset)
    plus the oolong spec's own :func:`~ctc.tasks.oolong.spec.parse`.

    Only these two use the literal "answer:" marker the ``oolong`` preset gates on; the other two
    reviewer examples ("Label:", "Outliers:") use different templated markers and are covered by
    :func:`test_require_before_anchors_on_the_last_occurrence_for_any_marker` below, which exercises
    the same should_stop fix generically rather than tying it to one hardcoded marker string.
    """
    from ctc.tasks.oolong import spec

    cond = STOP_PRESETS["oolong"]
    for text, want in [
        # (a): marker echoed in the preamble
        ("Counting each user: there are several.\nAnswer: 5", "5"),
        # (b): newline immediately after the marker
        ("Answer:\n1", "1"),
    ]:
        assert spec.parse(apply(text, cond)) == want


def test_require_before_anchors_on_the_last_occurrence_for_any_marker():
    """The last-occurrence fix is a property of should_stop/require_before in general, not special-

    cased to the literal "answer:" marker the ``oolong`` preset happens to use. A StopCondition
    shaped like that preset but gated on a different templated marker ("Label:", "Outliers:", from
    the same reviewer report) must likewise ignore an echo of the marker word in the preamble: with
    the old earliest-wins rule, the newline that closes the preamble sentence ended generation before
    the real answer line was ever reached.
    """
    label_cond = StopCondition(text_stops=("\n",), keep_stop=False, require_before="label:")
    text = "The most common label: spam, ahead of ham.\nLabel: spam"
    assert apply(text, label_cond) == text  # not truncated after "...ham."

    outliers_cond = StopCondition(text_stops=("\n",), keep_stop=False, require_before="outliers:")
    text = "Outliers: the minority topic is X.\nOutliers: [7], [10], [12]"
    assert apply(text, outliers_cond) == text  # not truncated after "...topic is X."


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
    "text",
    [
        "[[1, 4], [3, 7]] trailing",
        "<think>r\ne\na</think>[[1, 4]] trailing",
        "no stop here at all",
        "",
        "<think>unclosed and rambling",
    ],
)
def test_apply_matches_incremental_truncation(text):
    """A batched or remote backend cannot check mid-stream; it must still get the same string.

    Without this equivalence, cross-backend score parity would not mean anything.
    """
    cond = STOP_PRESETS["pairs"]

    # Simulate a token-by-token loop that stops the moment the rule fires.
    stripped = strip_think(text) if cond.strip_think else text
    incremental = stripped
    for i in range(len(stripped) + 1):
        at = should_stop(stripped[:i], cond)
        if at is not None:
            incremental = stripped[:i][:at]
            break

    assert incremental == apply(text, cond)


# ── presets are sane ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(STOP_PRESETS))
def test_presets_have_a_positive_budget(name):
    assert STOP_PRESETS[name].max_new_tokens > 0


def test_eos_preset_has_no_text_stop():
    """grouping/reorder answers are legitimately multi-line; any text stop would cut them."""
    assert STOP_PRESETS["eos"].text_stops == ()
    assert STOP_PRESETS["eos"].eos
