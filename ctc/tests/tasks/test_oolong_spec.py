"""
``ctc.tasks.oolong.spec`` -- the numeric answer's comma-grouping edge case.

The NUMERIC branch of :func:`~ctc.tasks.oolong.spec.score` finds the last number-looking substring
in the parsed text and diffs it against gold. A plain ``-?\\d+\\.?\\d*`` regex splits a thousands-
grouped answer like ``"1,234"`` at the comma, so ``nums[-1]`` was ``"234"`` -- the last three digits
of the real answer -- rather than the number the model actually gave.
"""

from __future__ import annotations

import pytest

from ctc.tasks.oolong import spec


def _numeric_score(parsed: str, gold: float) -> float:
    return spec.score(parsed, {"answers": [gold], "_meta": {"answer_type": "NUMERIC"}})["score"]


@pytest.mark.parametrize(
    "parsed, gold",
    [
        ("1,234", 1234),
        ("1234", 1234),
        ("-5", -5),
        ("3.5", 3.5),
    ],
)
def test_numeric_answer_parses_exactly(parsed, gold):
    assert _numeric_score(parsed, gold) == 1.0


def test_comma_grouped_thousands_are_not_truncated_at_the_comma():
    """The regression case: a plain digit-run regex reads "1,234" as "234", off by 1000."""
    assert _numeric_score("The total is 1,234 items.", 1234) == 1.0


def test_a_comma_separated_list_of_numbers_is_not_merged_into_one():
    """ "3, 5, 7" must be read as three separate numbers -- 3, 5, and 7 -- not merged into 357 or
    3457. Each number here is spaced and fewer than the three digits the thousands-grouping shape
    requires, so it never matches that alternative; the LAST of the three numbers is what the
    NUMERIC scoring rule reads, exactly as it did before this fix.
    """
    assert _numeric_score("The counts were 3, 5, 7", 7) == 1.0
    # Confirm it is genuinely reading the last number, not a merged one.
    assert _numeric_score("The counts were 3, 5, 7", 357) != 1.0


def test_decimal_and_negative_numeric_answers_are_unaffected():
    assert _numeric_score("The change was -5 units.", -5) == 1.0
    assert _numeric_score("The average was 3.5 per day.", 3.5) == 1.0


def test_numeric_partial_credit_still_decays_geometrically_off_a_comma_grouped_number():
    """A comma-grouped number that is off by one point should score like any other off-by-one --

    0.75 -- not like a wildly wrong number the way a truncated "234" (off by 1000) would.
    """
    got = spec.score("1,235", {"answers": [1234], "_meta": {"answer_type": "NUMERIC"}})
    assert got["score"] == pytest.approx(0.75)
    assert got["exact_match"] == 0.0
