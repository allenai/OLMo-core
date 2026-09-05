"""Keep sequential comparisons isolated and their output paths unambiguous."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "examples/olmo_ddp/olmoe3_profile_node.py"
_SPEC = importlib.util.spec_from_file_location("profile_node", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_legacy_plan():
    assert _MODULE.profile_plan(["baseline", "kda-128"], ["timing"]) == [
        ("baseline", "timing"),
        ("kda-128", "timing"),
    ]


def test_explicit_plan():
    assert _MODULE.profile_plan([], [], "baseline:timing,kda-128:timing,kda-128:torch") == [
        ("baseline", "timing"),
        ("kda-128", "timing"),
        ("kda-128", "torch"),
    ]


def test_repeated_names_do_not_change_variants():
    pairs = [("baseline", "timing"), ("kda-128", "timing")]
    assert _MODULE.named_profile_plan("aa", pairs, 2) == [
        ("aa-repeat1-baseline", "baseline", "timing"),
        ("aa-repeat1-kda-128", "kda-128", "timing"),
        ("aa-repeat2-baseline", "baseline", "timing"),
        ("aa-repeat2-kda-128", "kda-128", "timing"),
    ]
    assert _MODULE.named_profile_plan("old", [("baseline", "timing")]) == [
        ("old", "baseline", "timing")
    ]
    for invalid in (0, 5):
        with pytest.raises(ValueError):
            _MODULE.named_profile_plan("aa", pairs, invalid)


@pytest.mark.parametrize(
    "plan", ["x:timing,x:timing", "../x:timing", "x:typo", "x", "x:timing:other"]
)
def test_invalid_plan(plan):
    with pytest.raises(ValueError):
        _MODULE.profile_plan([], [], plan)
