import pytest

from ctc.format import rungs


@pytest.mark.parametrize(
    "label,tokens",
    [
        ("2k", 2048),
        ("4k", 4096),
        ("32k", 32768),
        ("512k", 524288),
        ("1m", 1048576),
        ("2048", 2048),
        (" 8k ", 8192),
        ("8K", 8192),
    ],
)
def test_parse_rung(label, tokens):
    assert rungs.parse_rung(label) == tokens


@pytest.mark.parametrize("bad", ["", "k", "7q", "2 k b", "-4k", "2.5k"])
def test_parse_rung_rejects_junk(bad):
    with pytest.raises(ValueError):
        rungs.parse_rung(bad)


def test_labels_are_binary_not_decimal():
    """32k must be 32768, matching the rung_<n>.jsonl filenames the data pipeline writes.

    If this were 32000, eval would look for a rung file the builder never produced.
    """
    assert rungs.parse_rung("32k") == 32 * 1024


def test_round_trip():
    for label in ("2k", "128k", "512k", "1m"):
        assert rungs.normalize(label) == label


def test_bare_count_normalizes_to_a_label():
    assert rungs.normalize("2048") == "2k"
    assert rungs.normalize("1500") == "1500"  # not a clean multiple, stays bare


def test_sort_is_by_length_not_lexical():
    """Lexical sort puts '128k' before '2k' and silently reorders every ladder plot."""
    assert rungs.sort_rungs(["128k", "2k", "32k", "8k"]) == ["2k", "8k", "32k", "128k"]


def test_sort_dedupes_across_equivalent_spellings():
    assert rungs.sort_rungs(["2k", "2048", "4k"]) == ["2k", "4k"]
