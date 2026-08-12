"""
Rung labels: the shared vocabulary for context-length ladder steps.

A *rung* is a context-size budget (``"32k"`` = 32768 tokens), realised per task as however many
documents fit that budget -- so the same rung means a different document count for every task, and
that mapping lives in :data:`ctc.data.ladders.LADDERS`, not here.

This module is deliberately tiny and deliberately shared. Data generation writes ``rung_<tokens>.jsonl``
and evaluation selects it, so if the two disagreed about what ``"32k"`` means, eval would silently
grade a file built for a different budget. There is no fixed set of legal rungs: tasks range from nq
(currently 2k/4k/8k) to the xlong ladders (up to 512k), and a new one must not require a code change.
"""

from __future__ import annotations

import re
from typing import Iterable, List

__all__ = ["parse_rung", "format_rung", "sort_rungs", "normalize"]

_RUNG_RE = re.compile(r"^(\d+)\s*([kmKM]?)$")
_SUFFIX = {"": 1, "k": 1024, "m": 1024 * 1024}


def parse_rung(label: str) -> int:
    """
    Convert a rung label to a token count.

    :param label: ``"2k"``, ``"512k"``, ``"1m"``, or a bare token count like ``"2048"``.
        Binary units: ``"32k"`` is 32768, not 32000, matching the ``rung_<n>.jsonl`` filenames
        the data pipeline writes.

    :returns: The context budget in tokens.

    :raises ValueError: If the label is not a recognised rung.
    """
    m = _RUNG_RE.match(label.strip())
    if not m:
        raise ValueError(
            f"bad rung label {label!r}; expected forms like '2k', '128k', '1m', or '2048'"
        )
    digits, suffix = m.group(1), m.group(2).lower()
    return int(digits) * _SUFFIX[suffix]


def format_rung(tokens: int) -> str:
    """
    Convert a token count back to its canonical label.

    :param tokens: Context budget in tokens.

    :returns: ``"32k"`` for exact binary multiples, otherwise the bare count.
    """
    for suffix, size in (("m", _SUFFIX["m"]), ("k", _SUFFIX["k"])):
        if tokens >= size and tokens % size == 0:
            return f"{tokens // size}{suffix}"
    return str(tokens)


def normalize(label: str) -> str:
    """
    Round-trip a label through its token count, so ``"2048"`` and ``"2k"`` compare equal.

    :param label: Any accepted rung label.

    :returns: The canonical label.
    """
    return format_rung(parse_rung(label))


def sort_rungs(labels: Iterable[str]) -> List[str]:
    """
    Sort rung labels by actual context length.

    Lexical sort puts ``"128k"`` before ``"2k"``, which silently reorders every ladder plot.

    :param labels: Rung labels in any order.

    :returns: Canonical labels, ascending by token count, duplicates removed.
    """
    seen = {}
    for label in labels:
        seen[parse_rung(label)] = None
    return [format_rung(tokens) for tokens in sorted(seen)]
