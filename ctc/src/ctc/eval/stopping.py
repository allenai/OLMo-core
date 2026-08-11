"""
When to stop generating, and what to keep afterwards.

Stopping is where a surprising share of this project's bad numbers came from. It is pure string
handling with no model involved, yet every mistake in it looks exactly like the model failing:

* **Stopping too early.** A newline stop applied to a checkpoint that opens with ``<think>`` cuts
  the generation at the first newline *inside* the reasoning block, so the answer -- which comes
  after ``</think>`` -- is never emitted. The task reads as a total collapse.
* **Not stopping at all.** No-cot checkpoints frequently never emit EOS. Left to run to
  ``max_new_tokens`` they answer correctly and then ramble, and the ramble is what a lenient parser
  scores. This is why set-answer tasks stop at the closing ``]]``.
* **Stopping on the model clearing its throat.** Models routinely emit a formatting newline
  *before* the answer. A newline stop that fires there returns an empty string for every example --
  obliq and retrieval both scored around chance, with every generation empty, until this was found.
* **Keeping the wrong span.** When a model does emit ``<think>``, the reasoning must be stripped
  before parsing, or a parser scanning for ids finds the ones the model was *considering* rather
  than the ones it concluded with.

So the rules live here, in one place, testable without a GPU, rather than being re-derived per
evaluator. Two rules carry most of the weight:

    **A text stop never fires inside an unclosed ``<think>`` block**, and **never fires before any
    real content has been produced.**

Between them they separate "the model failed" from "we truncated it".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["StopCondition", "STOP_PRESETS", "strip_think", "apply"]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class StopCondition:
    """
    How generation ends for one task.

    :param eos: Stop when the model emits the EOS token. Always honoured, and always safe.
    :param text_stops: Substrings that end generation when they appear. Checked against the decoded
        text, and **suppressed inside an unclosed** ``<think>`` block.
    :param keep_stop: Whether the matched substring stays in the output. ``True`` for a closing
        delimiter like ``]]`` which the parser needs; ``False`` for a newline, which it does not.
    :param strip_think: Remove a ``<think>...</think>`` block before parsing.
    :param require_content: Suppress text stops until some non-whitespace content exists. Defends
        against the leading formatting newline described in the module docstring, which otherwise
        returns an empty generation for every example.
    :param require_before: Suppress text stops until this substring appears (case-insensitive).
        Used by oolong, whose answer follows a templated ``answer:`` line, so an earlier newline is
        part of the preamble rather than the end of the answer.
    :param max_new_tokens: Decode budget. Sized too small, a correct answer is truncated into a
        parse failure -- which reads as a capability limit rather than a config mistake.
    """

    eos: bool = True
    text_stops: Tuple[str, ...] = ()
    keep_stop: bool = True
    strip_think: bool = True
    require_content: bool = True
    require_before: Optional[str] = None
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not self.eos and not self.text_stops:
            raise ValueError(
                "a StopCondition with neither eos nor text_stops can only end at max_new_tokens, "
                "which lets a no-cot checkpoint ramble past a correct answer"
            )


#: Named presets, mirroring the pre-migration evaluators' ``stop`` field.
STOP_PRESETS = {
    # Set-answer tasks: the pair family (contradiction, redundancy, strmatch, mathmatch) and the
    # cycle family (cycle, groups4, textgroups). The answer is single-line JSON, so BOTH terminators
    # are needed and the earliest wins:
    #
    #   "]]"  ends a populated answer like [[1, 4], [3, 7]] exactly, and is kept because the parser
    #         needs the closing bracket.
    #   "\n"  ends an EMPTY answer -- "[]" contains no "]]", so a "]]"-only rule would never fire,
    #         the model would ramble to the budget, and parse_pairs would return None. A correct
    #         "there are no pairs" answer would be recorded as a parse failure.
    #
    # The trailing newline is harmless to the parsers, which tolerate surrounding whitespace.
    "pairs": StopCondition(text_stops=("]]", "\n"), keep_stop=True, max_new_tokens=512),
    # Short free-text answers. The answer is one line; the newline is not part of it.
    "newline": StopCondition(text_stops=("\n",), keep_stop=False, max_new_tokens=64),
    # Long structured answers (grouping, reorder) where EOS is genuinely emitted and any text stop
    # would cut a valid multi-line answer short.
    "eos": StopCondition(text_stops=(), max_new_tokens=2048),
    # oolong: the answer follows a templated "answer:" line, so a newline before that is preamble.
    "oolong": StopCondition(
        text_stops=("\n",), keep_stop=False, require_before="answer:", max_new_tokens=256
    ),
}


def _in_unclosed_think(text: str) -> bool:
    """
    Whether ``text`` currently sits inside an unclosed ``<think>`` block.

    :param text: Text generated so far.

    :returns: True when the last ``<think>`` has no matching ``</think>`` after it.
    """
    open_at = text.rfind(THINK_OPEN)
    if open_at == -1:
        return False
    return THINK_CLOSE not in text[open_at:]


def strip_think(text: str) -> str:
    """
    Drop a reasoning block, keeping what the model concluded.

    :param text: Raw generation.

    :returns: The text after ``</think>`` when present. An *unclosed* ``<think>`` returns the text
        unchanged rather than empty -- the model was cut off mid-reasoning, and returning "" would
        record that as a confident empty answer instead of a truncation.
    """
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE, 1)[1]
    return text


def should_stop(text: str, cond: StopCondition) -> Optional[int]:
    """
    Whether generation should end, given the text produced so far.

    :param text: Decoded text generated so far.
    :param cond: The task's stop condition.

    :returns: The index just past the matched stop substring (so the caller can truncate), or
        ``None`` to keep generating.
    """
    if _in_unclosed_think(text):
        # A stop token inside the reasoning block is not the end of the answer; the answer has not
        # started yet.
        return None
    # When a marker gates stopping, the search must also START after it. Gating alone is not
    # enough: the first newline in an oolong generation is in the preamble, so searching from
    # position 0 would end the answer before it began.
    search_from = 0
    if cond.require_before is not None:
        marker_at = text.lower().find(cond.require_before.lower())
        if marker_at == -1:
            return None
        search_from = marker_at + len(cond.require_before)

    best: Optional[int] = None
    for stop in cond.text_stops:
        at = text.find(stop, search_from)
        while at != -1:
            # A stop is only real once something has been said. Without this, the formatting
            # newline that models emit before answering ends generation immediately and every
            # example scores on an empty string.
            if cond.require_content and not text[:at].strip():
                at = text.find(stop, at + 1)
                continue
            end = at + len(stop) if cond.keep_stop else at
            best = end if best is None else min(best, end)
            break
    return best


def apply(text: str, cond: StopCondition) -> str:
    """
    Truncate and clean a finished generation.

    Applies the same rules the decode loop applies incrementally, so a backend that cannot check
    mid-stream (a batched or remote one) still produces the same string as the token-by-token path.
    That equivalence is what makes cross-backend score parity meaningful.

    :param text: The raw generation.
    :param cond: The task's stop condition.

    :returns: The text a parser should see.
    """
    if cond.strip_think:
        text = strip_think(text)
    at = should_stop(text, cond)
    return text if at is None else text[:at]
