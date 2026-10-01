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

import re
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "StopCondition",
    "STOP_PRESETS",
    "OOLONG_ANSWER_MARKERS",
    "strip_think",
    "in_unclosed_think",
    "apply",
]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
FENCE = "```"

#: Every marker an oolong question templates its answer line with. The question text says e.g.
#: "Give your final answer in the form 'Label: answer'", so a model that complies emits
#: ``Label: True`` -- which is neither stopped nor parsed by an ``answer:``-only rule. Over the
#: r2k split, 184/500 questions template ``Label:`` and 12/500 ``User:``; scoring those against
#: an ``answer:``-only marker records a correct answer as a miss.
OOLONG_ANSWER_MARKERS = ("answer:", "label:", "user:")


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
    :param require_before: Suppress text stops until one of these substrings appears
        (case-insensitive); the **last** match in the text so far also becomes the point the stop
        search starts from, matching what the task's parser reads -- see :func:`should_stop`, and
        :func:`apply` for why that is only safe on a growing prefix. Used where the answer
        follows a templated marker line, so an earlier newline is part of the preamble rather than
        the end of the answer: oolong's questions template three different markers
        (:data:`OOLONG_ANSWER_MARKERS`), and outlier's instruction mandates a sentence *before* the
        ``Outliers:`` line.
    :param bracketed_answer: The answer is a bracketed JSON literal. Suppresses text stops
        while a ``[`` is still unclosed, and while a markdown code fence is still open -- models
        routinely pretty-print the literal across lines or wrap it in ```` ```json ````, and a
        newline stop then ends the generation on the fence line. Both conditions are computed from
        the prefix alone, so the incremental and whole-string paths still agree.
    :param max_new_tokens: Decode budget. Sized too small, a correct answer is truncated into a
        parse failure -- which reads as a capability limit rather than a config mistake.
    """

    eos: bool = True
    text_stops: Tuple[str, ...] = ()
    keep_stop: bool = True
    strip_think: bool = True
    require_content: bool = True
    require_before: Tuple[str, ...] = ()
    bracketed_answer: bool = False
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
    "pairs": StopCondition(
        text_stops=("]]", "\n"), keep_stop=True, bracketed_answer=True, max_new_tokens=512
    ),
    # Short free-text answers. The answer is one line; the newline is not part of it.
    "newline": StopCondition(text_stops=("\n",), keep_stop=False, max_new_tokens=64),
    # Long structured answers (grouping, reorder) where EOS is genuinely emitted and any text stop
    # would cut a valid multi-line answer short.
    "eos": StopCondition(text_stops=(), max_new_tokens=2048),
    # oolong: the answer follows a templated marker line, so a newline before that is preamble.
    "oolong": StopCondition(
        text_stops=("\n",),
        keep_stop=False,
        require_before=OOLONG_ANSWER_MARKERS,
        max_new_tokens=256,
    ),
    # outlier: the instruction MANDATES a sentence naming the majority and outlier attributes
    # before the "Outliers:" line, so the first newline is the end of that sentence and not the
    # end of the answer. Under a plain newline stop the ids never reach the parser at all.
    "outliers": StopCondition(
        text_stops=("\n",),
        keep_stop=False,
        require_before=("outliers:",),
        max_new_tokens=256,
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


def _in_open_fence(text: str) -> bool:
    """
    Whether ``text`` currently sits inside an unclosed markdown code fence.

    :param text: Text generated so far.

    :returns: True when an odd number of ``````` markers have been emitted.
    """
    return text.count(FENCE) % 2 == 1


def _in_open_bracket(text: str) -> bool:
    """
    Whether ``text`` currently sits inside an unclosed ``[``.

    Brackets inside a JSON string would miscount, but a pair/cycle answer contains only integers,
    so counting is enough and stays O(n) on the prefix.

    :param text: Text generated so far.

    :returns: True when more ``[`` than ``]`` have been emitted.
    """
    return text.count("[") > text.count("]")


def in_unclosed_think(text: str) -> bool:
    """
    Public wrapper for :func:`_in_unclosed_think`.

    A downstream harness (allenai/olmo-eval vendors this module) needs to ask the same question
    this module asks internally when deciding whether a stop should fire: is the generation
    currently truncated mid-reasoning? A harness that hits its own token budget while a ``<think>``
    block is still open should treat that generation as a parse failure -- there is no concluded
    answer yet, only the ids and claims the model was *considering* -- rather than handing the
    truncated reasoning to a parser and scoring whatever it happens to find there.

    :param text: Text generated so far (or the full generation, if decoding already finished).

    :returns: True when the last ``<think>`` has no matching ``</think>`` after it.
    """
    return _in_unclosed_think(text)


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
    #
    # The anchor is the LAST marker in the text so far. This function sees a growing prefix, so
    # "last" means "most recent": a model that names the marker word along the way ("Counting each
    # user: ...", "the most common label: ...") re-anchors on each one, and the parser
    # (ctc.tasks.oolong.spec.parse) likewise reads what follows the last marker, so the two agree on
    # every prefix. They agree on a FINISHED string only if it is cut where the loop would have cut
    # it -- which is what apply() guarantees by replaying prefixes instead of calling this once on
    # the whole text, where "last" would mean "whatever the model echoed after answering".
    search_from = 0
    if cond.require_before:
        hits = []
        for marker in cond.require_before:
            at = text.lower().rfind(marker.lower())
            if at != -1:
                hits.append((at, marker))
        if not hits:
            return None
        marker_at, marker = max(hits)
        search_from = marker_at + len(marker)
        # A newline immediately after the marker is formatting ("Answer:\n1"), not the end of the
        # answer -- skip it, or the empty span between the marker and that newline satisfies a
        # "\n" stop and the answer is truncated to nothing.
        if search_from < len(text) and text[search_from] == "\n":
            search_from += 1

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
            through = text[: at + len(stop)]
            if cond.bracketed_answer and (_in_open_fence(through) or _in_open_bracket(through)):
                # Same idea as the <think> rule, judged on the prefix so the incremental and
                # whole-string paths agree: a newline in the middle of a pretty-printed literal,
                # or on the ```json line that opens it, is formatting and not the end of the
                # answer. The span INCLUDES the stop itself, because "]]" is precisely the token
                # that closes the literal it ends. Skip it and keep looking.
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
    at = _first_firing_stop(text, cond)
    return text if at is None else text[:at]


def _first_firing_stop(text: str, cond: StopCondition) -> Optional[int]:
    """
    Replay the decode loop over a finished string: the first prefix at which :func:`should_stop`
    fires decides where the text ends.

    Calling :func:`should_stop` once on the whole string is not the same thing. ``require_before``
    anchors on the last marker, which is right for a growing prefix -- the newest marker is the one
    the model just emitted -- but wrong for a finished string in which the model kept going after
    its answer. Every oolong corpus line carries ``User:``, so a no-cot checkpoint that answers and
    then echoes a corpus line moves the anchor onto the echo, and the echo is what gets graded. The
    decode loop never sees that echo: it stopped at the newline after the answer. Neither may this.

    Checking every prefix is quadratic, so only the prefixes at which the verdict can flip from
    "keep going" to "stop" are checked. Growing the prefix by one character changes what
    :func:`should_stop` sees in three ways: a stop substring is completed (a stop can begin to
    fire); a ``</think>`` is completed (suppression lifts from every stop before it -- the one
    condition here that is not local to the prefix up to a stop); or a marker, a ``<think>``, a
    bracket or a fence is completed, which can only prevent a stop from firing. The first two are
    therefore the only prefix ends worth checking, in order, and overlapping occurrences count:
    ``]]]`` completes a ``]]`` at two positions. The randomized test against a character-by-
    character replay is what holds this argument to account.

    :param text: The finished generation, already think-stripped if the condition asks for it.
    :param cond: The task's stop condition.

    :returns: Where to truncate, or ``None`` if no stop ever fires.
    """
    events = tuple(cond.text_stops) + (THINK_CLOSE,)
    ends = sorted(
        {
            m.start() + len(event)
            for event in events
            for m in re.finditer(f"(?={re.escape(event)})", text)
        }
    )
    for end in ends:
        at = should_stop(text[:end], cond)
        if at is not None:
            return at
    return None
