"""
``strmatch`` data: N short strings, find every pair matching *this example's* similarity criterion.

The string analogue of :mod:`ctc.tasks.mathmatch`, and the one member of the pair family whose
criterion is **per example** rather than fixed by the instruction: the threshold is written into
``queries[0]`` and the prompt is the instruction followed by that sentence
(:func:`ctc.tasks.strmatch.spec.build_query`). :mod:`ctc.tasks.redundancy` is the fixed-criterion
member built on the same machinery, and the two must not converge -- a generator that stopped
varying the criterion would produce data that still validates, still scores, and quietly stops
testing the thing strmatch exists to test.

Two relations, both ported:

* ``substring`` (default) -- a pair matches when the two strings share a run of ``span_len``
  **consecutive** words, in the same order;
* ``wordset`` -- a pair matches when they share ``share_words`` words in any order or position.

Every word not placed in a planted block is unique within its example, so the planted gold pairs are
the only ones meeting the criterion, by construction rather than by a check.

.. warning::
   **The pre-migration construction is solvable by counting shared words -- measured, not
   theorised.** Gold pairs shared exactly ``span_len`` words and the hard negatives exactly
   ``span_len - 1``, with every other pair sharing none, so ranking pairs by bag-of-words overlap
   put gold on top *every time*: on the shipped ``eval_rungs/strmatch/rung_2048.jsonl``, the
   maximum-overlap pair was gold in **200 of 200** examples against a 0.004 chance baseline. The
   contiguity half of the criterion -- the entire difference between ``substring`` and ``wordset``
   -- was decorative. This is the same shortcut class as ``cycle``'s rarest-names and ``groups4``'s
   closest-pair (``records/data-generator-port.md`` trap 13): the data passes every structural
   check while a heuristic that never reads a word's position solves it.

   The old hard negatives do not close it. They are a near miss on *run length* (``span_len - 1``
   shared words), which forces the model to count the run exactly but leaves the count itself
   sufficient. What closes it is a near miss on *contiguity*: ``num_scattered`` decoy pairs that
   share **at least** ``span_len`` words with no two of them adjacent in either string, so their
   bag-of-words overlap meets or beats gold's while their longest shared run is one word. Both
   kinds of hard negative are built, because they defend different axes.

   ``num_scattered=0`` reproduces the pre-migration construction exactly, for a byte-comparison
   against a shipped file. It should not be used to build anything.

   :func:`ctc.data.audit.overlap_pair_is_gold` runs on every build and is what notices if this is
   ever undone.

.. warning::
   **The generic shrink cannot make this task's shorter rungs, and the failure is silent.**
   :func:`ctc.data.build.shrink` protects gold and drops random *distractors* -- but a scattered
   decoy is a pair, and dropping either half of it destroys it. Shrinking a 1216-document example
   to 72 keeps every gold pair intact and leaves essentially no decoy pair whole, so the shortcut
   this module exists to close comes back at every rung below the longest. Measured on the first
   build attempt: ``overlap_pair_is_gold`` 0.935 on the shrunk 2k rung against 0.000 on data built
   at n=72 directly. :func:`nested_ladder` is why this generator declares ``shrink_safe=False``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...data.generators.base import Generator
from ...data.gold import remap_groups, shuffle_with_remap
from ...data.schema import make_document, make_example
from ...data.wordlist import WORDS

__all__ = ["build_example", "nested_ladder", "criterion_text", "scattered_pairs_for", "GENERATOR"]

#: One scattered decoy pair per this many documents, once past the ``2 * num_pairs`` floor. Chosen
#: so the decoys grow with the corpus: a fixed count would make the bag-of-words shortcut *stronger*
#: at long context, which is the exact shape of the cycle failure (trap 13).
SCATTERED_PER_DOCS = 25


def criterion_text(relation: str, threshold: int) -> str:
    """
    The example's own criterion sentence, stored in ``queries[0]``.

    Verbatim from the pre-migration generator: it is the second half of every strmatch prompt the
    shipped checkpoints were trained and evaluated on, so its wording is data, not prose.

    :param relation: ``"substring"`` or ``"wordset"``.
    :param threshold: Run length (substring) or shared-word count (wordset).

    :returns: The criterion sentence.

    :raises ValueError: On an unknown relation.
    """
    if relation == "substring":
        return (
            f"Find all pairs of strings that contain a run of at least {threshold} consecutive "
            f"words in common (the same {threshold} words, in the same order, appearing "
            f"contiguously in both strings)."
        )
    if relation == "wordset":
        return (
            f"Find all pairs of strings that share at least {threshold} words in common (the same "
            f"word appearing in both strings, regardless of position)."
        )
    raise ValueError(f"relation must be 'substring' or 'wordset', got {relation!r}")


def scattered_pairs_for(num_docs: int, num_pairs: int) -> int:
    """
    :param num_docs: Corpus size N.
    :param num_pairs: Gold pairs K.

    :returns: How many contiguity decoys a corpus of this size gets by default -- at least twice as
        many as gold, so the bag-of-words heuristic cannot win on a tie-break, and one more per
        :data:`SCATTERED_PER_DOCS` documents beyond that.
    """
    return max(2 * num_pairs, num_docs // SCATTERED_PER_DOCS)


def _spread_positions(n_slots: int, count: int, rng: random.Random) -> List[int]:
    """
    Choose ``count`` positions out of ``n_slots`` with no two adjacent.

    :param n_slots: String length L.
    :param count: How many positions are needed.
    :param rng: Seeded RNG.

    :returns: The positions, ascending, each at least 2 apart -- so the words placed at them cannot
        form a shared run of length 2 or more, whatever the other string looks like.

    :raises ValueError: If ``n_slots`` is too short to separate ``count`` positions.
    """
    if n_slots < 2 * count - 1:
        raise ValueError(
            f"cannot place {count} mutually non-adjacent words in a string of {n_slots}: "
            f"str_len must be at least {2 * count - 1}"
        )
    chosen = sorted(rng.sample(range(n_slots - count + 1), count))
    return [value + offset for offset, value in enumerate(chosen)]


class _Corpus:
    """
    The planted structure of one example, kept as separate lists so a prefix of each is a valid
    smaller corpus.

    That is the whole reason this is a class rather than a flat list: the eval ladder needs shorter
    rungs whose documents are a *subset* of the longer ones **and** whose decoy pairs are intact,
    and taking a prefix of each list gives both at once.

    :param gold: Gold pairs, as ``(string, string)``.
    :param hardneg: Near-miss pairs at ``threshold - 1``.
    :param scattered: Contiguity decoys.
    :param distractors: Strings in no pair at all.
    """

    def __init__(
        self,
        gold: List[Tuple[List[str], List[str]]],
        hardneg: List[Tuple[List[str], List[str]]],
        scattered: List[Tuple[List[str], List[str]]],
        distractors: List[List[str]],
    ) -> None:
        self.gold = gold
        self.hardneg = hardneg
        self.scattered = scattered
        self.distractors = distractors

    def select(self, n_hardneg: int, n_scattered: int, n_distract: int):
        """
        :param n_hardneg: Hard-negative pairs to keep.
        :param n_scattered: Scattered decoy pairs to keep.
        :param n_distract: Distractors to keep.

        :returns: ``(strings, gold_positions)`` in construction order, gold first.
        """
        strings: List[List[str]] = []
        positions: List[Tuple[int, int]] = []
        for first, second in self.gold:
            positions.append((len(strings), len(strings) + 1))
            strings.extend([first, second])
        for first, second in self.hardneg[:n_hardneg] + self.scattered[:n_scattered]:
            strings.extend([first, second])
        strings.extend(self.distractors[:n_distract])
        return strings, positions


def _resolve(
    num_docs: int,
    num_pairs: int,
    relation: str,
    span_len: int,
    share_words: int,
    str_len: int,
    num_hardneg: Optional[int],
    num_scattered: Optional[int],
) -> Tuple[int, int, int, int]:
    """
    Turn the knobs into the four counts the construction needs.

    :param num_docs: Corpus size N.
    :param num_pairs: Gold pairs K.
    :param relation: ``"substring"`` or ``"wordset"``.
    :param span_len: Run length, for ``substring``.
    :param share_words: Shared-word threshold, for ``wordset``.
    :param str_len: Words per string L.
    :param num_hardneg: Hard negatives, or ``None`` for K.
    :param num_scattered: Scattered decoys, or ``None`` for the default.

    :returns: ``(threshold, n_hardneg, n_scattered, n_distract)``.

    :raises ValueError: On an unknown relation, a threshold that does not fit ``str_len``,
        ``num_scattered`` under ``wordset``, or a corpus too small to hold the planted pairs.

    .. note::
       A *defaulted* decoy count is capped at what the corpus can hold, so a deliberately tiny
       ``num_docs`` degrades to fewer decoys instead of raising; the effective counts are written
       into the example's ``_meta``, so a corpus that lost its defence says so. An *explicit* count
       is honoured or refused -- a build script that asks for 20 scattered pairs in 30 documents
       has made an error, and silently building something else is how a defence disappears.
    """
    threshold = span_len if relation == "substring" else share_words
    criterion_text(relation, threshold)  # validates `relation`
    if not 1 <= threshold <= str_len:
        raise ValueError(f"threshold {threshold} must be in [1, str_len={str_len}]")

    #: Decoy PAIRS the corpus has room for once the gold pairs are placed.
    capacity = num_docs // 2 - num_pairs
    n_hardneg = max(0, min(num_pairs, capacity)) if num_hardneg is None else num_hardneg
    if relation == "wordset":
        if num_scattered:
            raise ValueError(
                "num_scattered is a contiguity decoy: under relation='wordset' a pair sharing the "
                "threshold word count IS a match, so scattered decoys would be unlabelled gold. "
                "Set num_scattered=0 for wordset."
            )
        n_scattered = 0
    elif num_scattered is None:
        n_scattered = max(0, min(scattered_pairs_for(num_docs, num_pairs), capacity - n_hardneg))
    else:
        n_scattered = num_scattered

    if n_scattered and threshold > (str_len + 1) // 2:
        raise ValueError(
            f"str_len={str_len} cannot hold {threshold} mutually non-adjacent words; scattered "
            f"decoys need str_len >= {2 * threshold - 1}, or pass num_scattered=0"
        )
    n_distract = num_docs - 2 * (num_pairs + n_hardneg + n_scattered)
    if n_distract < 0:
        raise ValueError(
            f"num_docs={num_docs} cannot hold {num_pairs} gold + {n_hardneg} hard-negative + "
            f"{n_scattered} scattered pairs ({2 * (num_pairs + n_hardneg + n_scattered)} strings)"
        )
    return threshold, n_hardneg, n_scattered, n_distract


def _plant(
    rng: random.Random,
    *,
    threshold: int,
    relation: str,
    str_len: int,
    num_pairs: int,
    n_hardneg: int,
    n_scattered: int,
    n_distract: int,
) -> _Corpus:
    """
    Draw one example's strings: K matching pairs, the two kinds of decoy, and the distractors.

    One draw of distinct words for the whole example, as pre-migration: every word outside a planted
    block occurs exactly once, which is what makes the planted pairs the *only* matches without a
    verification pass over all N^2 of them.

    :param rng: Seeded RNG.
    :param threshold: Match threshold.
    :param relation: ``"substring"`` or ``"wordset"``.
    :param str_len: Words per string L.
    :param num_pairs: Gold pairs K.
    :param n_hardneg: Hard-negative pairs.
    :param n_scattered: Scattered decoy pairs.
    :param n_distract: Distractors.

    :returns: The planted corpus.

    :raises ValueError: If the example would need more distinct words than the frozen vocabulary
        holds.
    """
    scatter_cap = (str_len + 1) // 2
    # Scattered decoys draw at or just above the gold threshold. Drawing *only* above it would hand
    # the shortcut back in mirror image ("the pairs sharing exactly k words are the gold ones"), so
    # half of them sit exactly at k and are separated from gold by contiguity alone.
    scatter_sizes = [
        min(threshold + rng.choice((0, 0, 1, 2)), scatter_cap) for _ in range(n_scattered)
    ]

    def pair_cost(size: int) -> int:
        return size + 2 * (str_len - size)

    need = (
        num_pairs * pair_cost(threshold)
        + n_hardneg * pair_cost(threshold - 1)
        + sum(pair_cost(size) for size in scatter_sizes)
        + n_distract * str_len
    )
    if need > len(WORDS):
        raise ValueError(
            f"this example needs {need} distinct words but the frozen vocabulary holds "
            f"{len(WORDS)}; lower num_docs or str_len"
        )
    words = rng.sample(WORDS, need)
    cursor = 0

    def take(count: int) -> List[str]:
        nonlocal cursor
        chunk = words[cursor : cursor + count]
        cursor += count
        return chunk

    def contiguous_pair(size: int) -> Tuple[List[str], List[str]]:
        """Two strings sharing one run of ``size`` consecutive words, at independent offsets."""
        block = take(size)
        offset_a = rng.randint(0, str_len - size)
        offset_b = rng.randint(0, str_len - size)
        first = take(offset_a) + block + take(str_len - size - offset_a)
        second = take(offset_b) + block + take(str_len - size - offset_b)
        return first, second

    def wordset_pair(size: int) -> Tuple[List[str], List[str]]:
        """Two strings sharing ``size`` words as a set; positions irrelevant, so both shuffle."""
        block = take(size)
        first = block + take(str_len - size)
        second = block + take(str_len - size)
        rng.shuffle(first)
        rng.shuffle(second)
        return first, second

    def scattered_pair(size: int) -> Tuple[List[str], List[str]]:
        """Two strings sharing ``size`` words with no two of them adjacent in either string."""
        block = take(size)
        filler = take(2 * (str_len - size))
        out: List[List[str]] = []
        for _ in range(2):
            slots: List[Optional[str]] = [None] * str_len
            order = list(block)
            # A different order in each string as well as different positions: matching order is
            # half of what "contiguous run" means, and leaving it fixed would make the pair a near
            # miss on one axis only.
            rng.shuffle(order)
            for position, word in zip(_spread_positions(str_len, len(order), rng), order):
                slots[position] = word
            out.append([word if word is not None else filler.pop() for word in slots])
        return out[0], out[1]

    match = contiguous_pair if relation == "substring" else wordset_pair
    return _Corpus(
        gold=[match(threshold) for _ in range(num_pairs)],
        hardneg=[match(threshold - 1) for _ in range(n_hardneg)],
        scattered=[scattered_pair(size) for size in scatter_sizes],
        distractors=[take(str_len) for _ in range(n_distract)],
    )


def _emit(
    strings: Sequence[List[str]],
    positions: Sequence[Tuple[int, int]],
    *,
    rng: random.Random,
    query: str,
    meta: Mapping[str, object],
) -> Dict:
    """
    Shuffle the strings and record where the gold pairs landed.

    :param strings: Strings in construction order, gold first.
    :param positions: Construction-order gold pair positions.
    :param rng: Seeded RNG; consumes exactly one shuffle.
    :param query: The example's criterion sentence.
    :param meta: Per-example metadata.

    :returns: The unified-format example, gold **1-based**.
    """
    items, old_to_new = shuffle_with_remap(list(strings), rng=rng, base=1)
    return make_example(
        documents=[make_document(" ".join(item)) for item in items],
        queries=[query],
        source="strmatch",
        gold=remap_groups([list(pair) for pair in positions], old_to_new),
        meta=dict(meta),
    )


def _meta_for(
    relation: str, threshold: int, str_len: int, n_hardneg: int, n_scattered: int
) -> Dict:
    meta: Dict[str, object] = {
        "relation": relation,
        "str_len": str_len,
        "num_hardneg": n_hardneg,
        "num_scattered": n_scattered,
    }
    meta["span_len" if relation == "substring" else "share_words"] = threshold
    return meta


def build_example(
    rng: random.Random,
    *,
    num_docs: int,
    num_pairs: int = 3,
    relation: str = "substring",
    span_len: int = 3,
    share_words: int = 3,
    str_len: int = 10,
    num_hardneg: Optional[int] = None,
    num_scattered: Optional[int] = None,
) -> Dict:
    """
    Build one strmatch example.

    :param rng: Seeded RNG.
    :param num_docs: Corpus size N -- the scaling axis.
    :param num_pairs: Gold pairs K.
    :param relation: ``"substring"`` (shared contiguous run) or ``"wordset"`` (shared word set).
    :param span_len: Run length k, used when ``relation="substring"``.
    :param share_words: Shared-word threshold W, used when ``relation="wordset"``. Kept as a second
        parameter rather than folded into one ``threshold`` because the pre-migration CLI had two,
        and a build script that sets the wrong one must fail loudly rather than build at a default.
    :param str_len: Words per string L.
    :param num_hardneg: Near-miss pairs at ``threshold - 1``, per example. Defaults to K, as it did
        pre-migration. These defend the *counting* axis.
    :param num_scattered: Contiguity decoy pairs; see :func:`scattered_pairs_for` for the default
        and the module warning for why they exist. Meaningless under ``wordset``, where sharing the
        words IS the criterion, so a non-zero value there is an error rather than a no-op.

    :returns: A unified-format example whose ``gold_doc_indices`` holds 1-based pairs, and whose
        ``queries[0]`` states this example's criterion.

    :raises ValueError: If the relation is unknown, the threshold does not fit in ``str_len``,
        ``num_scattered`` is set under ``wordset``, or ``num_docs`` cannot hold the planted pairs.
    """
    threshold, n_hardneg, n_scattered, n_distract = _resolve(
        num_docs, num_pairs, relation, span_len, share_words, str_len, num_hardneg, num_scattered
    )
    corpus = _plant(
        rng,
        threshold=threshold,
        relation=relation,
        str_len=str_len,
        num_pairs=num_pairs,
        n_hardneg=n_hardneg,
        n_scattered=n_scattered,
        n_distract=n_distract,
    )
    strings, positions = corpus.select(n_hardneg, n_scattered, n_distract)
    return _emit(
        strings,
        positions,
        rng=rng,
        query=criterion_text(relation, threshold),
        meta=_meta_for(relation, threshold, str_len, n_hardneg, n_scattered),
    )


def nested_ladder(
    rng: random.Random,
    *,
    index: int,
    rungs: Mapping[str, int],
    num_docs: int = 0,
    num_pairs: int = 3,
    relation: str = "substring",
    span_len: int = 3,
    share_words: int = 3,
    str_len: int = 10,
    num_hardneg: Optional[int] = None,
    num_scattered: Optional[int] = None,
) -> Optional[Dict[str, Dict]]:
    """
    Build every rung of one row at once, keeping the decoy pairs whole.

    The generic nested ladder shrinks the longest rung by dropping random non-gold documents, which
    for this task quietly deletes the defence: a scattered decoy is a *pair*, and half of one is a
    distractor. Here the corpus is planted once at the longest rung as separate lists, and each rung
    takes a prefix of every list -- so a shorter rung's documents are a subset of the longer one's
    (the ladder is genuinely nested), every rung's gold is the same K pairs (it grades the same
    question), and every rung carries the decoy density a standalone build at that size would have.

    :param rng: Seeded RNG for the planting.
    :param index: Example counter; used to key the per-rung shuffles.
    :param rungs: rung label -> document count.
    :param num_docs: Ignored -- the ladder sets the size per rung. Accepted because the builder
        passes the whole resolved config through.
    :param num_pairs: Gold pairs K.
    :param relation: ``"substring"`` or ``"wordset"``.
    :param span_len: Run length, for ``substring``.
    :param share_words: Shared-word threshold, for ``wordset``.
    :param str_len: Words per string L.
    :param num_hardneg: Hard negatives per example, constant across rungs.
    :param num_scattered: Scattered decoys; ``None`` scales them per rung.

    :returns: rung label -> example.

    :raises ValueError: As :func:`build_example`, for the longest rung.
    """
    counts = {
        label: _resolve(
            size, num_pairs, relation, span_len, share_words, str_len, num_hardneg, num_scattered
        )
        for label, size in rungs.items()
    }
    longest = max(rungs, key=lambda label: rungs[label])
    threshold, max_hardneg, max_scattered, max_distract = counts[longest]
    corpus = _plant(
        rng,
        threshold=threshold,
        relation=relation,
        str_len=str_len,
        num_pairs=num_pairs,
        n_hardneg=max_hardneg,
        n_scattered=max_scattered,
        n_distract=max_distract,
    )

    out: Dict[str, Dict] = {}
    for label in rungs:
        _, n_hardneg, n_scattered, n_distract = counts[label]
        strings, positions = corpus.select(n_hardneg, n_scattered, n_distract)
        # One stream per rung, so adding a rung never perturbs another rung's ordering.
        out[label] = _emit(
            strings,
            positions,
            rng=random.Random(f"{index}:strmatch:{label}"),
            query=criterion_text(relation, threshold),
            meta=_meta_for(relation, threshold, str_len, n_hardneg, n_scattered),
        )
    return out


GENERATOR = Generator(
    name="strmatch",
    task="strmatch",
    source="strmatch",
    build_example=build_example,
    defaults={
        "num_docs": 72,
        "num_pairs": 3,
        "relation": "substring",
        "span_len": 3,
        "share_words": 3,
        "str_len": 10,
        "num_hardneg": None,
        "num_scattered": None,
    },
    shrink_safe=False,  # see nested_ladder: a random shrink destroys the contiguity decoys
    build_ladder=nested_ladder,
    notes="pure synthetic; per-example criterion, from a frozen wordlist (no corpus, no network)",
)
