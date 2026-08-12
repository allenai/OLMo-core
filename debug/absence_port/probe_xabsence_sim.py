"""
How solvable is xabsence by word overlap, and what closes it?

Simulates the ported assembly against the real staged pool and runs the same
``unmatched_by_lexical_overlap`` probe the build audit runs, over three axes:

* **decoys** -- co-placing each orphan's nearest cross-form neighbour as a matched pair;
* **pool size** -- the BUILD_MATRIX blocker B3 axis;
* **max_overlap** -- how tight the mining filter that produced the pool was.

Throwaway measurement kept for the numbers quoted in ``ctc/src/ctc/data/sources/paraphrase.py``.
Run on the login node (no GPU, no network)::

    python debug/absence_port/probe_xabsence_sim.py
"""

from __future__ import annotations

import json
import random
import re
import statistics
from typing import Dict, List, Sequence, Set, Tuple

POOL = "/scratch/users/prasann/corpus-reasoning/data/xabsence_pool_pubmed.jsonl"
WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> Set[str]:
    """:param text: Any claim. :returns: Its lowercased alphanumeric token set."""
    return set(WORD.findall(text.lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """:param a: One token set. :param b: The other. :returns: Their Jaccard overlap."""
    return len(a & b) / max(1, len(a | b))


def load(max_overlap: float, size: int) -> Tuple[List[Set[str]], List[Set[str]]]:
    """
    :param max_overlap: Drop twins whose word-Jaccard exceeds this, as the loader now does on read.
    :param size: Entries to keep, sampled deterministically.

    :returns: ``(originals, paraphrases)`` as parallel token sets.
    """
    rows = [json.loads(line) for line in open(POOL, encoding="utf-8") if line.strip()]
    kept = [r for r in rows if jaccard(tokens(r["claim"]), tokens(r["paraphrase"])) <= max_overlap]
    idx = random.Random(1).sample(range(len(kept)), min(size, len(kept)))
    return [tokens(kept[i]["claim"]) for i in idx], [tokens(kept[i]["paraphrase"]) for i in idx]


def neighbours(
    sources: Sequence[Set[str]], destinations: Sequence[Set[str]], top: int
) -> List[List[int]]:
    """
    :param sources: Token sets to find neighbours for.
    :param destinations: Token sets to search, parallel to ``sources``.
    :param top: Neighbours per source.

    :returns: Destination indices, best first, never the source's own twin.
    """
    n = len(sources)
    return [
        [
            j
            for _, j in sorted(
                ((jaccard(sources[i], destinations[j]), j) for j in range(n) if j != i),
                reverse=True,
            )[:top]
        ]
        for i in range(n)
    ]


def probe(max_overlap: float, size: int, pairs: int, top: int, k: int = 3, trials: int = 200):
    """
    :param max_overlap: Mining filter re-applied on read.
    :param size: Pool entries available.
    :param pairs: Matched pairs per example, ``P``.
    :param top: Decoys co-placed per orphan; ``0`` reproduces the pre-migration construction.
    :param k: Orphans per example.
    :param trials: Examples simulated.

    :returns: The mean share of orphans landing in the ``k`` lowest-overlap positions.
    """
    originals, paraphrases = load(max_overlap, size)
    n = len(originals)
    if n < pairs + k:
        return None
    near: Dict[str, List[List[int]]] = {
        "A": neighbours(originals, paraphrases, top) if top else [[] for _ in range(n)],
        "B": neighbours(paraphrases, originals, top) if top else [[] for _ in range(n)],
    }
    rng = random.Random(0)
    hits = []
    for _ in range(trials):
        orphans = rng.sample(range(n), k)
        sides = [rng.choice("AB") for _ in orphans]
        forced = [j for i, s in zip(orphans, sides) for j in near[s][i]]
        forced = [j for j in dict.fromkeys(forced) if j not in orphans][:pairs]
        rest = [j for j in range(n) if j not in orphans and j not in forced]
        matched = forced + rng.sample(rest, pairs - len(forced))
        block_a = [("A", originals[j]) for j in matched]
        block_a += [("A", originals[i]) for i, s in zip(orphans, sides) if s == "A"]
        block_b = [("B", paraphrases[j]) for j in matched]
        block_b += [("B", paraphrases[i]) for i, s in zip(orphans, sides) if s == "B"]
        docs = block_a + block_b
        on_a = sum(1 for s in sides if s == "A")
        gold = set(range(len(block_a) - on_a, len(block_a)))
        gold |= set(range(len(docs) - (k - on_a), len(docs)))
        best = [max((jaccard(t, u) for s, u in docs if s != side), default=0.0) for side, t in docs]
        lowest = set(sorted(range(len(docs)), key=lambda i: best[i])[:k])
        hits.append(len(gold & lowest) / k)
    return statistics.mean(hits)


def main() -> None:
    """Print the three sweeps."""
    print("decoys (max_overlap 0.30, whole 659-entry pool):")
    for pairs in (28, 58, 120):
        row = [f"P={pairs:>3} n={2 * pairs + 3:>3} chance {3 / (2 * pairs + 3):.3f}"]
        for top in (0, 1, 2, 3, 4, 8):
            row.append(f"top{top}={probe(0.30, 659, pairs, top):.3f}")
        print("  " + "  ".join(row))

    print("\npool size (max_overlap 0.30, top-2 decoys):")
    for pairs in (28, 58):
        row = [f"P={pairs:>3}"]
        for size in (120, 250, 450, 659):
            row.append(f"pool{size}={probe(0.30, size, pairs, 2):.3f}")
        print("  " + "  ".join(row))

    print("\nmining filter (whole pool, P=28, top-2 decoys):")
    for max_overlap in (0.30, 0.22, 0.18):
        bare = probe(max_overlap, 659, 28, 0)
        fixed = probe(max_overlap, 659, 28, 2)
        print(f"  max_overlap<={max_overlap}: no decoys {bare:.3f}  top-2 decoys {fixed:.3f}")


if __name__ == "__main__":
    main()
