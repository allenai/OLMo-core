"""
The ``contradiction`` data contract: source corpus -> task JSONL.

Separate from ``spec.py`` because the two change for different reasons. Editing the spec reprices
every result that already exists; editing a generator only affects data built afterwards. When both
lived in one file it was easy to touch one while meaning to touch the other, and the format
fingerprint could tell you *that* something moved but not which.

Contradiction has three source corpora, each with its own way of finding claim pairs that genuinely
conflict, so each gets a module under ``sources/``. They share this file's assembly and filtering;
only the claim mining differs.

.. warning::
   **Never harvest fillers with a domain-agnostic glob.** The pre-migration builder globbed a
   mutable directory with ``contradiction_*_k3.jsonl``, which also matched the FEVER and wiki
   corpora, so PubMed contradiction evals shipped Wikipedia claims as distractors -- 92.2% of
   fillers at 2k rising to 99.6% at 32k, against gold that was 100% PubMed. "Find the contradicting
   pair among n docs" then collapses to "find the biomedical sentences". Pin a manifest instead.

   The leak is **closed** for both the xlong and CTC-suite ladders (rebuilt and re-verified at
   0.00% on 2026-08-04/05), so this is a constraint on new generators rather than an outstanding
   repair. Two results of that rebuild are worth carrying forward, because each overturned a
   headline:

   * The contamination **depressed** the scores rather than flattering them -- it was a train/eval
     domain shift, not a shortcut -- and worst at the long end: 32k went 0.335 -> 0.559 on the
     clean ladder, roughly 10 SE at eval_size 500.
   * The published "dense collapses at 32k" was therefore an artifact of the ladder. The real dense
     curve declines gracefully, and the dense-vs-chunked absolute gap **narrows** with context
     (0.441 at 2k -> 0.369 at 32k) rather than widening.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator

__all__ = ["SOURCES", "build"]

#: source name -> claim miner. Populated as each source module lands.
SOURCES: Dict[str, Callable[..., Iterator[Dict]]] = {}


def build(source: str, *, rung: str, seed: int = 0, **opts) -> Iterator[Dict]:
    """
    Generate contradiction examples from one source corpus.

    :param source: One of :data:`SOURCES`.
    :param rung: Rung label; selects the claim count from ``SPEC.extra["claims_per_rung"]``.
    :param seed: RNG seed. Recorded in the output so a build is reproducible.
    :param opts: Source-specific options.

    :returns: Unified-format examples.

    :raises KeyError: If ``source`` is unknown.
    :raises NotImplementedError: While no source modules are ported.
    """
    if not SOURCES:
        raise NotImplementedError(
            "no contradiction sources ported yet. The pre-migration generators are "
            "generate_pubmed_contradiction_data.py (747 lines), "
            "generate_fever_contradiction_data.py (469) and "
            "generate_wiki_contradiction_data.py (546); each becomes a module under sources/."
        )
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}; have {', '.join(sorted(SOURCES))}")
    return SOURCES[source](rung=rung, seed=seed, **opts)
