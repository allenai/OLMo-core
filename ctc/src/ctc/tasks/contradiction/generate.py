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
   **The FEVER filler leak is still open for the CTC-suite ladder.** FEVER filler claims were found
   in PubMed contradiction evals. It is fixed for the xlong ladders and not for this one, so a
   PubMed contradiction number off the CTC ladder may be contaminated. Resolve before rebuilding.
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
