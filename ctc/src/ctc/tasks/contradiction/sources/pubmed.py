"""
``contradiction`` from PubMed -- the suite's flagship O(N^2) ladder.

N claim sentences, K of them arranged into contradicting pairs. One sentence of each pair is real
PubMed text and the other is a minimally-perturbed version of it that cannot also be true; every
other document is real PubMed text. Finding the pairs is quadratic in N by construction, which is
the property the whole scaling study is built on.

The mining lives in :mod:`ctc.data.sources.pubmed` (it needs a model); this module is the
assembly, and the assembly is where the false-negative control lives.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from ....data.generators.base import Generator
from ....data.sources import pubmed as source
from ..generate import assemble, place_pairs

__all__ = ["build_example", "GENERATOR"]


def build_example(
    rng: random.Random,
    *,
    index: int,
    corpus: source.PubMedPool,
    num_docs: int,
    num_pairs: int = 3,
) -> Optional[Dict]:
    """
    Build one contradiction example.

    :param rng: Seeded RNG.
    :param index: Example counter. Gold pairs are consumed in pool order rather than sampled, so no
        pair is silently reused inside a split while another goes unused.
    :param corpus: The claim pool.
    :param num_docs: Corpus size N -- the scaling axis.
    :param num_pairs: Gold pairs K.

    :returns: A unified-format example with 1-based gold pairs, or ``None`` when the pool is
        exhausted or cannot supply enough fillers.

    :raises ValueError: If ``num_docs`` cannot hold ``2 * num_pairs`` gold claims.
    """
    if num_docs < 2 * num_pairs:
        raise ValueError(f"num_docs={num_docs} cannot hold {num_pairs} gold pairs")
    start = index * num_pairs
    if start + num_pairs > len(corpus.pairs):
        return None
    pairs = corpus.pairs[start : start + num_pairs]

    statements, positions = place_pairs([(p.claim, p.contradiction) for p in pairs])
    # THE false-negative control: fillers come only from abstracts that contributed no gold to
    # THIS example. PubMed's scale means two unrelated abstracts almost never state the same fact,
    # but the abstract a gold claim came from very well might restate the fact its contradiction
    # denies -- which would be an unlabelled second contradicting pair, scored as a model error.
    fillers = source.sample_fillers(
        corpus.fillers,
        exclude_abstracts=[p.abstract_id for p in pairs],
        exclude_texts=statements,
        count=num_docs - len(statements),
        rng=rng,
    )
    if len(statements) + len(fillers) < num_docs:
        return None

    return assemble(
        statements + fillers,
        positions,
        source="pubmed_perturbation",
        rng=rng,
        meta={
            "num_pairs": num_pairs,
            "modes": sorted({p.mode for p in pairs}),
            "provenance": corpus.provenance,
        },
    )


GENERATOR = Generator(
    name="contradiction",
    task="contradiction",
    source="pubmed_perturbation",
    build_example=build_example,
    defaults={"num_docs": 100, "num_pairs": 3},
    corpus=source.load_pool,
    corpus_defaults={
        "pairs_path": None,
        "num_abstracts": 20_000,
        "num_mined_pairs": 30_000,
        "seed": 42,
        "mode": "realistic",
        "model": "gemini-2.5-flash",
        "base_url": None,
        "max_concurrent": 25,
        "max_overlap": 0.5,
        "validity_filter": True,
        "cache_dir": "data/.cache/ctc_llm",
    },
    indexed=True,
    notes="PubMed claims; gold pairs need a model to mine once, then reuse via pairs_path",
)
