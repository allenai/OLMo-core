"""
Tiny synthetic corpora, one per pool type, so the corpus-backed generators can be tested offline.

The whole point of the ``ctc.data.sources`` seam is that a pool is a plain dataclass and only its
loader touches the network. These builders are what cashes that in: every construction property
that matters -- gold index base, filler provenance, hard-negative prefixes, the scale-K invariant,
ladder nesting -- is exercised here with no HF download, no Lucene index, no GPU and no model.

The corpora are deliberately small but **not degenerate**: enough abstracts that a filler can be
drawn from a non-gold one, enough articles that a run can exclude every title another run touched,
enough labels that an OOLONG aggregate has a unique argmax. A fixture that is too small stops
testing the property and starts testing the fallback path.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ctc.data.sources import amazon, fever, oolong, pubmed, wiki100w
from ctc.data.sources.retrieval import Candidate, QueryPool, RetrievalPool

__all__ = [
    "pubmed_pool",
    "fever_pool",
    "retrieval_pool",
    "rerank_pool",
    "article_pool",
    "review_pool",
    "oolong_pool",
]


def pubmed_pool(pairs: int = 40, abstracts: int = 60, per_abstract: int = 8) -> pubmed.PubMedPool:
    """
    :param pairs: Gold claim pairs.
    :param abstracts: Filler abstracts. Includes the gold abstracts, so the "fillers never come
        from a gold abstract" rule has something to exclude.
    :param per_abstract: Sentences per abstract.

    :returns: A PubMed pool.
    """
    claim_pairs = tuple(
        pubmed.ClaimPair(
            claim=f"Treatment {i} reduced mortality by {i} percent in the trial cohort.",
            contradiction=f"Mortality rose by {i + 40} percent under regimen {i} in a later study.",
            abstract_id=f"a{i}",
            mode="realistic",
        )
        for i in range(pairs)
    )
    fillers = {
        f"a{i}": tuple(
            f"Abstract {i} sentence {j} reports an unrelated measurement of outcome {i}-{j}."
            for j in range(per_abstract)
        )
        for i in range(abstracts)
    }
    return pubmed.PubMedPool(pairs=claim_pairs, fillers=fillers, provenance={"pairs": "fixture"})


def fever_pool(pairs: int = 40, pages: int = 30, fillers: int = 400) -> fever.FeverPool:
    """
    :param pairs: REFUTES pairs.
    :param pages: Wikipedia pages carrying NEI and SUPPORTS rows.
    :param fillers: Flat filler claims.

    :returns: A FEVER pool.
    """
    return fever.FeverPool(
        pairs=tuple(
            (f"Person {i} was born in July.", f"Person {i} was born on 16 May 1991.", f"page{i}")
            for i in range(pairs)
        ),
        nei_by_page={
            f"page{i}": tuple(f"Person {i} may have visited city {j}." for j in range(6))
            for i in range(pages)
        },
        support_pairs_by_page={
            f"page{i}": tuple(
                (f"Person {i} acted in film {j}.", f"Person {i} starred in film {j} in 200{j}.")
                for j in range(3)
            )
            for i in range(pages)
        },
        fillers=tuple(
            f"Filler claim number {i} about an unrelated subject." for i in range(fillers)
        ),
        pages=tuple(f"page{i}" for i in range(pages)),
    )


def retrieval_pool(
    queries: int = 40, hard: int = 30, corpus: int = 600, source: str = "nq"
) -> RetrievalPool:
    """
    :param queries: Prepared queries, each with one gold document.
    :param hard: Hard negatives per query, hardest first.
    :param corpus: Filler documents shared by every query.
    :param source: Corpus tag.

    :returns: A retrieval pool with no pre-drawn fill, so fillers come from ``corpus``.
    """

    def body(text: str, salt: int) -> str:
        """Pad to a length that does not correlate with gold/hard/filler role.

        Without this the gold passage is systematically the longest or shortest in its pool and
        the length shortcut probe fires on the fixture rather than on anything a generator did.
        """
        return text + " " + " ".join(["context"] * (5 + (salt * 37) % 40))

    prepared = tuple(
        QueryPool(
            query=f"who invented gadget {i}?",
            gold=(Candidate(f"g{i}", body(f"Gadget {i} was invented by inventor {i}.", i * 3)),),
            hard=tuple(
                Candidate(
                    f"h{i}-{j}",
                    body(f"Gadget {i} is often confused with gadget {j}.", i * 5 + j),
                )
                for j in range(hard)
            ),
            answers=(f"inventor {i}",),
        )
        for i in range(queries)
    )
    return RetrievalPool(
        queries=prepared,
        corpus=tuple(
            Candidate(f"c{i}", body(f"Unrelated passage {i} on another subject.", i * 7))
            for i in range(corpus)
        ),
        source=source,
    )


def rerank_pool(queries: int = 40, hard: int = 20, fill: int = 300) -> RetrievalPool:
    """
    :param queries: Prepared queries.
    :param hard: CE-verified hard negatives per query.
    :param fill: Pre-drawn, pre-scored random fill per query.

    :returns: A rerank pool: per-query pre-drawn fill and a score for **every** document, which is
        what the graded ordering needs.
    """
    prepared: List[QueryPool] = []
    for i in range(queries):
        gold = (Candidate(f"g{i}", f"Passage answering query {i} directly and completely."),)
        hard_docs = tuple(
            Candidate(f"h{i}-{j}", f"Passage {j} nearly answering query {i}.") for j in range(hard)
        )
        fill_docs = tuple(
            Candidate(f"f{i}-{j}", f"Passage {j} unrelated to query {i}.") for j in range(fill)
        )
        scores: Dict[str, float] = {gold[0].id: 10.0}
        scores.update({c.id: 5.0 - 0.1 * j for j, c in enumerate(hard_docs)})
        scores.update({c.id: -5.0 - 0.01 * j for j, c in enumerate(fill_docs)})
        prepared.append(
            QueryPool(
                query=f"query {i}",
                gold=gold,
                hard=hard_docs,
                fill=fill_docs,
                scores=scores,
            )
        )
    return RetrievalPool(queries=tuple(prepared), corpus=(), source="msmarco_trainhn")


def article_pool(articles: int = 120, chunks: int = 20) -> wiki100w.ArticlePool:
    """
    :param articles: Distinct Wikipedia articles.
    :param chunks: Passages per article.

    :returns: An article pool. Passage *length* is deliberately decorrelated from the article it
        belongs to: if every passage of one topic were the same length, the outlier would be
        findable by length alone and the shortcut probes would fire on the fixture rather than on
        anything the generator did.
    """
    return wiki100w.ArticlePool(
        [
            (
                f"Article {i}",
                [
                    f"Article {i} passage {j} describing topic {i}. "
                    + " ".join(["detail"] * (5 + (i * 31 + j * 17) % 40))
                    for j in range(chunks)
                ],
            )
            for i in range(articles)
        ],
        min_chunks=4,
    )


def review_pool(categories: int = 6, per_category: int = 400) -> amazon.ReviewPool:
    """
    :param categories: Product categories. At least two, or a category-outlier example is
        impossible by construction.
    :param per_category: Reviews per category, spread over all five star ratings.

    :returns: A review pool.
    """
    names = amazon.DEFAULT_CATEGORIES[:categories]
    return amazon.ReviewPool(
        {
            name: tuple(
                amazon.Review(
                    title=f"{name} review {i}",
                    text=f"This {name} item number {i} performed as described over several weeks.",
                    rating=(i % 5) + 1,
                    category=name,
                )
                for i in range(per_category)
            )
            for name in names
        }
    )


def oolong_pool(items: int = 400, labels: Tuple[str, ...] = ("spam", "ham", "other")):
    """
    :param items: Items in the single sub-dataset.
    :param labels: Canonical label order.

    :returns: An OOLONG pool with measured per-item token counts.
    """
    built = tuple(
        oolong.Item(
            line=f"Date: Jan {i % 28 + 1:02d}, 2020 || User: u{i % 7} || Instance: message {i}",
            user=f"u{i % 7}",
            label=labels[i % len(labels)],
            date=f"Jan {i % 28 + 1:02d}, 2020",
            month=(i % 12) + 1,
            tokens=12,
        )
        for i in range(items)
    )
    return oolong.OolongPool(
        items={"spamset": built},
        labels={"spamset": labels},
        preamble={"spamset": "Below are {N} labelled messages."},
        preamble_tokens={"spamset": 8},
    )
