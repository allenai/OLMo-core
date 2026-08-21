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

from ctc.data.sources import (
    amazon,
    gutenberg,
    oolong,
    openalex,
    paraphrase,
    pubmed,
    wiki100w,
)
from ctc.data.sources.retrieval import Candidate, QueryPool, RetrievalPool

__all__ = [
    "pubmed_pool",
    "retrieval_pool",
    "rerank_pool",
    "article_pool",
    "review_pool",
    "oolong_pool",
    "book_pool",
    "paraphrase_pool",
    "reorder_pool",
    "unit_pool",
    "openalex_pool",
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


def retrieval_pool(
    queries: int = 40, hard: int = 30, corpus: int = 600, source: str = "nq", gold: int = 1
) -> RetrievalPool:
    """
    :param queries: Prepared queries.
    :param hard: Hard negatives per query, hardest first.
    :param corpus: Filler documents shared by every query.
    :param source: Corpus tag.
    :param gold: Gold documents per query. More than one is not an edge case -- ``hotpotqa`` is
        always two and FiQA averages ~2.6 -- and it selects a different instruction wording and a
        different scoring path, so a single-gold-only fixture leaves both untested.

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
            # k=0 is byte-identical to the single-gold pool this fixture has always produced, so
            # raising `gold` adds hops without perturbing any existing ladder's fixture data.
            gold=tuple(
                Candidate(
                    f"g{i}" if k == 0 else f"g{i}-{k}",
                    body(
                        (
                            f"Gadget {i} was invented by inventor {i}."
                            if k == 0
                            else f"Gadget {i} hop {k} was first documented by archivist {i}."
                        ),
                        i * 3 + k * 11,
                    ),
                )
                for k in range(gold)
            ),
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


#: Ordinary words the fixture sentences are built from. Real enough that two sentences about the
#: same subject share vocabulary and two about different subjects do not, which is what the
#: xabsence decoy machinery has to have something to work on.
_SUBJECTS = (
    "cardiac", "hepatic", "renal", "pulmonary", "neural", "vascular",
    "skeletal", "immune", "thyroid", "gastric", "ocular", "dermal",
)  # fmt: skip
_OUTCOMES = ("mortality", "recovery", "inflammation", "perfusion", "density", "latency")


def book_pool(
    books: int = 12, runs_per_book: int = 3, sentences_per_run: int = 240
) -> gutenberg.BookPool:
    """
    :param books: Distinct books. ``BookPool.for_split`` cuts by book at a tenth, so this must be
        at least 10 for the eval split to hold more than one.
    :param runs_per_book: Prose runs per book.
    :param sentences_per_run: Sentences per run. Must clear the largest rung under test -- the
        absence ladder builds each rung independently from a window of that many CONSECUTIVE
        sentences, so a short run makes the long rungs silently unbuildable rather than smaller.

    :returns: A Gutenberg-style pool. Every sentence is unique and every four-word opening is
        unique, so the prefix-uniqueness filter never rejects a fixture window for a reason the
        test did not intend.
    """
    return gutenberg.BookPool(
        runs=tuple(
            gutenberg.ProseRun(
                book=f"book{b}",
                sentences=tuple(
                    f"Passage {b}-{r}-{s} recounts a quiet afternoon beside the "
                    f"{_SUBJECTS[(b + s) % len(_SUBJECTS)]} river."
                    for s in range(sentences_per_run)
                ),
            )
            for b in range(books)
            for r in range(runs_per_book)
        ),
        provenance={"corpus": "fixture", "books": books},
    )


def paraphrase_pool(entries: int = 300, clusters: int = 15) -> paraphrase.ParaphrasePool:
    """
    :param entries: Claim/paraphrase twins. Must exceed ``P + k`` for the rung under test, and by
        enough that two examples are not the same documents.
    :param clusters: Topic groups. Entries in one cluster draw from one small vocabulary, so the
        pool has genuine lexical near-neighbours -- roughly 20 of them per cluster at the defaults,
        which is what gives an orphan a plausible impostor to be confused with.

    :returns: A twin pool where a twin overlaps its original **more than, but not disjointly from**
        a topic neighbour. Both halves matter: with no near-neighbours the orphan is the only
        document without a lexical counterpart and the probe scores 1.0 whatever the generator
        does; with a *deterministic* margin the twin always wins by the same token and the probe
        scores 1.0 again. Real paraphrases sit in between, so the overlaps here are drawn.
    """
    import random as _random

    pairs = []
    for i in range(entries):
        cluster = i % clusters
        rng = _random.Random(i)
        vocabulary = [f"t{cluster}w{j}" for j in range(14)]
        shared = rng.sample(vocabulary, 5)
        pairs.append(
            paraphrase.ParaphrasePair(
                original=" ".join(
                    ["The", "study", "found", "that"] + shared + rng.sample(vocabulary, 3) + ["."]
                ),
                paraphrase=" ".join(
                    ["Investigators", "reported"]
                    + shared
                    + rng.sample(vocabulary, 3)
                    + ["overall."]
                ),
            )
        )
    return paraphrase.ParaphrasePool(
        pairs=tuple(pairs),
        provenance={"pool": "fixture", "entries": entries, "clusters": clusters},
    )


def unit_pool(queries: int = 400, gold: int = 1, source: str = "nq"):
    """
    :param queries: Query units. Must clear ``M + N - k`` for the rung under test *after*
        ``for_split`` takes its tenth, or an example cannot be assembled at all.
    :param gold: Gold documents per query. ``1`` is NQ, ``2`` is a HotpotQA bridge question -- and
        the difference is the pair count, which is the property qdmatch's two ladders differ on.
    :param source: Corpus tag, as ``"nq"`` or ``"hotpotqa"``.

    :returns: A qdmatch unit pool, projected from :func:`retrieval_pool` exactly the way a real
        build projects one -- so the fixture cannot drift from the production path.
    """
    from ctc.tasks.qdmatch.generate import UnitPool, units_from_retrieval

    pool = retrieval_pool(queries=queries, hard=2, corpus=8, source=source, gold=gold)
    return UnitPool(
        units=units_from_retrieval(pool),
        source=f"qdmatch_{'nq' if source == 'nq' else 'hpqa'}",
        provenance={"corpus": "fixture"},
    )


#: Level-0 concept values for the OpenAlex fixture. Nineteen, because that is how many top-level
#: fields the real taxonomy has -- a ceiling more data cannot raise, and the exact reason the
#: capacity clamp exists (port record trap 14). A fixture with unlimited coarse values would let a
#: regression in that clamp pass.
_FIELDS = tuple(f"Field {i}" for i in range(19))


def openalex_pool(
    papers: int = 2000, eval_share: int = 4, eval_year_min: int = 2024
) -> openalex.OpenAlexPool:
    """
    :param papers: Papers in the pool.
    :param eval_share: One paper in this many is published in the held-out years, so both sides of
        ``for_split`` are non-empty.
    :param eval_year_min: The year cut.

    :returns: An OpenAlex pool whose concept hierarchy narrows level by level -- 19 values at L0
        and hundreds by L3 -- so the per-level K clamp is exercised against a real ceiling rather
        than against an index that can supply any K asked of it.
    """
    built = []
    for i in range(papers):
        field = _FIELDS[i % len(_FIELDS)]
        built.append(
            openalex.Paper(
                id=f"W{i}",
                title=f"Paper {i} on {field}",
                abstract=f"Abstract {i} reports a study of {field} across several conditions.",
                year=eval_year_min if i % eval_share == 0 else eval_year_min - 5,
                concepts={
                    0: field,
                    1: f"{field} area {i % 47}",
                    2: f"{field} topic {i % 131}",
                    3: f"{field} niche {i % 313}",
                },
            )
        )
    return openalex.OpenAlexPool(
        papers=tuple(built),
        eval_year_min=eval_year_min,
        provenance={"corpus": "fixture", "papers": papers},
    )


def reorder_pool(books: int = 10, runs_per_book: int = 3, sentences_per_run: int = 850):
    """
    :param books: Distinct books. ``PassagePool.for_split`` cuts by book at a tenth, so this must
        be at least 10 for the eval split to hold more than one.
    :param runs_per_book: Prose runs per book. More than one on purpose: an example's block may
        cross a run seam, and a single-run fixture would never exercise that path.
    :param sentences_per_run: Sentences per run. The whole book has to supply the largest rung
        under test -- 233 passages of ~100 words at 32k -- or the long rungs come out unbuildable
        rather than smaller.

    :returns: A reorder passage pool, built from :func:`book_pool` through the same pure reduction
        a real build uses, so the fixture cannot drift from the production path.
    """
    from ctc.tasks.reorder.sources.gutenberg import passages_from_books

    return passages_from_books(
        book_pool(books=books, runs_per_book=runs_per_book, sentences_per_run=sentences_per_run)
    )
