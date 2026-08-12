"""
HotpotQA (``hotpotqa/hotpot_qa``, ``distractor``) -> a :class:`~ctc.data.sources.retrieval.RetrievalPool`.

The multi-hop member of the ``retrieval`` family: **two** gold paragraphs per question, neither of
which answers it alone. ``nq``, ``fiqa`` and ``scifact`` are effectively single-gold, so this is the
one in-distribution ladder where the multi-gold instruction wording and the multi-gold scoring path
are exercised at all.

The knobs here are not inferred from the dataset card; they are read off the corpus the shipped
HotpotQA numbers were computed against -- ``BUILD_MATRIX.md`` row 2 and the staged files
``hotpotqa_train_k{11,24,50,100,205}_bridge_hn{1,2,5,10,20}_4000.jsonl`` plus
``eval_rungs/hotpotqa/rung_{2048,4096,8192,16384}.jsonl``. Four properties were re-verified by
reading those files rather than trusted from the docs: **bridge questions only**, **exactly 2 gold
per question**, gold as a **flat 0-based** list, and hard negatives at exactly **n/10**.

.. note::
   **No titles are rendered.** Every shipped document is ``{"title": "", "text": ...}`` -- bare
   paragraph text. The pre-migration generator had ``--no-titles`` off by default but its
   length-alignment pass (on by default whenever hard negatives were used) blanked the field
   anyway, so blank is what every downstream number saw. It is also the safer default: a HotpotQA
   title is the entity the question is about, so rendering it hands over one hop. Here the title
   survives only as :attr:`~ctc.data.sources.retrieval.Candidate.id`, which is used for
   de-duplication and never emitted.

.. note::
   **The hard negatives are the benchmark's own distractors, not something we mined.** HotpotQA's
   ``distractor`` setting ships 8 TF-IDF-retrieved paragraphs alongside the 2 supporting ones, and
   the resulting "find 2 of 10" is the standard HotpotQA paragraph-retrieval setting.

   This is the one deliberate departure from the shipped pipeline, which mined its hard negatives
   from pyserini's ``wikipedia-dpr-100w`` instead. Those chunks are ~100 words against HotpotQA's
   ~89-word paragraphs *and* carry a leading ``"Title"\\n`` line, so that pipeline needed a whole
   extra pass (``align_hn_doc_lengths.py``, un-ported) to quantile-match the length distributions
   back together before gold stopped being findable by length alone -- measured on the shipped 16k
   rung, that pass leaves gold/hard/random at 71.4/65.5/70.5 mean words, i.e. it works, and it is
   also the only reason it is needed. Drawing every document from one snapshot removes the mismatch
   rather than correcting it, and drops the Lucene index with it.

   The length mismatch does not disappear just because everything now comes from one snapshot,
   though -- it moves *inside* HotpotQA. Its supporting paragraphs average 71.2 words and its
   distractors 96.1, so a corpus filled with distractors makes gold "the short ones". The filler
   pool is therefore drawn from other rows' *supporting* paragraphs; see :func:`filler_candidates`
   for the measurement and for why that is safe.

   It also changes what the cross-encoder is for. On ``nq`` and ``fiqa`` the CE filter exists
   because *we* mined the candidates off BM25 and its top hits are the passages most likely to be
   unlabelled positives. Here the candidate set is labelled by the benchmark, so the filter has a
   narrower job: order them hardest-first (:class:`~ctc.data.sources.retrieval.QueryPool` promises
   that, and a rung takes a *prefix*) and drop the few that out-score the gold outright. Hence
   ``ce_margin`` defaults to **0.0** rather than BEIR's 3.0 -- at 3.0 it would discard most of a
   supply that is only 8 deep.

.. warning::
   **The supply of hard negatives is 8, and that is a property of the corpus, not a knob.** Past
   roughly 80 documents per example, ``hard_frac=0.1`` asks for more hard negatives than a question
   has, so the realised hard fraction falls below 0.1 at the long rungs. That is visible per example
   in ``hard_neg_indices`` and must not be read as a build error. Topping the supply up from an
   external index is exactly what reintroduces the length mismatch above.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .retrieval import Candidate, QueryPool, RetrievalPool, margin_filter

__all__ = ["load_pool", "paragraphs", "prepare_query", "filler_candidates", "GOLD_PER_QUESTION"]

#: Supporting paragraphs per question. Fixed by the benchmark at two -- the multi-hop premise --
#: and re-verified twice: 6000 raw train rows all had exactly two distinct supporting titles, and
#: all 500 examples at all four shipped eval rungs carry exactly two gold indices. A row that does
#: not is dropped rather than emitted with one gold, because a single-gold example silently selects
#: the *single*-doc instruction wording and stops being a multi-hop question.
GOLD_PER_QUESTION = 2


def paragraphs(context: Dict) -> List[Tuple[str, str]]:
    """
    Flatten HotpotQA's parallel-array context into ``(title, text)`` pairs, in context order.

    :param context: A row's ``context``, i.e. ``{"title": [...], "sentences": [[...], ...]}``.

    :returns: One pair per paragraph, sentences joined with a single space.
    """
    return [
        (title, " ".join(sentence.strip() for sentence in sentences))
        for title, sentences in zip(context["title"], context["sentences"])
    ]


def _candidate(title: str, text: str) -> Candidate:
    """
    :param title: The Wikipedia article title, which is also the corpus-stable id.
    :param text: The paragraph body.

    :returns: A candidate carrying the **bare** paragraph, matching every shipped HotpotQA file.
        The title rides along as the id -- de-duplication only, never emitted. Folding it into the
        body the way the BEIR loader does would be wrong here: a BEIR title is part of the passage,
        whereas a HotpotQA title names the entity the question is about and would hand over a hop.
    """
    return Candidate(title, text)


def filler_candidates(row: Dict) -> List[Candidate]:
    """
    The ordinary distractors a row contributes to the shared filler pool.

    Its **supporting** paragraphs, not its retrieved distractors, and that is a length decision
    rather than a relevance one. HotpotQA's two populations are not the same size: over 3000 bridge
    rows the supporting paragraphs average 71.2 words and the TF-IDF-retrieved distractors 96.1.
    Filling with distractors makes gold the short documents in a corpus of long ones -- measured on
    a 17-document build, "name the two shortest" recovered 0.227 of gold against a 0.118 chance
    baseline. Drawing filler from the population gold itself comes from closes that gap by
    construction (0.137 vs 0.118, and ``gold_length_bias`` falls below its own chance baseline),
    and costs nothing, because both populations are ordinary Wikipedia intro paragraphs.

    This is the defect the pre-migration pipeline corrected after the fact with the un-ported
    ``align_hn_doc_lengths.py``. The benchmark's own distractors stay long, but they are the *hard
    negatives*, and being able to spot a hard negative by length does not help find gold.

    :param row: A ``hotpotqa/hotpot_qa`` ``distractor`` row, expected to be outside the question
        slice -- another question's gold is not this question's gold.

    :returns: Candidates in **context order**. Not set order: a set of strings iterates in an order
        that depends on ``PYTHONHASHSEED``, which would make the filler pool -- and so every file
        built from it -- differ between processes given the same seed.
    """
    supporting = set(row["supporting_facts"]["title"])
    return [
        _candidate(title, text)
        for title, text in paragraphs(row["context"])
        if title in supporting and text
    ]


def prepare_query(row: Dict) -> Optional[QueryPool]:
    """
    Turn one raw HotpotQA row into a prepared query. Pure -- no network, no model.

    Kept separate from :func:`load_pool` precisely because this is where a multi-gold task goes
    wrong quietly: mixing one supporting paragraph into the distractors leaves a structurally valid
    example whose second hop is scored as a model error.

    :param row: A ``hotpotqa/hotpot_qa`` ``distractor`` row.

    :returns: The prepared query, with ``hard`` holding the benchmark's own distractors **in context
        order** -- which is not a hardness ranking, so :func:`load_pool`'s cross-encoder pass
        reorders them. ``None`` when the row cannot back a multi-hop example.
    """
    gold_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
    if len(gold_titles) < GOLD_PER_QUESTION:
        return None
    by_title = {title: text for title, text in paragraphs(row["context"])}
    if not all(by_title.get(title) for title in gold_titles):
        # A supporting paragraph missing from the context makes the question unanswerable from what
        # the model is shown. Dropping the row beats emitting gold that points at nothing.
        return None
    gold_set = set(gold_titles)
    return QueryPool(
        query=row["question"],
        gold=tuple(_candidate(title, by_title[title]) for title in gold_titles),
        hard=tuple(
            _candidate(title, text)
            for title, text in paragraphs(row["context"])
            if title not in gold_set and text
        ),
        answers=(row["answer"],),
    )


def _rank_hard(prepared: Sequence[QueryPool], scorer, margin: float) -> List[QueryPool]:
    """
    Re-order each query's distractors hardest-first, dropping any that out-score its gold.

    :param prepared: Queries whose ``hard`` is still in context order.
    :param scorer: Cross-encoder.
    :param margin: Required gap below the best gold score; see the module note on why 0.0.

    :returns: The same queries with ``hard`` ranked. All pairs are scored in one call.
    """
    pairs: List[Tuple[str, str]] = []
    spans: List[Tuple[int, int, int]] = []
    for query in prepared:
        start = len(pairs)
        pairs.extend((query.query, candidate.text) for candidate in query.gold)
        middle = len(pairs)
        pairs.extend((query.query, candidate.text) for candidate in query.hard)
        spans.append((start, middle, len(pairs)))
    if not pairs:
        return list(prepared)
    scores = scorer.score(pairs)
    ranked = []
    for query, (start, middle, end) in zip(prepared, spans):
        hard = margin_filter(scores[start:middle], list(query.hard), scores[middle:end], margin)
        ranked.append(
            QueryPool(query=query.query, gold=query.gold, hard=tuple(hard), answers=query.answers)
        )
    return ranked


def load_pool(
    *,
    split: str = "train",
    num_questions: int = 25_000,
    seed: int = 42,
    question_type: str = "bridge",
    level: str = "all",
    filler_pool: int = 60_000,
    ce_filter: bool = True,
    ce_model: Optional[str] = None,
    ce_margin: float = 0.0,
    batch_size: int = 256,
) -> RetrievalPool:
    """
    Prepare HotpotQA questions with their two gold paragraphs, ranked distractors and a filler pool.

    :param split: ``distractor`` split. ``"train"`` has 90,447 rows; ``"validation"`` is the 7,405-row
        distractor dev set.
    :param question_type: ``"bridge"``, ``"comparison"`` or ``"all"``. Bridge by default because
        every shipped file is bridge-only (``hotpotqa_*_bridge_hn*``, BUILD_MATRIX row 2) and
        because it is the genuinely multi-hop type: a comparison question names *both* its entities
        outright, so its two gold paragraphs are largely findable by string match on the question
        and the ladder would be measuring lookup rather than hops. Bridge is 81% of the train
        split, so the filter costs little.
    :param level: ``"easy"``, ``"medium"``, ``"hard"`` or ``"all"``.
    :param num_questions: Questions to *consider*; the yield is slightly lower because rows whose
        supporting paragraphs are not both present in the context are dropped.
    :param seed: Shuffles the row order, fixing which question example *i* uses.
    :param filler_pool: Ordinary distractor paragraphs to collect, from rows **outside** the
        question slice. Two per row, since only supporting paragraphs are taken -- see
        :func:`filler_candidates`.
    :param ce_filter: Rank the benchmark's distractors hardest-first with a cross-encoder. On by
        default; see the module note on what it is and is not doing here.
    :param ce_model: Cross-encoder id; defaults to the MS MARCO MiniLM every shipped build used, so
        the scores share one scale with the other retrieval ladders.
    :param ce_margin: Required gap below the best gold score. 0.0, not BEIR's 3.0 -- see the module
        note.
    :param batch_size: Questions per cross-encoder call.

    :returns: The pool, questions in a seeded shuffled order.

    :raises ImportError: If ``datasets`` (or ``transformers`` with ``ce_filter``) is missing.
    :raises ValueError: If ``num_questions`` leaves no rows to draw filler from.
    """
    import random

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("HotpotQA loading needs `datasets`: pip install './ctc[sources]'") from e

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    if question_type != "all":
        types = dataset["type"]
        order = [i for i in order if types[i] == question_type]
    if level != "all":
        levels = dataset["level"]
        order = [i for i in order if levels[i] == level]

    # Two DISJOINT slices of the same shuffled order: the first backs the questions, the second only
    # ever contributes filler. Sharing one slice would let a question's own adversarial distractors
    # come back as its "random" fill whenever the CE pass dropped them -- the example would be
    # harder than `hard_neg_indices` records, and that field is the only account of how a pool was
    # built.
    question_ids, filler_ids = order[:num_questions], order[num_questions:]
    if not filler_ids:
        raise ValueError(
            f"num_questions={num_questions} consumes all {len(order)} matching rows, leaving none "
            "to draw filler from; lower it or widen question_type/level"
        )

    scorer = None
    if ce_filter:
        from .retrieval import cross_encoder

        scorer = cross_encoder(ce_model)

    prepared: List[QueryPool] = []
    for start in range(0, len(question_ids), batch_size):
        batch = [
            query
            for query in (
                prepare_query(row)
                for row in dataset.select(question_ids[start : start + batch_size])
            )
            if query is not None
        ]
        prepared.extend(batch if scorer is None else _rank_hard(batch, scorer, ce_margin))

    fillers: List[Candidate] = []
    seen: set = set()
    for row in dataset.select(filler_ids):
        if len(fillers) >= filler_pool:
            break
        for candidate in filler_candidates(row):
            if candidate.id in seen:  # one paragraph per Wikipedia title across the whole pool
                continue
            seen.add(candidate.id)
            fillers.append(candidate)

    return RetrievalPool(queries=tuple(prepared), corpus=tuple(fillers), source="hotpotqa")
