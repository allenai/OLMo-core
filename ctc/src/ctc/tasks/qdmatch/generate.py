"""
``qdmatch``: M queries and N documents under one shared index, a few of them relevant.

The N-squared retrieval task, and the only generator in the suite that is a **transform** rather
than a corpus loader. Its raw material is prepared *retrieval* queries -- a question plus the
document(s) that answer it -- which ``nq`` and ``hotpotqa`` already produce
(:mod:`ctc.data.sources.retrieval`). Nothing is re-fetched and no LLM is involved.

Per example the unit pool is sliced into three **disjoint** groups, and the disjointness is the
whole correctness argument:

* ``k`` *relevant* queries contribute their question **and all of its gold documents** -- these
  pairs are the gold;
* ``M - k`` *distractor queries* contribute a question and no document, so they match nothing;
* the rest contribute one gold document and no question, so they answer nothing.

Because no unit appears twice, a distractor document's own query is never in the example and a
distractor query's own document never is either. Any other true match would be an unlabelled
positive scored as a model error, which is exactly the failure the ``retrieval`` loaders' CE
filters exist to prevent one level down.

``k`` counts relevant **queries, not pairs.** NQ has one gold per question, so ``k`` queries give
``k`` pairs; HotpotQA bridge questions have two, so they give ``2k``.

.. warning::
   **The gold field is ``gold_pairs``, it is 1-based, and the pairs are ORDERED.** A pair is
   ``[query_id, document_id]``; swapping the two makes a different claim, which is why
   :func:`ctc.format.parsing.parse_qd_pairs` refuses to sort. Only the ``separate`` layout is
   built, and that is not a stylistic choice: it is what keeps every gold pair's query id *below*
   its document id, which in turn is what makes the shared nested-ladder shrink safe. See
   :func:`build_example`.

.. note::
   **The source file's hard-negative regime is irrelevant here, deliberately.** A retrieval unit
   contributes only its query and its gold; every negative it carries is discarded. So the
   pre-migration recipe's use of ``nq_train_k20_hn19_2500_aligned.jsonl`` -- a 95%-hard-negative
   file of the kind banned for the ``nq`` ladder itself (port record trap 7) -- carries none of
   that regime into qdmatch. Distractors here come from *other questions' gold*, not from mined
   negatives.

.. note::
   **Combinatorial reuse is expected and is recorded per example.** A 32k example consumes ~374
   units; 20k examples against a ~2.5k-question pool means each question appears in hundreds of
   examples with different partners. ``BUILD_MATRIX.md`` rows 21a/21b flag this as acceptable for a
   matching task -- the pairing is what varies -- but the reuse factor must be reported, so
   ``_meta.pool_units`` carries the pool size every example was drawn from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...data.generators.base import Generator
from ...data.schema import make_document, make_example
from ...data.sources.retrieval import EVAL_FRACTION, RetrievalPool

__all__ = ["QueryUnit", "UnitPool", "load_pool", "build_example", "NQ", "HOTPOTQA"]

#: Where this task's gold lives. Mirrors ``SPEC.extra["gold_field"]``.
GOLD_FIELD = "gold_pairs"


@dataclass(frozen=True)
class QueryUnit:
    """
    One question and the documents that answer it -- the atom qdmatch pools.

    :param query: The question text.
    :param gold: Its gold document bodies, in the source's order. At least one.
    """

    query: str
    gold: Tuple[str, ...]


@dataclass(frozen=True)
class UnitPool:
    """
    A whole corpus's worth of query units.

    :param units: Units **in a fixed, already-shuffled order**, so a split is a slice.
    :param source: The ``source`` tag emitted on every example built from this pool.
    :param provenance: What produced the pool; copied into every example's ``_meta``.
    """

    units: Tuple[QueryUnit, ...]
    source: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.units)

    def for_split(self, split: str) -> "UnitPool":
        """
        Slice off this split's units.

        Disjoint **by question**, using the same 10% every retrieval pool uses. Splitting anywhere
        else would leak: two qdmatch examples built from one question are the same question
        however differently their partners were drawn.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's units.

        :raises ValueError: On an unknown split name.
        """
        cut = max(1, int(round(len(self.units) * EVAL_FRACTION)))
        if split == "eval":
            kept = self.units[:cut]
        elif split == "train":
            kept = self.units[cut:]
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return UnitPool(units=kept, source=self.source, provenance=self.provenance)


def units_from_retrieval(
    pool: RetrievalPool, *, max_gold_per_query: int = 0
) -> Tuple[QueryUnit, ...]:
    """
    Project a prepared retrieval pool onto query units, discarding every negative.

    :param pool: A :class:`~ctc.data.sources.retrieval.RetrievalPool`.
    :param max_gold_per_query: Cap on gold documents kept per query; ``0`` keeps every one the
        source provides (1 for NQ, 2 for HotpotQA bridge). Capping changes the pair count, not the
        question count -- see the module note on ``k``.

    :returns: The units, in the pool's order.
    """
    units = []
    for query in pool.queries:
        gold = [candidate.text for candidate in query.gold if candidate.text]
        if not gold:
            continue
        if max_gold_per_query:
            gold = gold[:max_gold_per_query]
        units.append(QueryUnit(query=query.query, gold=tuple(gold)))
    return tuple(units)


def units_from_jsonl(path: str, *, max_gold_per_query: int = 0) -> Tuple[QueryUnit, ...]:
    """
    Read query units out of built **unified retrieval** JSONL -- the pre-migration path.

    This is what ``generate_qdmatch_data.py --from-train/--from-eval`` consumed, and it is the path
    that builds offline: the staged ``nq_train_k20_hn19_2500_aligned.jsonl`` and
    ``hotpotqa_*_bridge_unified_*.jsonl`` files need no Lucene index, no HF download and no
    cross-encoder.

    :param path: A unified-format retrieval JSONL file.
    :param max_gold_per_query: See :func:`units_from_retrieval`.

    :returns: The units, in file order.

    :raises ValueError: If a row's gold index falls outside its own document list, which means the
        file was written against a different index base and every pair built from it would be
        wrong.
    """
    from ...data.io import load_jsonl

    units = []
    for row in load_jsonl(path):
        documents = row.get("documents") or []
        queries = row.get("queries") or []
        gold_indices = row.get("gold_doc_indices") or []
        if gold_indices and isinstance(gold_indices[0], (list, tuple)):
            # A nested gold is one group per query; these files carry exactly one query per row.
            gold_indices = list(gold_indices[0])
        if not documents or not queries or not gold_indices:
            continue
        bad = [i for i in gold_indices if not 0 <= i < len(documents)]
        if bad:
            raise ValueError(
                f"{path}: gold indices {bad[:4]} fall outside [0, {len(documents) - 1}]. The "
                "retrieval family is 0-based (port record 5.2); a 1-based file read here would "
                "pair every query with the wrong document and still validate."
            )
        if max_gold_per_query:
            gold_indices = gold_indices[:max_gold_per_query]
        gold = tuple(documents[i]["text"] for i in gold_indices if documents[i].get("text"))
        if gold:
            units.append(QueryUnit(query=queries[0], gold=gold))
    return tuple(units)


def load_pool(
    *,
    source: str = "nq",
    path: Optional[str] = None,
    max_gold_per_query: int = 0,
    num_questions: int = 25_000,
    seed: int = 42,
    ce_filter: bool = True,
) -> UnitPool:
    """
    Load the query units qdmatch pools, from a staged retrieval file or from a live loader.

    :param source: ``"nq"`` or ``"hotpotqa"``. Fixes the emitted ``source`` tag either way, so a
        results table can tell the two ladders apart.
    :param path: A built unified-retrieval JSONL to read instead of loading the corpus. **This is
        the offline path**, and the reproducible one: the live loaders mine BM25 or download from
        HF. See :func:`units_from_jsonl`.
    :param max_gold_per_query: Cap on gold documents per query; ``0`` keeps all.
    :param num_questions: Questions to prepare, for the live path only.
    :param seed: Question order, for the live path only.
    :param ce_filter: Cross-encoder filtering, for the live path only. Left on -- it is what makes
        the *gold* trustworthy, and a wrong gold here becomes a wrong pair.

    :returns: The unit pool.

    :raises ValueError: On an unknown source, or when the pool comes out empty.
    """
    if source not in ("nq", "hotpotqa"):
        raise ValueError(
            f"qdmatch source must be 'nq' or 'hotpotqa', got {source!r}. ObliQ was dropped from "
            "the roster on 2026-07-19: it re-entered the suite as a standalone in-context "
            "retrieval row, not as a qdmatch variant (BUILD_MATRIX.md rows 21a/21c)."
        )
    if path:
        units = units_from_jsonl(path, max_gold_per_query=max_gold_per_query)
        provenance: Dict[str, Any] = {"corpus": path, "units": len(units)}
    else:
        if source == "nq":
            from ...data.sources import nq as loader
        else:
            from ...data.sources import hotpotqa as loader  # type: ignore[no-redef]
        pool = loader.load_pool(num_questions=num_questions, seed=seed, ce_filter=ce_filter)
        units = units_from_retrieval(pool, max_gold_per_query=max_gold_per_query)
        provenance = {"corpus": source, "units": len(units), "seed": seed}
    if not units:
        raise ValueError(f"qdmatch pool for {source!r} is empty; nothing to build from")
    return UnitPool(units=units, source=f"qdmatch_{_TAG[source]}", provenance=provenance)


#: Source name -> the short tag the shipped files use in ``source`` and in their filenames.
_TAG = {"nq": "nq", "hotpotqa": "hpqa"}


def build_example(
    rng: random.Random,
    *,
    corpus: UnitPool,
    num_docs: int,
    num_relevant: int = 3,
    layout: str = "separate",
) -> Optional[Dict]:
    """
    Build one qdmatch example.

    :param rng: Seeded RNG.
    :param corpus: The unit pool.
    :param num_docs: **Total items**, i.e. ``M + N``, because that is what the rung ladder and the
        shared shrink both measure. Queries take ``num_docs // 2`` of them and documents the rest;
        the pre-migration recipe's ``--num-docs q --num-queries q`` is this value halved.
    :param num_relevant: ``k``, the number of relevant **queries**. Each contributes every gold it
        has, so the pair count is ``k`` for NQ and ``2k`` for HotpotQA bridge.
    :param layout: ``"separate"`` only -- all queries, then all documents, each block internally
        shuffled.

    :returns: A unified-format example with 1-based, ordered ``gold_pairs``, or ``None`` when this
        draw's relevant queries carry more gold documents than the example has document slots.

    :raises ValueError: On ``num_docs`` too small to hold ``k`` relevant queries and their gold, or
        on the ``shuffled`` layout -- see below.
    :raises RuntimeError: If the pool holds fewer distinct units than one example consumes.
    """
    if layout != "separate":
        raise ValueError(
            f"layout={layout!r} is not built. The 'shuffled' ablation interleaves queries and "
            "documents, so a gold pair's document id can fall BELOW its query id -- and "
            "`ctc.data.build.shrink` sorts within a gold group when it derives a shorter rung, "
            "which would silently swap those pairs into a different claim. 'separate' keeps every "
            "query id below every document id, so that sort is a no-op and the nested ladder is "
            "safe. Re-enable it only alongside a shape-preserving shrink."
        )
    num_queries = num_docs // 2
    n_documents = num_docs - num_queries
    k = num_relevant
    if k > num_queries:
        raise ValueError(
            f"num_relevant={k} exceeds the {num_queries} query slots at num_docs={num_docs}"
        )
    if k < 1 or n_documents < k:
        raise ValueError(
            f"num_docs={num_docs} leaves {n_documents} document slot(s) for {k} relevant query(s)"
        )

    # Upper bound on distinct units consumed: k relevant + (M-k) distractor queries + (N-G)
    # distractor documents, with G >= k. Sampled as one block so no unit can land in two roles.
    need = num_queries + n_documents - k
    if len(corpus.units) < need:
        raise RuntimeError(
            f"{corpus.source}: pool has {len(corpus.units)} usable unit(s) < the {need} an "
            f"example at num_docs={num_docs} consumes; widen the pool or shorten the rung"
        )
    units = rng.sample(list(corpus.units), need)

    relevant = units[:k]
    gold_count = sum(len(unit.gold) for unit in relevant)
    if gold_count > n_documents:
        return None
    distractor_queries = units[k:num_queries]
    # One document per distractor unit, not all of its gold: the arithmetic above budgets one slot
    # each, and a HotpotQA unit's second paragraph is jointly relevant to a query that is not shown
    # anyway.
    distractor_docs = units[num_queries : num_queries + (n_documents - gold_count)]

    # Tag construction-order identity so gold survives the shuffles below without index arithmetic.
    query_items: List[Tuple[Optional[int], str]] = [
        (i if i < k else None, unit.query) for i, unit in enumerate(relevant + distractor_queries)
    ]
    doc_items: List[Tuple[Optional[int], str]] = [
        (i, text) for i, unit in enumerate(relevant) for text in unit.gold
    ]
    doc_items += [(None, unit.gold[0]) for unit in distractor_docs]

    rng.shuffle(query_items)
    rng.shuffle(doc_items)
    items = query_items + doc_items

    query_id: Dict[int, int] = {}
    doc_ids: Dict[int, List[int]] = {}
    for position, (owner, _) in enumerate(items, start=1):
        if owner is None:
            continue
        if position <= len(query_items):
            query_id[owner] = position
        else:
            doc_ids.setdefault(owner, []).append(position)
    gold_pairs = sorted([query_id[i], d] for i in range(k) for d in doc_ids[i])

    documents = [
        dict(make_document(text), type="query" if position < len(query_items) else "document")
        for position, (_, text) in enumerate(items)
    ]
    return make_example(
        documents=documents,
        queries=[],
        answers=[],
        source=corpus.source,
        gold=gold_pairs,
        gold_field=GOLD_FIELD,
        # num_queries/num_docs are deliberately NOT emitted. They were top-level fields
        # pre-migration, and the nested ladder drops distractors in place -- so a shrunk rung would
        # carry counts describing the example it was derived from. Both are recoverable from
        # `documents` by counting `type`. num_relevant and layout survive a shrink unchanged, since
        # gold is never dropped.
        meta={
            "num_relevant": k,
            "layout": layout,
            "pool_units": len(corpus.units),
            "provenance": corpus.provenance,
        },
    )


def _generator(name: str, source: str) -> Generator:
    """
    :param name: Ladder name.
    :param source: ``"nq"`` or ``"hotpotqa"``.

    :returns: The generator for one qdmatch corpus. Two ladders rather than one because the suite
        reports them as two rows (``BUILD_MATRIX.md`` 21a/21b) and a results table that cannot tell
        single-gold NQ from two-gold HotpotQA cannot say anything about multi-gold matching.
    """
    return Generator(
        name=name,
        task="qdmatch",
        source=f"qdmatch_{_TAG[source]}",
        build_example=build_example,
        defaults={"num_docs": 40, "num_relevant": 3, "layout": "separate"},
        corpus=load_pool,
        corpus_defaults={
            "source": source,
            "path": None,
            "max_gold_per_query": 0,
            "num_questions": 25_000,
            "seed": 42,
            "ce_filter": True,
        },
        # An example consumes ~num_docs units, not one, so there is no per-example question index.
        indexed=False,
        notes=(
            f"{source} queries and documents interleaved under one index; gold is ordered "
            "(query, document) pairs"
        ),
    )


#: Natural Questions: one gold per question, so ``k`` relevant queries give ``k`` pairs.
NQ = _generator("qdmatch_nq", "nq")
#: HotpotQA bridge: two gold per question, so ``k`` relevant queries give ``2k`` pairs -- the only
#: qdmatch row where a query maps to more than one document.
HOTPOTQA = _generator("qdmatch_hpqa", "hotpotqa")
