"""
Document serialization: how a list of documents becomes the context block of a prompt.

Every task renders its corpus differently -- claims are numbered ``Claim 1:``, expressions
``Expression 1:``, oolong and ruler are emitted verbatim with no wrapper at all -- but they all
share one non-negotiable rule: **items are joined by a blank line.**

That is not cosmetic. The chunked/landmark attention masks split the context on ``\\n\\n`` to
decide chunk boundaries, so an item that contains a bare ``\\n`` where the format expects ``\\n\\n``
lands in the wrong chunk, and one that contains an unexpected ``\\n\\n`` gets split across two.
Both failures are invisible in the text and show up only as a degraded score, which is how they
went undiagnosed the first time. The ``reorder`` serializer's ``\\n\\n`` -> ``\\n`` collapse below
is a direct fix for the second case.

Ported from ``corpus_reasoning/lib/data_format.py`` (``_format_doc`` / ``_format_documents``). The
task dispatch was an ``if task == ...`` chain over disjoint names; it is a table here, which is the
same mapping written down once. Output is byte-identical -- see ``ctc/tests/format/`` and the
golden fixture it checks against.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .prompts import (
    CLAIM_TEMPLATE,
    EXPRESSION_TEMPLATE,
    NGRAM_TEMPLATE,
    PASSAGE_TEMPLATE,
    PASSAGE_TEMPLATE_GROUPING,
    PASSAGE_TEMPLATE_ID,
    PASSAGE_TEMPLATE_NO_TITLE,
    PASSAGE_TEMPLATE_NO_TITLE_ID,
    PASSAGE_TEMPLATE_REORDER,
    STRING_TEMPLATE,
    TEXTGROUPS_TEMPLATE,
)

__all__ = [
    "format_doc",
    "format_documents",
    "shows_doc_ids",
    "visible_doc_id_range",
    "ITEM_SEPARATOR",
    "TASKS_WITH_DOC_IDS",
]

#: Items are always joined by a blank line -- the chunk-boundary delimiter. See the module note.
ITEM_SEPARATOR = "\n\n"

#: Tasks whose default-serialized documents carry a visible ``[N]`` identifier, because the answer
#: is a set of those identifiers. Every other task using the default serializer renders documents
#: unnumbered, so a model cannot answer by position.
TASKS_WITH_DOC_IDS = ("retrieval", "cot_retrieval", "outlier", "rerank")

Document = Mapping[str, Any]


def format_doc(doc: Document, use_titles: bool = True, doc_id: Optional[int] = None) -> str:
    """
    Render a single document with the passage template.

    :param doc: A document mapping with a ``text`` key and an optional ``title``.
    :param use_titles: Include the title when the document has one.
    :param doc_id: 1-based identifier to prefix. ``None`` renders the document unnumbered.

    :returns: The formatted document line.
    """
    title = doc.get("title")
    text = doc["text"]
    if doc_id is not None:
        if use_titles and title:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=title, text=text)
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=text)
    if use_titles and title:
        return PASSAGE_TEMPLATE.format(title=title, text=text)
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=text)


def _numbered(template: str) -> Callable[[Sequence[Document]], List[str]]:
    """Build a serializer that renders each item through a ``{id}``/``{text}`` template."""

    def render(documents: Sequence[Document]) -> List[str]:
        return [template.format(id=i + 1, text=doc["text"]) for i, doc in enumerate(documents)]

    return render


def _qdmatch(documents: Sequence[Document]) -> List[str]:
    """Interleaved queries and documents under one shared 1-based index.

    The order is decided by the generator (it encodes the separate-vs-shuffled layout), so this
    only tags each item; it must not re-sort.
    """
    out = []
    for i, item in enumerate(documents):
        tag = "Query" if item.get("type") == "query" else "Document"
        out.append(f"[{i + 1}] {tag}: {item['text']}")
    return out


def _xabsence(documents: Sequence[Document]) -> List[str]:
    """Two corpora A and B, pre-ordered as an A block then a B block, one shared 1-based index."""
    return [
        f"[{i + 1}] {item.get('corpus', 'A')}: {item['text']}" for i, item in enumerate(documents)
    ]


def _absence(documents: Sequence[Document]) -> List[str]:
    """Neutral numbered items -- claims, poem lines, numbers or diff lines by source domain."""
    return [f"[{i + 1}] {doc['text']}" for i, doc in enumerate(documents)]


def _reorder(documents: Sequence[Document]) -> List[str]:
    """Passages with shuffled display IDs.

    Gutenberg passages contain internal ``\\n\\n`` paragraph breaks. They are collapsed to ``\\n``
    so each passage stays a single chunk; without this only a passage's first paragraph gets
    wrapped and the remainder leaks into the unmasked region.
    """
    return [
        PASSAGE_TEMPLATE_REORDER.format(id=i + 1, text=doc["text"].replace("\n\n", "\n"))
        for i, doc in enumerate(documents)
    ]


def _grouping(documents: Sequence[Document]) -> List[str]:
    return [
        PASSAGE_TEMPLATE_GROUPING.format(id=i + 1, title=doc.get("title", ""), text=doc["text"])
        for i, doc in enumerate(documents)
    ]


def _verbatim(documents: Sequence[Document]) -> List[str]:
    """No per-document wrapper.

    ``oolong`` arrives with the benchmark's own formatting already applied; ``summarization``
    concatenates its sources; ``ruler`` needles must read as natural sentences, since a ``[N]``
    prefix would let the model find them by shape instead of by content.
    """
    return [doc["text"] for doc in documents]


#: task name -> item renderer. Tasks absent from this table use the default renderer, which numbers
#: documents only for :data:`TASKS_WITH_DOC_IDS` and is the only serializer honouring ``use_titles``.
_SERIALIZERS: Dict[str, Callable[[Sequence[Document]], List[str]]] = {
    "qdmatch": _qdmatch,
    "xabsence": _xabsence,
    "contradiction": _numbered(CLAIM_TEMPLATE),
    "redundancy": _numbered(CLAIM_TEMPLATE),
    "cycle": _numbered(CLAIM_TEMPLATE),
    "absence": _absence,
    "matching_ngram": _numbered(NGRAM_TEMPLATE),
    "mathmatch": _numbered(EXPRESSION_TEMPLATE),
    "groups4": _numbered(EXPRESSION_TEMPLATE),
    "textgroups": _numbered(TEXTGROUPS_TEMPLATE),
    "strmatch": _numbered(STRING_TEMPLATE),
    "reorder": _reorder,
    "grouping": _grouping,
    "grouping_labeled": _grouping,
    "oolong": _verbatim,
    "summarization": _verbatim,
    "ruler": _verbatim,
}


#: Serializers that emit no identifier at all. Everything else numbers items from 1.
_UNNUMBERED = ("oolong", "summarization", "ruler")


def shows_doc_ids(task: str) -> bool:
    """
    :param task: Task name.

    :returns: Whether this task's rendered context shows a per-item identifier. Only then is
        :data:`~ctc.format.fingerprint.FormatFingerprint.doc_id_range` meaningful -- a task whose
        items are unnumbered cannot suffer the digit-range failure, and recording a range for it
        would invent a constraint.
    """
    if task in _UNNUMBERED:
        return False
    return task in _SERIALIZERS or task in TASKS_WITH_DOC_IDS


def visible_doc_id_range(
    examples: Sequence[Mapping[str, Any]], task: str
) -> Optional[Tuple[int, int]]:
    """
    Measure the span of document identifiers a dataset actually shows the model.

    This is the field behind the digit-range bug: training never rendered an id above 697, eval
    rendered up to 1423, and the model's failure on ids it had never seen read as long-context
    collapse. Measuring it from the data is the point -- the intended corpus size is a claim, and
    the claim is what was wrong.

    :param examples: Unified-format examples.
    :param task: Task name.

    :returns: ``(min, max)`` identifier, or ``None`` when this task renders items unnumbered or
        the examples carry no documents.
    """
    if not shows_doc_ids(task):
        return None
    counts = [len(ex.get("documents") or ()) for ex in examples]
    counts = [c for c in counts if c > 0]
    if not counts:
        return None
    # Every serializer numbers from 1 through len(documents), so the span over a dataset is
    # 1..max. Derived rather than assumed: if a serializer ever numbers differently, this is the
    # one place to fix.
    return (1, max(counts))


def format_documents(documents: Sequence[Document], task: str, use_titles: bool = True) -> str:
    """
    Render a document list into the context block for ``task``.

    :param documents: Document mappings, already in presentation order. This function never
        reorders them -- position is meaningful for several tasks.
    :param task: Task name. Unknown names fall back to the default passage serializer rather than
        raising, matching the behaviour data generation has always had.
    :param use_titles: Include titles. Only the default serializer consults this; the numbered
        per-task templates render text alone.

    :returns: The formatted context block, items joined by a blank line.
    """
    serializer = _SERIALIZERS.get(task)
    if serializer is not None:
        return ITEM_SEPARATOR.join(serializer(documents))

    use_ids = task in TASKS_WITH_DOC_IDS
    return ITEM_SEPARATOR.join(
        format_doc(doc, use_titles=use_titles, doc_id=(i + 1) if use_ids else None)
        for i, doc in enumerate(documents)
    )
