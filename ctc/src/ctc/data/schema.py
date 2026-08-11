"""
The unified example format: one constructor, and the first real validator this pipeline has had.

Every task emits the same JSON object, and until now that was enforced by convention alone -- 49
independent constructions of the dict across 34 files, no schema, no check. The drift was not
cosmetic: three document shapes against 11 readers calling ``doc.get("title")``; metadata under
``_meta`` in 14 files and bare ``meta`` in 4; ``reorder`` writing ``source_type`` instead of
``source``, so every ``source``-keyed consumer skipped it.

Normalised on write: ``title`` omitted rather than written as ``None``/``""`` (the serializers
already treat all three alike, so this is byte-cleanliness, not behaviour); metadata always
``_meta``; ``source`` always present. Tolerated on read: :func:`meta_of` accepts both metadata
spellings, because shipped eval data uses the bare one and rewriting it would invalidate
checkpoints' format fingerprints for no gain.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..format.registry import TaskSpec
from .gold import check_indices

__all__ = ["make_example", "make_document", "meta_of", "gold_of", "validate", "SchemaError"]

#: Gold lives here unless a spec overrides it via ``extra["gold_field"]`` -- qdmatch uses
#: ``gold_pairs`` and reorder ``gold_order``, neither of which is a list of document indices.
DEFAULT_GOLD_FIELD = "gold_doc_indices"


class SchemaError(ValueError):
    """Raised when an example violates the unified format or its task's declared contract."""


def make_document(text: str, title: Optional[str] = None) -> Dict[str, Any]:
    """
    :param text: Document body. An empty one shortens the context while still counting toward
        ``n``, so a rung silently comes out under its label.
    :param title: Optional; omitted from the result when falsy.

    :returns: The document mapping.

    :raises SchemaError: If ``text`` is empty or not a string.
    """
    if not isinstance(text, str) or not text:
        raise SchemaError(f"document text must be a non-empty string, got {text!r}")
    return {"text": text, "title": title} if title else {"text": text}


def make_example(
    *,
    documents: Sequence[Mapping[str, Any]],
    source: str,
    queries: Sequence[str] = (),
    answers: Sequence[str] = (),
    gold: Any = None,
    gold_field: str = DEFAULT_GOLD_FIELD,
    meta: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Build one unified-format example. Key order is fixed, so a diff between two builds shows only
    real differences.

    :param documents: The context in display order; use :func:`make_document`.
    :param source: Corpus tag. Required -- it is how a mixed corpus is separated later, and how the
        filler-provenance audit detects a leak like the FEVER claims that reached PubMed
        contradiction evals.
    :param queries: Question(s); a list even for one.
    :param answers: Answer(s); legitimately empty for the set-valued tasks, whose answer is the
        gold index structure.
    :param gold: The gold structure -- indices, pairs, groups or a permutation.
    :param gold_field: Where to store it. Read off the task spec, not hardcoded.
    :param meta: Per-example metadata, always stored as ``_meta``.
    :param extra: Additional top-level fields.

    :returns: The example.

    :raises SchemaError: If ``documents`` is empty or ``source`` is blank.
    """
    if not documents:
        raise SchemaError("an example needs at least one document")
    if not source or not isinstance(source, str):
        raise SchemaError(f"source must be a non-empty string, got {source!r}")

    example: Dict[str, Any] = {
        "documents": list(documents),
        "queries": list(queries),
        "answers": list(answers),
    }
    if gold is not None:
        example[gold_field] = gold
    example["source"] = source
    if meta:
        example["_meta"] = dict(meta)
    example.update(extra)
    return example


def meta_of(example: Mapping[str, Any]) -> Dict[str, Any]:
    """
    :param example: A unified-format example.

    :returns: Its metadata under either spelling, or ``{}``.
    """
    return dict(example.get("_meta") or example.get("meta") or {})


def gold_field_for(spec: TaskSpec) -> str:
    """
    :param spec: The task spec.

    :returns: The example key holding this task's gold.
    """
    return (spec.extra or {}).get("gold_field", DEFAULT_GOLD_FIELD)


def gold_of(example: Mapping[str, Any], spec: TaskSpec) -> Any:
    """
    :param example: A unified-format example.
    :param spec: The task spec.

    :returns: The gold structure, or ``None``.
    """
    return example.get(gold_field_for(spec))


def validate(example: Mapping[str, Any], spec: TaskSpec, *, require_gold: bool = True) -> None:
    """
    Check one example against the unified format *and* its task's declared contract.

    The contract half is the point: a structurally perfect example whose gold is counted from the
    wrong base is the failure this suite has actually suffered, and a shape-only check cannot see
    it.

    :param example: The example to check.
    :param spec: The task it claims to belong to.
    :param require_gold: Demand a gold structure. Off for tasks whose gold field is vestigial
        (``oolong`` declares one and always writes ``[]``).

    :raises SchemaError: On any violation, naming the field and what was expected.
    """
    docs = example.get("documents")
    if not isinstance(docs, list) or not docs:
        raise SchemaError(f"{spec.name}: 'documents' must be a non-empty list")
    for i, doc in enumerate(docs):
        if not isinstance(doc, Mapping) or not doc.get("text"):
            raise SchemaError(f"{spec.name}: document {i} has no non-empty 'text'")
    for field in ("queries", "answers"):
        if not isinstance(example.get(field), list):
            raise SchemaError(f"{spec.name}: '{field}' must be a list (empty is fine)")
    if not example.get("source"):
        raise SchemaError(
            f"{spec.name}: 'source' is required -- it is how a mixed corpus is separated later, "
            "and how the filler-provenance audit detects cross-corpus leakage"
        )

    field = gold_field_for(spec)
    gold = example.get(field)
    if gold is None:
        if require_gold:
            raise SchemaError(f"{spec.name}: missing gold field {field!r}")
    elif not isinstance(gold, list):
        raise SchemaError(f"{spec.name}: {field!r} must be a list, got {type(gold).__name__}")
    else:
        check_indices(gold, len(docs), base=spec.gold_index_base, field=field)

    if "hard_neg_indices" in example:
        check_indices(
            example["hard_neg_indices"], len(docs), base=spec.gold_index_base, field="hard_negs"
        )
    scores = example.get("ce_scores")
    if scores is not None and len(scores) != len(docs):
        raise SchemaError(
            f"{spec.name}: 'ce_scores' has {len(scores)} entries for {len(docs)} documents; the "
            "rerank grader indexes it positionally, so it must be parallel"
        )


def validate_all(
    examples: Sequence[Mapping[str, Any]], spec: TaskSpec, *, require_gold: bool = True
) -> List[str]:
    """
    Validate many, collecting problems rather than stopping at the first.

    :param examples: Examples to check.
    :param spec: The task they claim to belong to.
    :param require_gold: See :func:`validate`.

    :returns: One message per bad example, prefixed with its row number; empty when all pass.
    """
    problems = []
    for i, example in enumerate(examples):
        try:
            validate(example, spec, require_gold=require_gold)
        except SchemaError as e:
            problems.append(f"row {i}: {e}")
    return problems
