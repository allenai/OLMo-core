"""
Hashed document IDs: show each document's identifier as a distinct random 5-digit integer instead of
its 1-based position.

**One algorithm, two byte-identical copies.** This file exists verbatim in OLMo-core
(``olmo_core/data/corpus_reasoning_prompts/doc_id_hash.py``, used by the CTC SFT converter's
``--doc-id-style hash``) and in olmo-eval (``olmo_eval/evals/tasks/ctc_suite/doc_id_hash.py``, used
by ``CTC_SUITE_DOC_ID_STYLE=hash``). It is stdlib-only so it can be copied without edits. Both repos
pin :data:`PINNED_IDS` in a unit test, and OLMo-core's
``recipes/eval/setA_audit/check_doc_marker_parity.py --doc-id-style hash`` checks the two files are
identical and that the eval prompt is a token prefix of the training sequence on real rows. Change
one copy, change the other, and bump :data:`SEED_VERSION` if the ids an example gets would change.

**Why.** With ``Document [1] .. [N]`` the largest visible id is the corpus size, so a model trained
up to N documents has never seen an id above N, and an eval with more documents shows it ids it has
never produced (the trap where eval ids exceeded the training id range read as long-context
collapse). Hashed ids come from one fixed range, [10000, 99999], at every corpus size.

**How the ids are drawn.** :func:`hash_doc_ids`: a SHA-256 seed over the documents' texts, then
SHA-256 in counter mode, each 8-byte word reduced mod 90,000, duplicates skipped. Deterministic
(re-rendering an example gives the same ids), independent of Python's ``random`` (which is not
guaranteed stable across versions), distinct within an example, sampled without replacement.

**The renderers are not changed.** A prompt is rendered with sequential ids exactly as before, then:

* :func:`relabel_prompt` rewrites each document's *label* -- the text immediately before its body,
  e.g. ``Document [3]: ``, ``Claim 3: ``, ``[3] Query: `` -- from position to hashed id. Each label
  is located through its document's body and must be found, or it raises. Nothing else in the prompt
  (instructions and their format examples, questions) is touched.
* :func:`relabel_answer` (training targets) maps every integer of an id-only answer through the same
  map. Order is kept: an answer that was sorted by id stays in document order, and a ranking or
  permutation keeps its order. Answers with any non-id content (a CoT) are refused.
* :func:`unmap_ids` (grading) maps every integer in a generation back to its 1-based position before
  the unchanged parsers and graders run. An integer that is not one of this example's ids becomes
  ``n_docs + 1``: out of range, so every parser treats it exactly as it treats any out-of-range id
  in the sequential setting (dropped, or counted wrong) -- never a crash, never a hit.

Tasks with unnumbered items (``oolong``, ``ruler``, ``summarization``, text-diff ``absence``) are
returned unchanged by all three. A task this module has no label for raises rather than silently
staying sequential.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SEED_VERSION",
    "ID_MIN",
    "ID_MAX",
    "DOC_ID_STYLES",
    "PINNED_EXAMPLE",
    "PINNED_IDS",
    "hash_doc_ids",
    "shows_doc_ids",
    "relabel_prompt",
    "relabel_answer",
    "unmap_ids",
]

#: Folded into the seed: bump it when a change would give an example different ids.
SEED_VERSION = "ctc-doc-id-hash-v1"
#: Inclusive id range: every id has exactly five digits.
ID_MIN = 10000
ID_MAX = 99999
_SPAN = ID_MAX - ID_MIN + 1

#: ``seq`` = the 1-based positions every existing shard and eval uses; ``hash`` = this module.
DOC_ID_STYLES = ("seq", "hash")

#: A fixed input and the ids it must get, pinned by a unit test in BOTH repos.
PINNED_EXAMPLE: Dict[str, Any] = {
    "documents": [{"text": "alpha"}, {"text": "beta"}, {"text": ""}, {"text": "gamma delta"}]
}
PINNED_IDS = [49902, 34343, 92463, 80680]

Doc = Mapping[str, Any]

#: Tasks whose items are rendered without an identifier.
_UNNUMBERED = frozenset({"oolong", "ruler", "summarization"})


def _default_heads(i: int, doc: Doc) -> List[str]:
    # format_doc: "Document [i] (Title: t): text" when titles are on and the doc has one, else
    # "Document [i]: text". Which one the renderer used is a caller option, so try the titled form
    # first (it is the more specific) and fall back.
    title = doc.get("title")
    heads = [f"Document [{i}]: "]
    if title:
        heads.insert(0, f"Document [{i}] (Title: {title}): ")
    return heads


#: task -> label(s) that immediately precede a document's rendered text, for id ``i``. Mirrors the
#: serializers of olmo_core ``corpus_reasoning_prompts.data_format._format_documents`` and the
#: eval's vendored ``ctc.format.documents`` (which are byte-identical to each other).
_HEADS: Dict[str, Callable[[int, Doc], List[str]]] = {
    "retrieval": _default_heads,
    "cot_retrieval": _default_heads,
    "outlier": _default_heads,
    "rerank": _default_heads,
    "qdmatch": lambda i, d: [f"[{i}] {'Query' if d.get('type') == 'query' else 'Document'}: "],
    "xformmatch": lambda i, d: [
        f"[{i}] {'Original' if d.get('type') == 'original' else 'Rewrite'}: "
    ],
    "xabsence": lambda i, d: [f"[{i}] {d.get('corpus', 'A')}: "],
    "absence": lambda i, d: [f"[{i}] "],
    "contradiction": lambda i, d: [f"Claim {i}: "],
    "redundancy": lambda i, d: [f"Claim {i}: "],
    "cycle": lambda i, d: [f"Claim {i}: "],
    "recmatch": lambda i, d: [f"Record {i}: "],
    "matching_ngram": lambda i, d: [f"Snippet {i}: "],
    "mathmatch": lambda i, d: [f"Expression {i}: "],
    "groups4": lambda i, d: [f"Expression {i}: "],
    "textgroups": lambda i, d: [f"Passage {i}: "],
    "strmatch": lambda i, d: [f"String {i}: "],
    "reorder": lambda i, d: [f"Passage [{i}]: "],
    "grouping": lambda i, d: [f"Document [{i}](Title: {d.get('title', '')}) "],
    "grouping_labeled": lambda i, d: [f"Document [{i}](Title: {d.get('title', '')}) "],
}

#: Words an id-only answer may contain besides ids and punctuation (anchors and JSON keys).
_ANSWER_WORDS = (
    "Relevant Documents",
    "Relevant Document",
    "Outliers",
    "Ranking",
    "Missing",
    "Unmatched",
    "doc_ids",
    "groups",
)
_ANSWER_CHARS = frozenset('[]{}(),:"# \n')
_INT = re.compile(r"[0-9]+")


def _is_textdiff(example: Mapping[str, Any]) -> bool:
    meta = example.get("_meta") or example.get("meta") or {}
    return isinstance(meta, Mapping) and meta.get("format") == "textdiff"


def shows_doc_ids(example: Mapping[str, Any], task: str) -> bool:
    """
    :returns: Whether ``task`` renders this example's documents with a visible id (and so whether
        the other functions change anything).

    :raises ValueError: For a task this module has no label for -- a hash run must not silently
        keep sequential ids.
    """
    if task in _UNNUMBERED or (task == "absence" and _is_textdiff(example)):
        return False
    if task not in _HEADS:
        raise ValueError(f"doc_id_hash: no document label known for task {task!r}")
    return bool(example.get("documents"))


def hash_doc_ids(example: Mapping[str, Any]) -> List[int]:
    """
    The hashed id of each document, in document order.

    :param example: A unified-format example (``documents`` = list of ``{"text": ...}``).

    :returns: ``len(documents)`` distinct integers in ``[ID_MIN, ID_MAX]``, a function of the
        documents' texts only.
    """
    texts = [str(d.get("text", "")) for d in example.get("documents") or ()]
    n = len(texts)
    if n > _SPAN:
        raise ValueError(f"doc_id_hash: {n} documents exceed the {_SPAN} available ids")
    seed = hashlib.sha256("\x1e".join([SEED_VERSION, *texts]).encode("utf-8")).digest()
    out: List[int] = []
    seen = set()
    counter = 0
    while len(out) < n:
        block = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        counter += 1
        for j in range(0, 32, 8):
            v = ID_MIN + int.from_bytes(block[j : j + 8], "big") % _SPAN
            if v not in seen:
                seen.add(v)
                out.append(v)
                if len(out) == n:
                    break
    return out


def _rendered_text(task: str, doc: Doc) -> str:
    text = str(doc.get("text", ""))
    # reorder's serializer collapses a passage's paragraph breaks so it stays one chunk
    return text.replace("\n\n", "\n") if task == "reorder" else text


def relabel_prompt(
    body: str, example: Mapping[str, Any], task: str, ids: Optional[Sequence[int]] = None
) -> str:
    """
    Rewrite each document label in a sequential-id prompt to its hashed id.

    :param body: The prompt as the renderer produced it (sequential ids), before any document
        markers are inserted.
    :param example: The unified example it was rendered from.
    :param task: The renderer's task / spec name.
    :param ids: Precomputed :func:`hash_doc_ids` (computed when omitted).

    :raises ValueError: If any document's label cannot be located.
    """
    if not shows_doc_ids(example, task):
        return body
    docs = example["documents"]
    ids = list(ids) if ids is not None else hash_doc_ids(example)
    heads_of = _HEADS[task]
    edits: List[Tuple[int, int, str]] = []
    cursor = 0
    for i, doc in enumerate(docs):
        text = _rendered_text(task, doc)
        heads = heads_of(i + 1, doc)
        found: Optional[Tuple[int, int]] = None  # (label start, which head)
        if text:
            pos = body.find(text, cursor)
            while pos != -1 and found is None:
                for k, h in enumerate(heads):
                    start = pos - len(h)
                    if start >= cursor and body.startswith(h, start):
                        found = (start, k)
                        break
                if found is None:
                    pos = body.find(text, pos + 1)
        else:  # an empty document is rendered as its bare label
            hits = [(body.find(h, cursor), k) for k, h in enumerate(heads)]
            hits = [hk for hk in hits if hk[0] != -1]
            if hits:
                found = min(hits)
        if found is None:
            raise ValueError(
                f"doc_id_hash: label of document {i + 1}/{len(docs)} ({heads[-1]!r}) not found "
                f"before its text in the rendered {task} prompt"
            )
        start, k = found
        old = heads[k]
        edits.append((start, start + len(old), heads_of(ids[i], doc)[k]))
        cursor = start + len(old) + len(text)
    pieces: List[str] = []
    pos = 0
    for start, end, new in edits:
        pieces += [body[pos:start], new]
        pos = end
    pieces.append(body[pos:])
    return "".join(pieces)


def relabel_answer(
    answer: str, example: Mapping[str, Any], task: str, ids: Optional[Sequence[int]] = None
) -> str:
    """
    Map every id in an id-only training target to its hashed id, keeping the answer's order.

    :raises ValueError: If the answer holds anything but ids, punctuation and a known anchor word
        (a CoT, a ``Q1:`` prefix), or an integer that is not a valid 1-based id.
    """
    if not shows_doc_ids(example, task):
        return answer
    n = len(example["documents"])
    skeleton = _INT.sub("#", answer)
    for word in _ANSWER_WORDS:
        skeleton = skeleton.replace(word, "")
    extra = set(skeleton) - _ANSWER_CHARS
    if extra:
        raise ValueError(
            f"doc_id_hash: the {task} answer is not id-only (has {sorted(extra)[:8]}); hashed ids "
            f"need an id-only target (cot_mode none): {answer[:120]!r}"
        )
    ids = list(ids) if ids is not None else hash_doc_ids(example)

    def _sub(m: "re.Match[str]") -> str:
        v = int(m.group())
        if not 1 <= v <= n:
            raise ValueError(f"doc_id_hash: answer id {v} outside 1..{n} for {task}")
        return str(ids[v - 1])

    return _INT.sub(_sub, answer)


def unmap_ids(
    text: str, example: Mapping[str, Any], task: str, ids: Optional[Sequence[int]] = None
) -> str:
    """
    Map every integer in a generation back to the 1-based id it stands for; an integer that is not
    one of this example's hashed ids becomes ``n_docs + 1`` (out of range for every parser).
    """
    if not shows_doc_ids(example, task):
        return text
    ids = list(ids) if ids is not None else hash_doc_ids(example)
    inverse = {h: i + 1 for i, h in enumerate(ids)}
    invalid = str(len(ids) + 1)
    return _INT.sub(lambda m: str(inverse.get(int(m.group()), invalid)), text)
