"""
Document-chunked SFT data primitives, shared by the converter (training) and the native eval harness
so the two produce byte-identical token layouts.

Documents are marked with **registered special tokens** ``<|box_start|>`` / ``<|box_end|>`` (the
corpus-reasoning convention -- real Qwen3 reserved tokens, single-id, round-trip cleanly), wrapped at
the *string* level so the tokenizer emits the boundary ids natively. :func:`segment_prompt_to_chunks`
renders a task prompt, wraps each context document/item-line, tokenizes, and recovers the per-document
spans by scanning the boundary ids (:func:`find_chunk_spans`) into a list of :class:`ChunkSegment`.

Two emitters consume those segments:

* :func:`emit_document_chunk_dense` -- the **dense** path: just the wrapped tokens (markers included),
  no landmarks, no padding. Runtime ``chunk_id`` reconstruction
  (:func:`~olmo_core.nn.attention.chunked_mask.build_chunk_ids_from_tokens`) rebuilds the roles from
  the boundary ids; :class:`~olmo_core.nn.attention.DocumentChunkedAttention` masks accordingly.

* :func:`emit_document_chunk_landmark` -- the **landmark** path: **first-fit bin-packs** documents into
  landmark windows (block ``= mem_freq + 1``, last slot a landmark) and inserts a landmark at every
  block-end. A document smaller than a window is kept whole inside a single window (multiple small
  documents may share a window); a document larger than a window starts at a window boundary and spans
  consecutive whole windows. FREE runs (instruction / query / answer) fill greedily. Partial windows
  are filled with ``pad_id`` (marked ``PAD`` at runtime, so non-attendable). Every window is full, so
  the periodic ``is_mem`` landmark pattern stays valid and chunk boundaries stay block-consistent.
"""

import logging
import re
from typing import List, NamedTuple, Optional, Pattern, Tuple

log = logging.getLogger(__name__)

__all__ = [
    "ChunkSegment",
    "DOC_START_ID",
    "DOC_END_ID",
    "DOC_START_STR",
    "DOC_END_STR",
    "find_chunk_spans",
    "segment_prompt_to_chunks",
    "emit_document_chunk_dense",
    "emit_document_chunk_landmark",
]

# Default document-boundary markers: existing Qwen3 reserved special tokens (single id, no vocab
# growth / embedding resize), matching corpus-reasoning ``scripts/lib/chunked_attention.py``.
DOC_START_STR = "<|box_start|>"
DOC_END_STR = "<|box_end|>"
DOC_START_ID = 151648
DOC_END_ID = 151649


class ChunkSegment(NamedTuple):
    """One contiguous run of an example.

    :param tokens: Token ids of the run. For a context chunk these **include** the ``<|box_start|>`` /
        ``<|box_end|>`` boundary tokens (the markers are part of the document span).
    :param label_mask: Per-token loss mask, parallel to ``tokens`` (``True`` only on answer tokens).
    :param is_context_chunk: ``True`` for a context document (isolated; kept whole within a window when
        it fits), ``False`` for a FREE run (instruction / query / answer).
    """

    tokens: List[int]
    label_mask: List[bool]
    is_context_chunk: bool


def find_chunk_spans(
    input_ids: List[int], doc_start_id: int = DOC_START_ID, doc_end_id: int = DOC_END_ID
) -> List[Tuple[int, int]]:
    """
    Scan ``input_ids`` for ``<|box_start|> ... <|box_end|>`` pairs (port of corpus-reasoning
    ``find_chunk_spans``). Returns ``(start, end)`` index pairs **inclusive** of both boundary tokens,
    in document order. Unterminated opens are ignored.
    """
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, tid in enumerate(input_ids):
        if tid == doc_start_id:
            start = i
        elif tid == doc_end_id and start is not None:
            spans.append((start, i))
            start = None
    return spans


def _wrap_item_lines(text: str, item_re: Pattern, start_str: str, end_str: str) -> str:
    """Wrap each line matching ``item_re`` (e.g. OOLONG ``Date: ... || ...`` items) with the boundary
    strings; non-matching lines (intro / instruction / question) stay FREE.

    Consecutive item lines are made **contiguous**: the ``\\n`` separating two item lines is folded
    into the following item's chunk so it is not FREE (matching the document path -- no id / label /
    separator between items leaks as a globally-visible FREE token). Only the ``\\n`` at the boundary
    between a free (non-item) line and an item run stays FREE.
    """
    lines = text.split("\n")
    is_item = [bool(line.strip() and item_re.search(line)) for line in lines]
    out: List[str] = []
    for i, line in enumerate(lines):
        lead = "" if i == 0 else "\n"
        if is_item[i] and i > 0 and is_item[i - 1]:
            # Contiguous item: fold the separating newline INTO this chunk (not FREE).
            out.append(f"{start_str}{lead}{line}{end_str}")
        elif is_item[i]:
            # First item of a run: the leading newline borders free text, so it stays FREE.
            out.append(f"{lead}{start_str}{line}{end_str}")
        else:
            out.append(f"{lead}{line}")
    return "".join(out)


def _wrap_documents(text: str, documents: List[dict], start_str: str, end_str: str) -> str:
    """Wrap the document block with boundary strings so **nothing between the first and last document
    is FREE** -- every id / title / label AND the inter-document ``\\n\\n`` separators are inside a
    chunk. Only the leading instruction / question prefix and the trailing positioned query / answer
    stay FREE (multi-document tasks: absence / retrieval / contradiction / ...).

    The chunks are made **contiguous**: each document's chunk starts where the previous one ended (so
    the separator + ``Document [N] (Title: ...):`` label that precede its body are absorbed into it),
    the first document's chunk starts at the blank line that ends the free prefix, and the last
    document's chunk extends over any trailing whitespace so the document -> suffix separator is not
    FREE either. This is stricter than corpus-reasoning's paragraph wrap (which leaves the ``\\n\\n``
    joins free); here the only FREE tokens are the instruction/question and the query/answer.
    """
    # Locate each document body in order (spans in ORIGINAL-text coordinates; markers inserted after).
    body_spans: List[Tuple[int, int]] = []
    cursor = 0
    n_docs = 0
    n_unmatched = 0
    for d in documents:
        body = str(d.get("text", "")).strip()
        if not body:
            continue
        n_docs += 1
        idx = text.find(body, cursor)
        if idx == -1:
            idx = text.find(body)
        if idx == -1:
            # Formatting altered the text so it no longer occurs verbatim -> this document stays FREE
            # (attends everything), silently breaking chunk isolation for it. Surface it so the loss of
            # isolation is visible rather than silent.
            n_unmatched += 1
            log.warning(
                "document-chunk wrapping: document text not found verbatim in the rendered prompt; "
                "it stays FREE (unisolated). First 80 chars: %r",
                body[:80],
            )
            continue
        body_spans.append((idx, idx + len(body)))
        cursor = idx + len(body)
    if n_unmatched:
        log.warning(
            "document-chunk wrapping: %d/%d documents could not be wrapped (stay FREE / unisolated) "
            "for this example.",
            n_unmatched,
            n_docs,
        )
    if not body_spans:
        return text

    # Build CONTIGUOUS chunk spans covering the whole document block with no FREE gaps.
    chunk_spans: List[Tuple[int, int]] = []
    for i, (bstart, bend) in enumerate(body_spans):
        if i == 0:
            # Start at the blank line ending the free prefix (include the "\n\n" so it isn't FREE).
            para = text.rfind("\n\n", 0, bstart)
            cstart = 0 if para == -1 else para
        else:
            cstart = chunk_spans[i - 1][1]  # contiguous: absorb the separator + this doc's label
        if i == len(body_spans) - 1:
            cend = bend  # last doc: swallow trailing whitespace so the doc->suffix "\n\n" isn't FREE
            while cend < len(text) and text[cend] in "\n\r\t ":
                cend += 1
        else:
            cend = bend
        chunk_spans.append((cstart, cend))

    # Single linear pass: walk the text once emitting (free-gap, start, chunk, end) pieces and join.
    # (Repeated ``out = out[:i] + ... + out[i:]`` insertion is O(docs x len) -- quadratic for
    # many-document examples like contradiction with ~950 claims; this is O(len).) chunk_spans are
    # increasing and contiguous (each cstart == the previous cend for i>0), so ``pos`` only moves
    # forward and a free gap appears only before the first chunk (the prefix).
    pieces: List[str] = []
    pos = 0
    for cstart, cend in chunk_spans:
        if cstart > pos:
            pieces.append(text[pos:cstart])
        pieces.append(start_str)
        pieces.append(text[cstart:cend])
        pieces.append(end_str)
        pos = cend
    if pos < len(text):
        pieces.append(text[pos:])
    return "".join(pieces)


def segment_prompt_to_chunks(
    tok,
    example: dict,
    task: str,
    *,
    query_position: str = "both",
    cot_mode: str = "plan",
    chunk_by: str = "line",
    item_regex: str = r"\|\|",
    include_answer: bool = True,
    use_titles: bool = False,
    doc_start_id: int = DOC_START_ID,
    doc_end_id: int = DOC_END_ID,
    doc_start_str: str = DOC_START_STR,
    doc_end_str: str = DOC_END_STR,
) -> Tuple[List[ChunkSegment], List[int], List[bool]]:
    """
    Render a task prompt with document boundaries marked by special tokens, tokenize, and split it
    into :class:`ChunkSegment`. The single source of truth for both training (``include_answer=True``)
    and eval prefill (``include_answer=False``) so their token layouts match exactly.

    :param tok: A *fast* HuggingFace tokenizer with the Qwen3 chat template and the boundary special
        tokens registered (``<|box_start|>`` / ``<|box_end|>`` already exist in Qwen3).
    :param example: A unified-format example (``documents`` / ``queries`` / ``answers`` / ...).
    :param task: Unified task name (``oolong`` / ``absence`` / ``retrieval`` / ...).
    :param chunk_by: ``"line"`` (each item line matching ``item_regex`` is a document -- OOLONG) or
        ``"document"`` (each ``documents[i]`` is a document).
    :param include_answer: Append the assistant answer (training) or stop at the generation prompt
        (eval prefill).
    :param use_titles: Render document titles (``Document [N] (Title: ...): ...``). Defaults to
        ``False`` -- titles are dropped so a per-document title can't hand the model a shortcut (e.g.
        review-outlier titles that name the product category = the outlier attribute). Must match
        between training and eval (both call this with the same value).

    :returns: ``(segments, ids, mask)`` -- the segment list (for the landmark emitter), the flat token
        ids (markers included; for the dense emitter), and the per-token loss mask.
    """
    from olmo_core.data.corpus_reasoning_prompts import build_prompt

    if not getattr(tok, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for offset-based loss masking.")

    prompt, answer = build_prompt(
        example,
        task=task,
        query_position=query_position,
        use_alpaca=False,
        cot_mode=cot_mode,
        use_titles=use_titles,
    )
    if chunk_by == "line":
        prompt = _wrap_item_lines(prompt, re.compile(item_regex), doc_start_str, doc_end_str)
    elif chunk_by == "document":
        prompt = _wrap_documents(prompt, example.get("documents", []), doc_start_str, doc_end_str)
    else:
        raise ValueError(f"Unknown chunk_by {chunk_by!r}; expected 'line' or 'document'.")

    messages = [{"role": "user", "content": prompt}]
    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if include_answer:
        text = tok.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not text.startswith(prompt_str):
            raise RuntimeError("Rendered prompt is not a prefix of the full conversation.")
    else:
        text = prompt_str

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    boundary = len(prompt_str)
    mask = [bool(include_answer and start >= boundary) for (start, _end) in offsets]

    spans = find_chunk_spans(ids, doc_start_id, doc_end_id)
    segments: List[ChunkSegment] = []
    pos = 0
    for s, e in spans:
        if s > pos:
            segments.append(ChunkSegment(ids[pos:s], mask[pos:s], False))
        segments.append(ChunkSegment(ids[s : e + 1], mask[s : e + 1], True))
        pos = e + 1
    if pos < len(ids):
        segments.append(ChunkSegment(ids[pos:], mask[pos:], False))
    return segments, ids, mask


def emit_document_chunk_dense(segments: List[ChunkSegment]) -> Tuple[List[int], List[bool]]:
    """
    Emit the **dense** (no-landmark) layout: simply the segments' tokens concatenated in order
    (boundary markers already present). The caller appends EOS. Runtime ``chunk_id`` reconstruction
    rebuilds the roles from the boundary ids.
    """
    out_ids: List[int] = []
    out_mask: List[bool] = []
    for seg in segments:
        if len(seg.tokens) != len(seg.label_mask):
            raise ValueError("segment tokens and label_mask must have equal length")
        out_ids.extend(seg.tokens)
        out_mask.extend(seg.label_mask)
    return out_ids, out_mask


def emit_document_chunk_landmark(
    segments: List[ChunkSegment],
    *,
    mem_freq: int,
    mem_id: int,
    pad_id: int,
) -> Tuple[List[int], List[bool]]:
    """
    Emit the **landmark** layout: first-fit bin-pack the segments into landmark windows
    (block ``= mem_freq + 1``) and append a landmark (``mem_id``) at every block-end. See the module
    docstring for the packing rule. The boundary markers ``<|box_start|>`` / ``<|box_end|>`` are
    already inside the context segments' tokens.

    :param segments: The example's ordered :class:`ChunkSegment` list (from
        :func:`segment_prompt_to_chunks`).
    :param mem_freq: Regular tokens between landmarks; block size is ``mem_freq + 1``.
    :param mem_id: Landmark (memory) token id appended at every block-end.
    :param pad_id: Padding id filling a partial window before its landmark (marked ``PAD`` at runtime).

    :returns: ``(input_ids, label_mask)`` lists whose length is a multiple of ``mem_freq + 1``, with a
        landmark at every block-end. Landmark / pad positions are excluded from the loss.
    """
    if mem_freq < 1:
        raise ValueError(f"mem_freq must be >= 1 (got {mem_freq})")

    out_ids: List[int] = []
    out_mask: List[bool] = []
    cur: List[int] = []  # current window content (len <= mem_freq)
    cur_mask: List[bool] = []

    def flush() -> None:
        if not cur:
            return
        pad_n = mem_freq - len(cur)
        out_ids.extend(cur)
        out_ids.extend([pad_id] * pad_n)
        out_ids.append(mem_id)
        out_mask.extend(cur_mask)
        out_mask.extend([False] * pad_n)
        out_mask.append(False)
        cur.clear()
        cur_mask.clear()

    def fill_greedy(tokens: List[int], masks: List[bool]) -> None:
        for t, m in zip(tokens, masks):
            if len(cur) == mem_freq:
                flush()
            cur.append(t)
            cur_mask.append(m)

    for seg in segments:
        toks, msk = list(seg.tokens), list(seg.label_mask)
        if len(toks) != len(msk):
            raise ValueError("segment tokens and label_mask must have equal length")
        if not toks:
            continue
        if seg.is_context_chunk and len(toks) <= mem_freq:
            # Atomic small document: keep whole within one window (start a fresh window if needed).
            if len(cur) + len(toks) > mem_freq:
                flush()
            cur.extend(toks)
            cur_mask.extend(msk)
        elif seg.is_context_chunk:
            # Document larger than a window: start at a window boundary, then span whole windows.
            flush()
            fill_greedy(toks, msk)
        else:
            # FREE run (not a document): fill greedily, may straddle window boundaries.
            fill_greedy(toks, msk)
    flush()
    return out_ids, out_mask
