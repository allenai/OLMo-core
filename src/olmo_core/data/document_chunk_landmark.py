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
from typing import Callable, Dict, List, NamedTuple, Optional, Pattern, Tuple

log = logging.getLogger(__name__)

__all__ = [
    "ChunkSegment",
    "DOC_START_ID",
    "DOC_END_ID",
    "DOC_START_STR",
    "DOC_END_STR",
    "EOS_TOKEN_ID",
    "LANDMARK_TOKEN_ID",
    "PAD_TOKEN_ID",
    "REAL_VOCAB_SIZE",
    "RESERVED_IDS",
    "ReservedIds",
    "reserved_ids",
    "find_chunk_spans",
    "segment_prompt_to_chunks",
    "summary_span_text",
    "emit_document_chunk_dense",
    "emit_document_chunk_landmark",
    "emit_document_chunk_summary",
]

# ---------------------------------------------------------------------------------------------
# THE canonical reserved-token ids for the document-chunk / landmark data path.
#
# Import these -- never retype the literals. They were previously re-declared in ~14 files across
# two repos, which is precisely how a mismatch can go unnoticed (a shard built with one id and a
# model/eval expecting another produces plausible-looking numbers, not a crash).
#
# Document-boundary markers are *existing* Qwen3 reserved special tokens, so using them costs no
# vocab growth and no embedding resize. The catch: Qwen3 never TRAINED these rows, so their
# embeddings are bit-identical out-of-distribution vectors -- run
# ``src/scripts/data/fix_marker_embeddings.py`` on any base checkpoint before training on
# marker-bearing shards. See ``document-chunked-marker-embeddings.md``.
# ---------------------------------------------------------------------------------------------
DOC_START_STR = "<|box_start|>"
DOC_END_STR = "<|box_end|>"
DOC_START_ID = 151648
DOC_END_ID = 151649

#: End-of-text. NOTE: Qwen3's tokenizer resolves ``eos_token_id`` to 151645 (``<|im_end|>``), but
#: these SFT shards are trained to stop on 151643 -- pass ``--eos-token-id 151643`` at eval or
#: generation never terminates and rambles (see ``eval-lc-native-nocot-fullpath-bug``).
EOS_TOKEN_ID = 151643

#: Landmark ("memory") token inserted at the end of each landmark window, and the padding token used
#: to fill a short window. Both live PAST the real vocab (``REAL_VOCAB_SIZE``) in the embedding
#: matrix's padded region, so they are untrained too -- same repair applies.
LANDMARK_TOKEN_ID = 151860
PAD_TOKEN_ID = 151863

#: Qwen3's real vocabulary ends here; rows at or beyond this index are untrained padding.
REAL_VOCAB_SIZE = 151669

#: The filler unit for ``free_pad_repeat`` (see :func:`segment_prompt_to_chunks`). Deliberately
#: content-free: it must add FREE *positions* (which attend the whole context) without adding any
#: information about the task, or the experiment would confound "more FREE capacity" with "a better
#: prompt". ~11 Qwen3 tokens per repeat.
FREE_PAD_SENTENCE = "Review the claims above carefully before answering. "


class ReservedIds(NamedTuple):
    """The reserved token ids the document-chunk / landmark path depends on, for one tokenizer.

    The ids are **tokenizer-specific** -- Qwen3 and Qwen3.5 have different vocabularies -- so a
    single module-level constant is not enough. Look the set up by family via :data:`RESERVED_IDS`
    rather than retyping literals, which is how a shard built for one tokenizer and a model expecting
    another silently produce plausible-but-wrong numbers.

    :param doc_start: ``<|box_start|>`` -- opens a context document.
    :param doc_end: ``<|box_end|>`` -- closes a context document.
    :param eos: The id the SFT shards are trained to stop on (NOT necessarily ``tok.eos_token_id``).
    :param landmark: The landmark / "memory" token placed at each landmark-window boundary.
    :param pad: Padding inside a short landmark window.
    :param real_vocab_size: Rows at or beyond this index are untrained embedding padding.
    :param summary: The ``<|summ|>`` token appended after each context document by
        :func:`emit_document_chunk_summary`. Like ``landmark`` and ``pad`` this lives past
        ``real_vocab_size``, so it is an **untrained** row and the base checkpoint must be repaired
        with ``src/scripts/data/fix_marker_embeddings.py`` before training on summary-token shards.
    """

    doc_start: int
    doc_end: int
    eos: int
    landmark: int
    pad: int
    real_vocab_size: int
    summary: int = -1


#: Reserved-id sets by model family. ``qwen3`` mirrors the module-level constants above.
RESERVED_IDS: Dict[str, ReservedIds] = {
    "qwen3": ReservedIds(
        doc_start=151648,
        doc_end=151649,
        eos=151643,
        landmark=151860,
        pad=151863,
        real_vocab_size=151669,
        summary=151866,
    ),
    # Verified against the Qwen3.5-0.8B-Base tokenizer files: base vocab ids 0..248043 plus 33
    # added specials 248044..248076 (``<|endoftext|>``=248044, ``<|box_start|>``=248049,
    # ``<|box_end|>``=248050), so real ids end at 248077; the embedding matrix has 248320 rows, so
    # landmark/pad sit in the untrained padded region [248077, 248320) like Qwen3's do.
    "qwen3_5": ReservedIds(
        doc_start=248049,
        doc_end=248050,
        eos=248044,
        landmark=248200,
        pad=248203,
        real_vocab_size=248077,
        summary=248210,
    ),
    # Verified against the Gemma-3 tokenizer (``google/gemma-3-4b-pt``, 262144 real ids + 64 rows of
    # embedding padding = ``vocab_size`` 262208). Gemma reserves ``<unused0>``..``<unused6241>``
    # (ids 6..262143); the first two stand in for ``<|box_start|>`` / ``<|box_end|>``, which Gemma
    # does not have. EOS is the real ``<eos>``=1 (Gemma's base EOS, matching ``eos_token_id`` in the
    # HF config). ``<image_soft_token>``=262144 is the last real row, so landmark/pad sit in the
    # untrained padded region [262145, 262208) exactly like the Qwen sets do.
    "gemma": ReservedIds(
        doc_start=6,
        doc_end=7,
        eos=1,
        landmark=262150,
        pad=262153,
        real_vocab_size=262145,
        summary=262156,
    ),
    # Llama 3 family (Llama-3.2-3B / Llama-3.1-*). Its tokenizer has 256 added specials at
    # 128000..128255, ~250 of them UNTRAINED ``<|reserved_special_token_N|>`` slots -- exactly what
    # the marker path wants (no vocab growth, no embedding resize). NOTE the embedding matrix is
    # 128256 rows == the full vocab, i.e. there is NO padded region past the vocab, so landmark/pad
    # must also be reserved-special ids rather than out-of-vocab rows.
    #   doc_start 128002 = ``<|reserved_special_token_0|>``, doc_end 128003 = ``..._token_1|>``,
    #   landmark 128011 = ``..._token_3|>``, pad 128012 = ``..._token_4|>``, eos 128001 =
    #   ``<|end_of_text|>`` (the BASE model's EOS; ``<|eot_id|>``=128009 is the chat one).
    # The document-chunk converter wraps documents with the literal strings ``<|box_start|>`` /
    # ``<|box_end|>`` and verifies ``tok.convert_tokens_to_ids`` against these ids, so the Llama
    # runs use a patched tokenizer copy in which those two reserved slots are RENAMED to
    # ``<|box_start|>`` / ``<|box_end|>`` (ids unchanged) -- see
    # ``src/scripts/data/make_llama_marker_tokenizer.py``.
    #   real_vocab_size 128000 = the end of the trained BPE vocab; every id at/after it is a
    # special token and most are untrained, so marker-embedding repair is MANDATORY here too.
    "llama": ReservedIds(
        doc_start=128002,
        doc_end=128003,
        eos=128001,
        landmark=128011,
        pad=128012,
        real_vocab_size=128000,
        summary=128013,
    ),
    # OLMo 3 (``allenai/Olmo-3-1025-7B``, dolma2 tokenizer). The real vocab is 0..100277 (100278
    # ids); olmo-core pads the embedding matrix to 100352, so landmark/pad can sit in the untrained
    # padded region [100278, 100352) exactly like the Qwen sets do.
    #   doc_start 100266 = ``<|extra_id_1|>``, doc_end 100267 = ``<|extra_id_2|>`` -- reserved
    # "extra id" slots that never occur in Dolma, so they cost no vocab growth and no embedding
    # resize. eos 100257 = ``<|endoftext|>`` (the base model's EOS).
    # As with the Llama set, the converter/eval wrap documents with the literal strings
    # ``<|box_start|>`` / ``<|box_end|>`` and verify ``tok.convert_tokens_to_ids`` against these
    # ids, so the OLMo runs use a patched tokenizer copy in which those two extra-id slots are
    # RENAMED to ``<|box_start|>`` / ``<|box_end|>`` (ids unchanged).
    #   Marker-embedding health is MEASURED, not assumed: see
    # ``src/scripts/train/memexpress/ctc_suite/olmo3_marker_audit.py``, which gates on cosine AND
    # norm and repairs from trained delimiter donor rows only if a gate fails.
    "olmo3": ReservedIds(
        doc_start=100266,
        doc_end=100267,
        eos=100257,
        landmark=100300,
        pad=100303,
        real_vocab_size=100278,
        summary=100306,
    ),
}


def reserved_ids(family: str = "qwen3") -> ReservedIds:
    """Look up the reserved-id set for a model family.

    :param family: ``"qwen3"`` or ``"qwen3_5"``.

    :returns: The :class:`ReservedIds` for that family.

    :raises KeyError: If the family is unknown -- better a loud failure than a silent wrong id.
    """
    try:
        return RESERVED_IDS[family]
    except KeyError:
        raise KeyError(
            f"unknown model family {family!r}; known: {sorted(RESERVED_IDS)}. "
            "Add an entry to RESERVED_IDS rather than hardcoding ids at the call site."
        ) from None


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


def summary_span_text(cell_idx: int, k: int, n_docs: int) -> str:
    """
    The natural-language **summary span** for one cell of the ``"summary_attention"`` layout.

    Built entirely from **real, already-trained tokens** -- deliberately NOT a new reserved special
    token. Qwen3 never trains its reserved rows (they are bit-identical, cosine 1.0000, and
    out-of-distribution in norm), which silently flatlines marker-dense training at CE ~0.79 for
    *every* mask including plain causal; `records/document-chunked-marker-embeddings.md` records that
    swapping the markers for ordinary tokens made training converge normally. A phrase therefore needs
    no checkpoint repair and adds no new embedding-bug surface.

    The text restates its own cell's claim indices, so (a) every span is textually **distinct** -- no
    verbatim repetition, the failure mode that voided the ``free_pad_repeat`` probe -- and (b) it
    carries in-chunk index redundancy, the only intervention that has ever *improved* the chunked mask
    (see ``results/masks-n100.md`` §3).

    :param cell_idx: 0-based index of the cell this span summarizes.
    :param k: The cell size (documents per cell).
    :param n_docs: Total number of context documents (clamps the final cell).

    :returns: The span's text, ready to be wrapped in the document-boundary markers.
    """
    lo = cell_idx * k + 1
    hi = min((cell_idx + 1) * k, n_docs)
    slots = " ".join(f"[{i}]" for i in range(lo, hi + 1))
    return f"\n\nSummary of claims {lo} to {hi}: {slots}"


#: Per-task text normalizations applied by the prompt renderer (``_format_documents`` in
#: ``corpus_reasoning/lib/data_format.py``) BEFORE a document's text is embedded in the rendered
#: prompt. ``_wrap_documents`` does a verbatim substring search against the ORIGINAL
#: ``documents[i]["text"]``, so any renderer-side transform must be mirrored here or the search
#: fails and the document silently stays FREE (unisolated) -- see the reorder wrapping bug
#: (docs/records/... n=100 wrapping fix): reorder collapses internal ``"\n\n"`` (Gutenberg
#: passages' own paragraph breaks) to ``"\n"`` so a passage remains one paragraph in the prompt;
#: without the same collapse here, 12/12 documents failed to match on the first n12 example.
_RENDERER_TEXT_NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "reorder": lambda body: body.replace("\n\n", "\n"),
}


def _wrap_documents(
    text: str,
    documents: List[dict],
    start_str: str,
    end_str: str,
    summary_every_k: int = 0,
    task: str = "",
) -> str:
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

    :param summary_every_k: When ``> 0``, emit the ``"summary_attention"`` layout: after every ``k``-th
        document insert an additional, separately-wrapped **summary span** (see
        :func:`summary_span_text`), so chunk indices run on a stride of ``k+1`` and a span is identified
        by ``(chunk_id % (k+1)) == k``. The span is emitted as its **own** chunk, deliberately breaking
        the contiguity rule above -- otherwise it would be absorbed into the *next* document's chunk and
        carry no separate role. It goes at the **END** of its cell because the mask is causal: a relay
        can only carry information forward, so it must already have read what it relays. ``0``
        (default) leaves the layout untouched.
    :param task: Unified task name. Selects a text normalization from
        ``_RENDERER_TEXT_NORMALIZERS`` mirroring what the prompt renderer does to a document's text
        before embedding it, so the verbatim search below matches what's actually in ``text``. Most
        tasks embed document text unmodified (no-op).
    """
    normalize = _RENDERER_TEXT_NORMALIZERS.get(task)
    # Locate each document body in order (spans in ORIGINAL-text coordinates; markers inserted after).
    body_spans: List[Tuple[int, int]] = []
    cursor = 0
    n_docs = 0
    n_unmatched = 0
    for d in documents:
        body = str(d.get("text", "")).strip()
        if not body:
            continue
        if normalize is not None:
            body = normalize(body)
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
            cend = (
                bend  # last doc: swallow trailing whitespace so the doc->suffix "\n\n" isn't FREE
            )
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
    # NB the summary spans are emitted HERE, in this pass -- i.e. AFTER every document body has been
    # located by ``text.find(body, cursor)`` above. Inserting them into ``text`` beforehand would shift
    # the coordinates and could make a body fail to match verbatim, which silently leaves that document
    # FREE (unisolated).
    n_docs = len(chunk_spans)
    pieces: List[str] = []
    pos = 0
    for i, (cstart, cend) in enumerate(chunk_spans):
        if cstart > pos:
            pieces.append(text[pos:cstart])
        pieces.append(start_str)
        pieces.append(text[cstart:cend])
        pieces.append(end_str)
        pos = cend
        # Close the cell: emit its summary span as a separate chunk (also after a short final cell, so
        # every cell has exactly one span and the ``% (k+1)`` stride holds).
        if summary_every_k > 0 and ((i + 1) % summary_every_k == 0 or i == n_docs - 1):
            pieces.append(start_str)
            pieces.append(summary_span_text(i // summary_every_k, summary_every_k, n_docs))
            pieces.append(end_str)
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
    free_pad_repeat: int = 0,
    repeat_doc_text: int = 1,
    summary_every_k: int = 0,
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
    :param summary_every_k: ``chunk_by="document"`` only: emit the ``"summary_attention"`` layout --
        one extra **summary span** chunk after every ``k`` documents (see :func:`summary_span_text` and
        :func:`_wrap_documents`). ``0`` (default) leaves the layout untouched. **Must match between
        training and eval**, like :attr:`free_pad_repeat`: it changes the chunk-index stride to
        ``k+1``, which is exactly what the ``"summary_attention"`` mask's arithmetic assumes, so a
        mismatch silently rebinds every chunk role. The *bandwidth* / *relay* knobs are NOT here --
        they live on the attention config, so one tokenized shard serves every rung of the ladder.

    :returns: ``(segments, ids, mask)`` -- the segment list (for the landmark emitter), the flat token
        ids (markers included; for the dense emitter), and the per-token loss mask.
    """
    from olmo_core.data.corpus_reasoning_prompts import build_prompt

    if not getattr(tok, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for offset-based loss masking.")

    # Duplicate each document's TEXT in place, so the chunk grows but the document COUNT and the
    # document-level attention graph are untouched. This is the control for ``free_pad_repeat``:
    # it adds a comparable number of tokens *inside* chunks (never FREE), so if widening the FREE
    # budget helps and this does not, the effect is specifically FREE-position capacity rather than
    # "more tokens / more compute". Applied before build_prompt so the wrapped span and the prompt
    # body see the same text.
    if repeat_doc_text > 1:
        example = dict(example)
        example["documents"] = [
            {**d, "text": " ".join([str(d.get("text", ""))] * repeat_doc_text)}
            for d in example.get("documents", [])
        ]

    prompt, answer = build_prompt(
        example,
        task=task,
        query_position=query_position,
        use_alpaca=False,
        cot_mode=cot_mode,
        use_titles=use_titles,
    )
    if summary_every_k > 0 and chunk_by != "document":
        raise ValueError(
            f"summary_every_k requires chunk_by='document' (got chunk_by={chunk_by!r}); the "
            "summary_attention chunk-index stride is defined over documents."
        )
    if chunk_by == "line":
        prompt = _wrap_item_lines(prompt, re.compile(item_regex), doc_start_str, doc_end_str)
    elif chunk_by == "document":
        prompt = _wrap_documents(
            prompt,
            example.get("documents", []),
            doc_start_str,
            doc_end_str,
            summary_every_k=summary_every_k,
            task=task,
        )
    else:
        raise ValueError(f"Unknown chunk_by {chunk_by!r}; expected 'line' or 'document'.")

    # FREE padding: extra tokens appended AFTER the wrapped documents and BEFORE the answer. They sit
    # outside every <|box_start|>..<|box_end|> span, so chunk-id reconstruction assigns them the FREE
    # role -- i.e. they attend the WHOLE context under every cross_doc_mode. This is the knob for the
    # "do the FREE tokens saturate?" experiment: document-chunked models score well at 20 documents
    # with ZERO gold-pair connectivity in the context stack (all the cross-document comparison happens
    # at the trailing FREE positions) and collapse at 100 documents. Widening the FREE budget tests
    # whether that collapse is a capacity limit of those positions.
    # Applied here, inside the SINGLE source of truth for train and eval prefill, so the two layouts
    # cannot drift apart.
    if free_pad_repeat > 0:
        prompt = prompt + "\n" + (FREE_PAD_SENTENCE * free_pad_repeat)

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


def emit_document_chunk_summary(
    segments: List[ChunkSegment],
    *,
    summary_token_id: int,
    n_summary_tokens: int,
) -> Tuple[List[int], List[bool]]:
    """
    Emit the **summary-token** layout: the dense layout with a run of ``n_summary_tokens`` copies of
    ``summary_token_id`` appended after every context document.

    The run is emitted **inline**, immediately after the document's closing ``<|box_end|>``, and is
    deliberately *not* given boundary markers of its own:
    :func:`~olmo_core.nn.attention.summary_mask.build_summary_roles` identifies summary tokens by id
    rather than by span, so extra markers would only add tokens that carry no information.

    The trailing query/answer needs no special treatment either. Roles are derived from *completed
    summary runs*, so everything after the last run is the query region by construction -- there is no
    "wrap the query" step and no way for the query to be mistaken for a document.

    Summary positions are excluded from the loss: they are structural, and nothing should be trained
    to emit them.

    :param segments: The example's ordered :class:`ChunkSegment` list (from
        :func:`segment_prompt_to_chunks`).
    :param summary_token_id: The ``<|summ|>`` token id.
    :param n_summary_tokens: How many summary tokens to append per context document. Must match the
        model's :class:`~olmo_core.nn.attention.summary_mask.SummaryMaskSpec`.

    :returns: ``(input_ids, label_mask)``.

    :raises ValueError: If ``n_summary_tokens < 1``, or a segment's tokens and mask disagree.
    """
    if n_summary_tokens < 1:
        raise ValueError(f"n_summary_tokens must be >= 1 (got {n_summary_tokens})")

    out_ids: List[int] = []
    out_mask: List[bool] = []
    for seg in segments:
        if len(seg.tokens) != len(seg.label_mask):
            raise ValueError("segment tokens and label_mask must have equal length")
        out_ids.extend(seg.tokens)
        out_mask.extend(seg.label_mask)
        if seg.is_context_chunk:
            out_ids.extend([summary_token_id] * n_summary_tokens)
            out_mask.extend([False] * n_summary_tokens)
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
