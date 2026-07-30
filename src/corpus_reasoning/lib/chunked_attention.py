"""Chunked document attention with block-diagonal masking.

Standard causal attention lets every token attend to all preceding tokens,
meaning each document can "see" other documents in the context. Chunked
attention restricts this: document tokens can only attend within their own
document AND to "free" tokens (query, instruction, padding) that precede them.

This isolates documents from each other while still allowing each document
to attend to the query/instruction. The hypothesis is that this prevents
shortcut learning where the model relies on cross-document attention patterns.

Implementation:
  - Special tokens <|doc_start|> and <|doc_end|> mark document boundaries.
  - wrap_documents() inserts these tokens around each "Document ..." block.
  - build_chunked_causal_mask() constructs a 4D attention mask where:
      * "Free" tokens (outside any doc) attend causally to all preceding tokens
      * Document tokens attend to: same-doc tokens + free tokens (causal)
      * Document tokens do NOT attend to tokens in other documents

The mask logic: causal AND (same_chunk OR row_free OR col_free)

Beyond the basic chunked pattern, this module also exposes an
AttentionPattern factory (build_flex_mask_mod / build_dense_bool_mask) that
produces masks for several attention variants used in ablations:

  - "standard":           full causal (no per-doc isolation)
  - "chunked":            within-doc only (original pattern above)
  - "doc_window" (k):     within-doc + the k previous docs (causal doc-window)
  - "last_token_anchor":  within-doc + each doc's last token (<|doc_end|>) is
                          globally attendable as KV (Longformer-style anchor,
                          causal-adapted so the anchor carries a summary)
  - "token_window" (w):   within-doc + a raw token-level causal window of
                          width w (can cross doc boundaries, for SWA ablations)
  - "bigbird":            doc_window(k) + last_token_anchor + a small fixed
                          number of random earlier-doc edges per doc
  - "random_token" (p):   within-doc + each cross-doc (q_token, k_token) edge
                          included independently with probability p. p=0
                          collapses to "chunked"; p=1 collapses to "standard".

All patterns keep the invariant that FREE tokens (query/answer/instruction)
attend to everything and are attended to by everything. Only context-context
pairs are restricted.
"""

from dataclasses import dataclass
from typing import Optional

import re
import torch

DOC_START = "<|doc_start|>"
DOC_END = "<|doc_end|>"

# Chunk ID conventions, shared by train and eval. Duplicated constants in
# train_chunked_fast.py should import from here.
PAD_CHUNK_ID = -2
FREE_CHUNK_ID = -1


def setup_tokenizer(tokenizer):
    """Add doc boundary tokens to tokenizer. Returns (doc_start_id, doc_end_id)."""
    tokenizer.add_special_tokens({"additional_special_tokens": [DOC_START, DOC_END]})
    return tokenizer.convert_tokens_to_ids(DOC_START), tokenizer.convert_tokens_to_ids(DOC_END)


# Paragraph-prefix patterns that identify a chunkable unit. One per task family:
#   retrieval / qa / outlier / grouping:  "Document (...", "Document [...", "Document:"
#   reorder:                               "Passage [..."
#   contradiction:                         "Claim N: ..." (claims are \n-separated,
#                                          so the whole claim block ends up as one chunk)
#   matching_ngram:                       "Snippet N: ..."
#   mathmatch:                             "Expression N: ..."
DOC_PREFIXES = ("Document (", "Document [", "Document:", "Passage [", "Claim ", "Snippet ", "Expression ")

# Alpaca's "### Input:\n" header can land glued to the first document when
# there's no leading question (qafter, grouping, outlier, etc.) because both
# sides are separated by a single \n. Splitting the prompt on \n\n then
# produces a single paragraph like "### Input:\nDocument [1] ...", which the
# old prefix check missed — silently dropping the first doc's wrapping. We
# match and peel off that header before classifying.
ALPACA_INPUT_HEADER_RE = re.compile(r'^(###\s+Input:\s*\n)')


def _wrap_paragraph(p: str) -> str:
    """If paragraph `p` contains a document/claim/passage block, wrap it with
    boundary tokens and return the result; otherwise return `p` unchanged."""
    stripped = p.strip()
    if not stripped:
        return p

    # Peel off a leading "### Input:\n" if present so it stays as free text.
    header_match = ALPACA_INPUT_HEADER_RE.match(stripped)
    if header_match:
        header = stripped[: header_match.end()]
        body = stripped[header_match.end():]
    else:
        header, body = "", stripped

    if body.startswith(DOC_PREFIXES):
        return f"{header}{DOC_START}{body}{DOC_END}"
    return p


def wrap_documents(text: str) -> str:
    """Wrap each document/claim/passage block with doc-boundary tokens.

    Recognized formats (any paragraph starting with one of these is chunked):
      - "Document (Title: ...): ..."      (qa / retrieval without IDs)
      - "Document [N] (Title: ...): ..."  (retrieval / cot_retrieval / outlier)
      - "Document [N](Title: ...) ..."    (grouping)
      - "Passage [N]: ..."                 (reorder)
      - "Claim N: ..."                     (contradiction — claim block is
                                            \\n-separated so all claims end up
                                            in a single chunk, which is what
                                            contradiction detection needs since
                                            it requires cross-claim attention)

    Non-document text (question, instruction, alpaca markers, dummy tokens)
    is left unwrapped — the attention mask treats unwrapped tokens as "free"
    (visible to all).
    """
    # Step 1: Separate the question(s) from the document section so we don't
    # accidentally wrap them. Only applies to formats that use "Question:" as a
    # marker (retrieval / qa / cot_retrieval). Other tasks have no such marker
    # and fall through to the full-text path.
    question_before_match = re.match(r'^(Question:.*?\n\n)', text)
    question_after_idx = text.rfind("\n\nQuestion:")

    query_before, query_after = "", ""
    if question_before_match:
        query_before = question_before_match.group(1)
        doc_section = text[len(query_before):]
        q_after_idx = doc_section.rfind("\n\nQuestion:")
        if q_after_idx != -1:
            query_after = doc_section[q_after_idx:]
            doc_section = doc_section[:q_after_idx]
    elif question_after_idx != -1:
        doc_section = text[:question_after_idx]
        query_after = text[question_after_idx:]
    else:
        doc_section = text

    # Step 2: Split on paragraph boundaries and wrap each doc-like paragraph.
    parts = doc_section.split("\n\n")
    wrapped = [_wrap_paragraph(p) for p in parts]

    # Step 3: Reassemble with the question(s) in their original positions.
    return query_before + "\n\n".join(wrapped) + query_after


def reorder_query(text: str, position: str = "after") -> str:
    """Move the 'Question: ...' line before or after documents in the input.

    Args:
        text: The input field (documents + question).
        position: "before" or "after".

    Returns:
        Reordered text.
    """
    if position == "after":
        # Check if question is at the start and move to end
        m = re.match(r'^(Question:.*?)(\n\n)([\s\S]*)', text)
        if m:
            return m.group(3) + "\n\n" + m.group(1)
        return text  # already after or no question found
    elif position == "before":
        # Check if question is at the end and move to start
        m = re.search(r'\n\n(Question:.*)$', text)
        if m:
            return m.group(1) + "\n\n" + text[:m.start()]
        return text  # already before or no question found
    elif position == "both":
        # Put question both before and after documents
        # First, find and extract the question
        m = re.search(r'\n\n(Question:.*)$', text)
        if m:
            question = m.group(1)
            docs = text[:m.start()]
            return question + "\n\n" + docs + "\n\n" + question
        # Maybe question is already at the start
        m = re.match(r'^(Question:.*?)(\n\n)([\s\S]*)', text)
        if m:
            question = m.group(1)
            docs = m.group(3)
            return question + "\n\n" + docs + "\n\n" + question
        return text
    else:
        raise ValueError(f"Invalid query position: {position!r}, expected 'before', 'after', or 'both'")


def find_chunk_spans(input_ids, doc_start_id, doc_end_id):
    """Find (start, end_exclusive) index spans for each document chunk.

    Scans token IDs for matching <|doc_start|>...<|doc_end|> pairs.
    The boundary tokens themselves are included in the span, so the mask
    treats them as part of the document (they only attend within their chunk).
    """
    spans = []
    start = None
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    for i, tid in enumerate(ids):
        if tid == doc_start_id:
            start = i
        elif tid == doc_end_id and start is not None:
            spans.append((start, i + 1))  # +1 to include the end token
            start = None
    return spans


def build_chunked_causal_mask(input_ids, doc_start_id, doc_end_id, dtype=torch.bfloat16):
    """Build 4D attention mask with block-diagonal document attention.

    - Tokens inside a document chunk attend within that chunk AND to
      non-chunk (query/instruction) tokens that precede them (causal).
    - Tokens outside any chunk attend to all preceding tokens (standard causal).

    This allows each document to "see" the query, while remaining isolated
    from other documents.

    Args:
        input_ids: (seq_len,) tensor of token IDs.
        doc_start_id, doc_end_id: special token IDs.
        dtype: mask dtype.

    Returns:
        (1, 1, seq_len, seq_len) float mask. 0 = attend, -inf = masked.
    """
    seq_len = len(input_ids)
    spans = find_chunk_spans(input_ids, doc_start_id, doc_end_id)

    if not spans:
        # No documents found — fall back to standard causal mask.
        # This happens when standard_attention=True or the input has no docs.
        mask = torch.triu(torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    # Step 1: Assign each token a chunk ID. Tokens outside any document get -1
    # ("free" tokens: instruction, question, padding, dummy tokens).
    chunk_id = torch.full((seq_len,), -1, dtype=torch.long)
    for idx, (s, e) in enumerate(spans):
        chunk_id[s:e] = idx

    # Step 2: Build the attention mask as a boolean matrix.
    # The mask is: causal AND (same_chunk OR row_free OR col_free)
    #
    # Intuition for each condition (all subject to causal — can't look ahead):
    #   same_chunk:  tokens in document 3 can attend to other tokens in document 3
    #   row_free:    free tokens (question, instruction) can attend to everything
    #   col_free:    all tokens can attend to free tokens (so docs can see the query)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    same_chunk = (chunk_id.unsqueeze(0) == chunk_id.unsqueeze(1)) & (chunk_id.unsqueeze(0) >= 0)
    row_free = (chunk_id < 0).unsqueeze(1).expand(-1, seq_len)
    col_free = (chunk_id < 0).unsqueeze(0).expand(seq_len, -1)
    bool_mask = causal & (same_chunk | row_free | col_free)

    # Step 3: Convert bool mask to float mask for SDPA.
    # SDPA expects 0.0 = attend, -inf = masked (added to attention logits).
    min_val = torch.finfo(dtype).min
    float_mask = torch.where(bool_mask, torch.zeros(1, dtype=dtype), torch.full((1,), min_val, dtype=dtype))
    # Shape: (1, 1, seq_len, seq_len) — batch=1, heads=1 (broadcast across heads)
    return float_mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Attention pattern factory
# ---------------------------------------------------------------------------

ATTENTION_PATTERNS = (
    "standard", "chunked", "doc_window", "last_token_anchor",
    "token_window", "bigbird", "random_token", "hierarchical_anchor",
)


@dataclass
class AttentionPattern:
    """Configuration for one attention pattern variant.

    All patterns preserve the invariant that FREE tokens (chunk_id = -1:
    query / answer / instruction) attend to everything and are attended to
    by everything. Parameters only affect context-context edges.
    """
    name: str = "chunked"
    # "doc_window" and "bigbird": doc i attends to docs [i-k, i].
    doc_window_k: int = 0
    # "token_window": raw token-level causal window width.
    token_window_w: int = 0
    # "bigbird": number of random earlier-doc edges per doc.
    num_random_doc_edges: int = 0
    # "random_token": Bernoulli keep-prob for each cross-doc (q_tok, k_tok)
    # edge. 0.0 collapses to "chunked"; 1.0 collapses to "standard".
    keep_prob: float = 1.0
    # "hierarchical_anchor": at each layer l (transformer layer), each chunk
    # attends to itself + its `num_anchors` nearest *strictly past* anchor
    # chunks at that layer's stride s_l. Strides go 1, stride_base,
    # stride_base^2, ... — each stride level is assigned to a contiguous
    # block of transformer layers (smaller strides at shallower layers), so
    # the per-chunk receptive field grows exponentially with depth.
    num_anchors: int = 2
    stride_base: int = 2
    # Seed for random sampling (bigbird random edges, random_token Bernoulli).
    # Deterministic per example via `random_seed + example_index`.
    random_seed: int = 42

    @classmethod
    def from_config(cls, cfg: dict) -> "AttentionPattern":
        """Parse an AttentionPattern from a training/eval YAML config.

        Precedence:
          - If `attention_pattern` is set, use it (and its sibling params).
          - Else if `standard_attention: true`, default to "standard".
          - Else default to "chunked" (the original block-diagonal pattern).
        """
        name = cfg.get("attention_pattern")
        if name is None:
            name = "standard" if cfg.get("standard_attention", False) else "chunked"
        if name not in ATTENTION_PATTERNS:
            raise ValueError(
                f"Unknown attention_pattern={name!r}; expected one of {ATTENTION_PATTERNS}"
            )
        return cls(
            name=name,
            doc_window_k=int(cfg.get("doc_window_k", 0)),
            token_window_w=int(cfg.get("token_window_w", 0)),
            num_random_doc_edges=int(cfg.get("num_random_doc_edges", 0)),
            keep_prob=float(cfg.get("keep_prob", 1.0)),
            num_anchors=int(cfg.get("num_anchors", 2)),
            stride_base=int(cfg.get("stride_base", 2)),
            random_seed=int(cfg.get("random_seed", 42)),
        )

    def tag(self) -> str:
        """Short string for run names / output dirs."""
        if self.name == "doc_window":
            return f"docwin-k{self.doc_window_k}"
        if self.name == "token_window":
            return f"tokwin-w{self.token_window_w}"
        if self.name == "bigbird":
            return f"bigbird-k{self.doc_window_k}-r{self.num_random_doc_edges}"
        if self.name == "last_token_anchor":
            return "anchor"
        if self.name == "random_token":
            return f"randtok-p{int(round(self.keep_prob * 100))}"
        if self.name == "hierarchical_anchor":
            return f"hieranc-N{self.num_anchors}-b{self.stride_base}"
        return self.name  # "standard", "chunked"

    def needs_anchor_tensor(self) -> bool:
        return self.name in ("last_token_anchor", "bigbird")

    def needs_random_edges(self) -> bool:
        return self.name == "bigbird" and self.num_random_doc_edges > 0

    def needs_random_token_mask(self) -> bool:
        return self.name == "random_token"


def build_is_anchor(input_ids: torch.Tensor, doc_end_id: int) -> torch.Tensor:
    """Return a bool tensor marking <|doc_end|> positions as anchors.

    Uses the doc-end boundary token as a per-doc "summary" position: by the
    time <|doc_end|> is emitted, causal attention has let its hidden state
    absorb every token in its document. Making this the anchor means later
    docs attending to an anchor see a real summary, not an empty prefix.

    Shape: same shape as input_ids (typically (seq_len,) or (B, S)).
    """
    return input_ids == doc_end_id


def build_random_doc_edges(num_docs: int, num_edges: int, seed: int,
                           max_docs: int) -> torch.Tensor:
    """Build a [max_docs, max_docs] bool adjacency matrix of random edges.

    For each doc i in [0, num_docs), sample `num_edges` distinct earlier docs
    uniformly from [0, i) and mark those edges True. The diagonal is NOT set
    here (within-doc attention is handled by the base pattern logic).

    The matrix is padded to [max_docs, max_docs] with False so the collator
    can stack variable-doc-count examples into a single batch tensor.
    """
    import numpy as np

    adj = torch.zeros((max_docs, max_docs), dtype=torch.bool)
    if num_edges <= 0 or num_docs <= 1:
        return adj
    rng = np.random.default_rng(seed)
    for i in range(1, num_docs):
        k = min(num_edges, i)
        picks = rng.choice(i, size=k, replace=False)
        for j in picks:
            adj[i, int(j)] = True
    return adj


def build_random_token_keep(seq_len: int, keep_prob: float, seed: int) -> torch.Tensor:
    """Bernoulli(keep_prob) sample for every (q_idx, kv_idx) pair.

    Returned tensor is shape (S, S) bool, deterministic for `seed`. The
    caller is responsible for combining this with cross-doc / causal /
    pad gating in the mask_mod — this function only produces the raw
    coin flips.

    p == 0.0 / 1.0 short-circuit to constant tensors so the limits are
    exactly chunked / standard with no FP edge cases.
    """
    if keep_prob >= 1.0:
        return torch.ones((seq_len, seq_len), dtype=torch.bool)
    if keep_prob <= 0.0:
        return torch.zeros((seq_len, seq_len), dtype=torch.bool)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.rand((seq_len, seq_len), generator=g) < keep_prob


def build_flex_mask_mod(pattern: AttentionPattern, chunk_ids: torch.Tensor,
                        is_anchor: Optional[torch.Tensor] = None,
                        doc_random: Optional[torch.Tensor] = None,
                        random_keep: Optional[torch.Tensor] = None):
    """Return a mask_mod closure for torch.nn.attention.flex_attention.

    Args:
        pattern: AttentionPattern describing the variant.
        chunk_ids: (B, S) int32. >=0: doc index; -1: free; -2: pad.
        is_anchor: (B, S) bool. Required for "last_token_anchor" and "bigbird".
        doc_random: (B, D, D) bool. Required for "bigbird" with random edges.
        random_keep: (B, S, S) bool. Required for "random_token" — per-(q,k)
            Bernoulli flips combined with the cross-doc gate inside the closure.

    The returned closure is referentially transparent in the captured tensors
    and safe to hand to FlexAttention's create_block_mask under torch.compile.
    """
    name = pattern.name
    k_win = pattern.doc_window_k
    t_win = pattern.token_window_w

    if name == "standard":
        # Plain causal, padding-aware. FREE and doc ids are treated the same
        # — no per-doc restriction, just don't attend to / from pad.
        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            qc = chunk_ids[b, q_idx]
            kc = chunk_ids[b, kv_idx]
            return causal & (qc != PAD_CHUNK_ID) & (kc != PAD_CHUNK_ID)
        return mask_mod

    if name == "chunked":
        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            qc = chunk_ids[b, q_idx]
            kc = chunk_ids[b, kv_idx]
            q_not_pad = qc != PAD_CHUNK_ID
            kv_not_pad = kc != PAD_CHUNK_ID
            same_chunk = (qc == kc) & (qc >= 0)
            q_free = qc == FREE_CHUNK_ID
            kv_free = kc == FREE_CHUNK_ID
            return causal & q_not_pad & kv_not_pad & (same_chunk | q_free | kv_free)
        return mask_mod

    if name == "doc_window":
        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            qc = chunk_ids[b, q_idx]
            kc = chunk_ids[b, kv_idx]
            q_not_pad = qc != PAD_CHUNK_ID
            kv_not_pad = kc != PAD_CHUNK_ID
            q_free = qc == FREE_CHUNK_ID
            kv_free = kc == FREE_CHUNK_ID
            # Context-context allowed iff qc in [kc, kc + k_win] (causal already
            # enforces qc >= kc at the doc level, since docs lay out in order).
            doc_diff = qc - kc
            doc_win_ok = (doc_diff >= 0) & (doc_diff <= k_win) & (qc >= 0) & (kc >= 0)
            return causal & q_not_pad & kv_not_pad & (doc_win_ok | q_free | kv_free)
        return mask_mod

    if name == "last_token_anchor":
        if is_anchor is None:
            raise ValueError("last_token_anchor requires is_anchor tensor")

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            qc = chunk_ids[b, q_idx]
            kc = chunk_ids[b, kv_idx]
            q_not_pad = qc != PAD_CHUNK_ID
            kv_not_pad = kc != PAD_CHUNK_ID
            q_free = qc == FREE_CHUNK_ID
            kv_free = kc == FREE_CHUNK_ID
            same_chunk = (qc == kc) & (qc >= 0)
            # KV anchor is globally attendable from any context token (other
            # docs can see this doc's summary). We require kc >= 0 so the
            # anchor only takes effect for real doc tokens, not pad.
            anchor_ok = is_anchor[b, kv_idx] & (kc >= 0)
            return causal & q_not_pad & kv_not_pad & (
                same_chunk | anchor_ok | q_free | kv_free
            )
        return mask_mod

    if name == "token_window":
        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            qc = chunk_ids[b, q_idx]
            kc = chunk_ids[b, kv_idx]
            q_not_pad = qc != PAD_CHUNK_ID
            kv_not_pad = kc != PAD_CHUNK_ID
            q_free = qc == FREE_CHUNK_ID
            kv_free = kc == FREE_CHUNK_ID
            same_chunk = (qc == kc) & (qc >= 0)
            tok_diff = q_idx - kv_idx
            # Token-window applies only between doc tokens — free tokens are
            # already fully connected. It can cross doc boundaries.
            tok_win_ok = (tok_diff >= 0) & (tok_diff <= t_win) & (qc >= 0) & (kc >= 0)
            return causal & q_not_pad & kv_not_pad & (
                same_chunk | tok_win_ok | q_free | kv_free
            )
        return mask_mod

    if name == "bigbird":
        if is_anchor is None:
            raise ValueError("bigbird requires is_anchor tensor")

        # Pre-clamp qc/kc to safe indices so doc_random[b, qc, kc] doesn't
        # wrap on -1/-2. The subsequent `(qc>=0)&(kc>=0)` gate nullifies the
        # fake lookups we did for pad/free positions.
        if doc_random is not None:
            def mask_mod(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                qc = chunk_ids[b, q_idx]
                kc = chunk_ids[b, kv_idx]
                q_not_pad = qc != PAD_CHUNK_ID
                kv_not_pad = kc != PAD_CHUNK_ID
                q_free = qc == FREE_CHUNK_ID
                kv_free = kc == FREE_CHUNK_ID
                doc_diff = qc - kc
                win_ok = (doc_diff >= 0) & (doc_diff <= k_win) & (qc >= 0) & (kc >= 0)
                anchor_ok = is_anchor[b, kv_idx] & (kc >= 0)
                safe_qc = torch.clamp(qc, min=0)
                safe_kc = torch.clamp(kc, min=0)
                rand_ok = doc_random[b, safe_qc, safe_kc] & (qc >= 0) & (kc >= 0)
                return causal & q_not_pad & kv_not_pad & (
                    win_ok | anchor_ok | rand_ok | q_free | kv_free
                )
            return mask_mod
        else:
            # No random edges — bigbird reduces to window + anchors.
            def mask_mod(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                qc = chunk_ids[b, q_idx]
                kc = chunk_ids[b, kv_idx]
                q_not_pad = qc != PAD_CHUNK_ID
                kv_not_pad = kc != PAD_CHUNK_ID
                q_free = qc == FREE_CHUNK_ID
                kv_free = kc == FREE_CHUNK_ID
                doc_diff = qc - kc
                win_ok = (doc_diff >= 0) & (doc_diff <= k_win) & (qc >= 0) & (kc >= 0)
                anchor_ok = is_anchor[b, kv_idx] & (kc >= 0)
                return causal & q_not_pad & kv_not_pad & (
                    win_ok | anchor_ok | q_free | kv_free
                )
            return mask_mod

    if name == "random_token":
        if random_keep is None:
            raise ValueError("random_token requires random_keep tensor")

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            qc = chunk_ids[b, q_idx]
            kc = chunk_ids[b, kv_idx]
            q_not_pad = qc != PAD_CHUNK_ID
            kv_not_pad = kc != PAD_CHUNK_ID
            q_free = qc == FREE_CHUNK_ID
            kv_free = kc == FREE_CHUNK_ID
            same_chunk = (qc == kc) & (qc >= 0)
            cross_doc = (qc != kc) & (qc >= 0) & (kc >= 0)
            keep = random_keep[b, q_idx, kv_idx]
            return causal & q_not_pad & kv_not_pad & (
                same_chunk | (cross_doc & keep) | q_free | kv_free
            )
        return mask_mod

    raise ValueError(f"Unknown attention pattern: {name}")


def build_dense_bool_mask(pattern: AttentionPattern, chunk_ids: torch.Tensor,
                          is_anchor: Optional[torch.Tensor] = None,
                          doc_random: Optional[torch.Tensor] = None,
                          random_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Materialize the pattern as a dense (B, S, S) bool mask.

    Used by (a) the SDPA training path, (b) the dense reference for unit
    tests, and (c) eval on backends that can't use FlexAttention.

    Takes the same args as build_flex_mask_mod. Output: True = attend.
    """
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    B, S = chunk_ids.shape
    device = chunk_ids.device

    q = torch.arange(S, device=device)
    kv = torch.arange(S, device=device)
    causal = (q.unsqueeze(1) >= kv.unsqueeze(0))  # (S, S)

    qc = chunk_ids.unsqueeze(2)  # (B, S, 1)
    kc = chunk_ids.unsqueeze(1)  # (B, 1, S)
    q_not_pad = qc != PAD_CHUNK_ID
    kv_not_pad = kc != PAD_CHUNK_ID
    q_free = qc == FREE_CHUNK_ID
    kv_free = kc == FREE_CHUNK_ID
    same_chunk = (qc == kc) & (qc >= 0)

    name = pattern.name
    context_ok = torch.zeros((B, S, S), dtype=torch.bool, device=device)

    if name == "standard":
        # No context-specific rule; everything non-pad in causal is allowed.
        allowed = causal.unsqueeze(0) & q_not_pad & kv_not_pad
        return allowed

    if name == "chunked":
        context_ok = same_chunk
    elif name == "doc_window":
        diff = qc - kc
        context_ok = (diff >= 0) & (diff <= pattern.doc_window_k) & (qc >= 0) & (kc >= 0)
    elif name == "last_token_anchor":
        if is_anchor is None:
            raise ValueError("last_token_anchor requires is_anchor")
        if is_anchor.dim() == 1:
            is_anchor = is_anchor.unsqueeze(0)
        anchor_kv = is_anchor.unsqueeze(1) & (kc >= 0)  # (B, 1, S)
        context_ok = same_chunk | anchor_kv
    elif name == "token_window":
        tok_diff = q.unsqueeze(1) - kv.unsqueeze(0)  # (S, S)
        tok_ok = ((tok_diff >= 0) & (tok_diff <= pattern.token_window_w)).unsqueeze(0)
        context_ok = same_chunk | (tok_ok & (qc >= 0) & (kc >= 0))
    elif name == "bigbird":
        if is_anchor is None:
            raise ValueError("bigbird requires is_anchor")
        if is_anchor.dim() == 1:
            is_anchor = is_anchor.unsqueeze(0)
        diff = qc - kc
        win_ok = (diff >= 0) & (diff <= pattern.doc_window_k) & (qc >= 0) & (kc >= 0)
        anchor_kv = is_anchor.unsqueeze(1) & (kc >= 0)
        context_ok = win_ok | anchor_kv
        if doc_random is not None:
            safe_qc = torch.clamp(qc, min=0)
            safe_kc = torch.clamp(kc, min=0)
            # Gather doc_random[b, qc, kc] densely.
            bidx = torch.arange(B, device=device).view(B, 1, 1).expand(B, S, S)
            rand_ok = doc_random[bidx, safe_qc.expand(B, S, S), safe_kc.expand(B, S, S)]
            rand_ok = rand_ok & (qc >= 0) & (kc >= 0)
            context_ok = context_ok | rand_ok
    elif name == "random_token":
        if random_keep is None:
            raise ValueError("random_token requires random_keep")
        if random_keep.dim() == 2:
            random_keep = random_keep.unsqueeze(0)
        cross_doc = (qc != kc) & (qc >= 0) & (kc >= 0)
        context_ok = same_chunk | (cross_doc & random_keep)
    elif name == "hierarchical_anchor":
        # build_dense_bool_mask is single-layer by contract; for the
        # hierarchical pattern, callers should use
        # build_hierarchical_per_layer_mask (returns one mask per
        # transformer layer). We collapse to "chunked" here so that
        # accidental single-mask use degrades to within-doc only rather
        # than silently behaving like one specific layer's stride.
        context_ok = same_chunk
    else:
        raise ValueError(f"Unknown attention pattern: {name}")

    allowed = causal.unsqueeze(0) & q_not_pad & kv_not_pad & (
        context_ok | q_free | kv_free
    )
    return allowed


# ---------------------------------------------------------------------------
# Hierarchical anchor (per-layer) masks
# ---------------------------------------------------------------------------
#
# Design: each chunk attends (block-style) to itself + the N nearest *strictly
# past* anchor chunks at this layer's stride s_l. Anchor positions are
# multiples of s_l in chunk-index space, so the set of anchors changes with
# the layer. Strides go 1, b, b^2, ... up to >= num_chunks; at stride=1 every
# chunk is an anchor (= dense local window of N+1), at stride=b^k only chunks
# {0, b^k, 2*b^k, ...} are anchors.
#
# Across log_b(num_chunks) stride levels, the cumulative receptive field of
# the rightmost chunk reaches every past chunk. With per-layer cost O((N+1) *
# S * chunk_size) and a hierarchy spread over the model's depth, the total is
# O((N+1) * S * chunk_size * log_b(num_chunks)) — the n log n target.


def compute_layer_strides(
    num_chunks: int, stride_base: int, num_transformer_layers: int,
) -> list[int]:
    """Map num_transformer_layers onto a geometric stride ladder.

    Levels: 1, b, b^2, ... while < num_chunks (so the topmost level always has
    at least 2 anchors, including index 0).

    Schedule: ascending strides assigned via even tiling — layer l ↦ levels[
    (l * K) // L]. Smaller strides land at shallower layers (local mixing
    first), and each level is repeated across multiple layers when L > K.

    If L < K the schedule will skip the deepest levels and the rightmost
    chunk's RF will not be full — caller should warn / increase depth.
    """
    if stride_base < 2:
        raise ValueError(f"stride_base must be >= 2 (got {stride_base})")
    if num_transformer_layers <= 0:
        return []
    levels = []
    s = 1
    while s < max(num_chunks, 2):
        levels.append(s)
        s *= stride_base
    if not levels:
        levels = [1]
    K = len(levels)
    L = num_transformer_layers
    return [levels[(l * K) // L] for l in range(L)]


def hierarchical_layer_context_mask(
    chunk_ids: torch.Tensor, stride: int, num_anchors: int,
) -> torch.Tensor:
    """One layer of the hierarchical_anchor pattern: (B, S, S) bool, True =
    a context-context edge that this layer should keep.

    Composes with the universal causal/free/pad rules in the caller. Captures
    only `same_chunk OR (kc is one of N nearest strictly-past anchors of qc
    at this stride)` — the *block* part of block-hierarchical attention.
    """
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    B, S = chunk_ids.shape

    qc = chunk_ids.unsqueeze(2)  # (B, S, 1)
    kc = chunk_ids.unsqueeze(1)  # (B, 1, S)
    same_chunk = (qc == kc) & (qc >= 0)

    # Negative chunk ids (free / pad) are clamped before modular arithmetic
    # to avoid Python's sign-aware floor-div surprising us. The final
    # `(qc>=0) & (kc>=0)` gate nullifies the fake lookups.
    safe_qc = qc.clamp(min=0)
    safe_kc = kc.clamp(min=0)

    # Largest anchor strictly < qc, in chunk-index space.
    a0 = ((safe_qc - 1).clamp(min=0) // stride) * stride  # (B, S, 1)
    # 0 = nearest past anchor, 1 = second nearest, ...
    rank = (a0 - safe_kc) // stride                       # (B, S, S)

    is_anchor_kc = (safe_kc % stride == 0)
    past_strict = safe_kc < safe_qc
    within_n = (rank >= 0) & (rank < num_anchors)
    hier_ok = (
        is_anchor_kc & past_strict & within_n & (qc >= 0) & (kc >= 0)
    )
    return same_chunk | hier_ok


def build_hierarchical_per_layer_mask(
    pattern: AttentionPattern,
    chunk_ids: torch.Tensor,
    num_transformer_layers: int,
) -> torch.Tensor:
    """Materialize all layers' masks as a (L, B, S, S) bool tensor.

    Memory-heavy for long S × deep models — useful for unit tests and
    diagnostics. Production use should call hierarchical_layer_context_mask
    on demand inside the attention wrapper, indexed by `module.layer_idx`.
    """
    if pattern.name != "hierarchical_anchor":
        raise ValueError(
            f"build_hierarchical_per_layer_mask: pattern.name="
            f"{pattern.name!r}, expected 'hierarchical_anchor'"
        )
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    B, S = chunk_ids.shape
    device = chunk_ids.device

    pos_only = chunk_ids[chunk_ids >= 0]
    num_chunks = (int(pos_only.max().item()) + 1) if pos_only.numel() > 0 else 0
    layer_strides = compute_layer_strides(
        num_chunks=max(num_chunks, 1),
        stride_base=pattern.stride_base,
        num_transformer_layers=num_transformer_layers,
    )

    q = torch.arange(S, device=device)
    kv = torch.arange(S, device=device)
    causal = (q.unsqueeze(1) >= kv.unsqueeze(0))  # (S, S)

    qc = chunk_ids.unsqueeze(2)
    kc = chunk_ids.unsqueeze(1)
    q_not_pad = qc != PAD_CHUNK_ID
    kv_not_pad = kc != PAD_CHUNK_ID
    q_free = qc == FREE_CHUNK_ID
    kv_free = kc == FREE_CHUNK_ID

    out = torch.empty(
        (num_transformer_layers, B, S, S), dtype=torch.bool, device=device,
    )
    for l, stride in enumerate(layer_strides):
        context_ok = hierarchical_layer_context_mask(
            chunk_ids, stride=stride, num_anchors=pattern.num_anchors,
        )
        out[l] = causal.unsqueeze(0) & q_not_pad & kv_not_pad & (
            context_ok | q_free | kv_free
        )
    return out


def build_hierarchical_sdpa_mask(
    chunk_ids: torch.Tensor, stride: int, num_anchors: int, dtype: torch.dtype,
) -> torch.Tensor:
    """One layer's (B, 1, S, S) float mask for the hierarchical_anchor pattern.

    0.0 where attended, finfo(dtype).min where masked. Includes the universal
    causal / pad / free-token rules. Built on demand inside the SDPA wrapper.
    """
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    B, S = chunk_ids.shape
    device = chunk_ids.device

    q = torch.arange(S, device=device)
    kv = torch.arange(S, device=device)
    causal = (q.unsqueeze(1) >= kv.unsqueeze(0))  # (S, S)

    qc = chunk_ids.unsqueeze(2)
    kc = chunk_ids.unsqueeze(1)
    q_not_pad = qc != PAD_CHUNK_ID
    kv_not_pad = kc != PAD_CHUNK_ID
    q_free = qc == FREE_CHUNK_ID
    kv_free = kc == FREE_CHUNK_ID

    context_ok = hierarchical_layer_context_mask(
        chunk_ids, stride=stride, num_anchors=num_anchors,
    )
    allowed = causal.unsqueeze(0) & q_not_pad & kv_not_pad & (
        context_ok | q_free | kv_free
    )

    min_val = torch.finfo(dtype).min
    float_mask = torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    )
    return float_mask.unsqueeze(1)  # (B, 1, S, S)


def install_hierarchical_sdpa_attention(model, state: dict) -> None:
    """Register `sdpa_hierarchical` that overrides the per-layer attention mask.

    Reads from `state` each call:
      - state["chunk_ids"]:     (B, S) int32 — populated per batch by
                                attach_hierarchical_pre_hook.
      - state["layer_strides"]: list[int] of length num_transformer_layers.
      - state["num_anchors"]:   int.

    Picks the stride for the calling layer via `module.layer_idx`, builds the
    layer's (B, 1, S, S) float mask, and forwards to stock SDPA.

    Conflicts with `install_smoothed_attention` because both claim the same
    `_attn_implementation` slot. Caller should assert exclusivity.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_hierarchical(module, query, key, value, attention_mask, **kwargs):
        chunk_ids = state.get("chunk_ids", None)
        layer_strides = state.get("layer_strides", None)
        if chunk_ids is None or layer_strides is None:
            return sdpa_fn(module, query, key, value, attention_mask, **kwargs)

        layer_idx = getattr(module, "layer_idx", 0) or 0
        if layer_idx >= len(layer_strides):
            # Defensive: HF can call with layers outside the planned range
            # (rare, but happens when a model adds extra heads). Fall back to
            # the deepest scheduled stride.
            layer_idx = len(layer_strides) - 1
        stride = int(layer_strides[layer_idx])
        num_anchors = int(state.get("num_anchors", 2))

        per_layer_mask = build_hierarchical_sdpa_mask(
            chunk_ids.to(query.device), stride=stride, num_anchors=num_anchors,
            dtype=query.dtype,
        )
        # HF's sdpa_attention_forward has the rule:
        #   is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
        # When `is_causal=True` is passed in kwargs (the default for decoder
        # attention) AND we hand an explicit 4D mask, this rule still leaves
        # is_causal=False and attn_mask=our_mask — what we want. But on some
        # paths (e.g. when the upstream layer treats sdpa as having implicit
        # causal handling) is_causal is set BEFORE this check and our mask is
        # silently dropped. Force is_causal=False to make sure SDPA uses our
        # explicit mask.
        kwargs["is_causal"] = False
        return sdpa_fn(module, query, key, value, per_layer_mask, **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = sdpa_hierarchical
    model.config._attn_implementation = "sdpa_hierarchical"
    for module in model.modules():
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "sdpa_hierarchical"


def attach_hierarchical_pre_hook(
    model,
    state: dict,
    doc_start_id: int,
    doc_end_id: int,
    pattern: AttentionPattern,
    num_transformer_layers: int,
):
    """Forward pre-hook that rebuilds chunk_ids each batch from input_ids and
    populates the state dict the SDPA wrapper consumes.

    The schedule (layer strides) is recomputed per batch because num_chunks
    can vary across examples; we use the batch-max chunk count.
    """
    free_id = FREE_CHUNK_ID
    pad_id = PAD_CHUNK_ID

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None:
            return
        B, S = input_ids.shape
        device = input_ids.device

        # Default everything to FREE; mark padding from the 2D attention_mask
        # if present, else trust the model to ignore pad positions.
        chunk_ids = torch.full((B, S), free_id, dtype=torch.int32, device=device)
        attn_2d = kwargs.get("attention_mask", None)
        if attn_2d is not None and attn_2d.dim() == 2:
            chunk_ids[~attn_2d.bool()] = pad_id

        # Per-row span scan. CPU-side because find_chunk_spans iterates IDs;
        # cheap relative to the upcoming attention.
        ids_cpu = input_ids.detach().cpu()
        max_chunks = 0
        for b in range(B):
            spans = find_chunk_spans(ids_cpu[b], doc_start_id, doc_end_id)
            for idx, (s, e) in enumerate(spans):
                chunk_ids[b, s:e] = idx
            if len(spans) > max_chunks:
                max_chunks = len(spans)

        layer_strides = compute_layer_strides(
            num_chunks=max(max_chunks, 1),
            stride_base=pattern.stride_base,
            num_transformer_layers=num_transformer_layers,
        )
        state["chunk_ids"] = chunk_ids
        state["layer_strides"] = layer_strides
        state["num_anchors"] = pattern.num_anchors

    return model.register_forward_pre_hook(pre_hook, with_kwargs=True)


# ---------------------------------------------------------------------------
# FlexAttention via state-dict (for hybrid models HF refuses to flex-load)
# ---------------------------------------------------------------------------
#
# The "flex" backend in train_chunked_fast / train_chunked_smoothed loads the
# model with attn_implementation="flex_attention" and passes the BlockMask via
# inputs["attention_mask"]. That works for pure-softmax architectures HF has
# registered (e.g. Llama).
#
# Qwen3.5 is hybrid: 6 softmax layers + 18 GDN linear-attention layers. HF
# refuses attn_implementation="flex_attention" on it (`_supports_flex_attn =
# False`), and even if forced, the linear-attn layers' `_update_linear_attn_mask`
# does `torch.all(attention_mask == 1)` — which crashes on a BlockMask.
#
# This installer sidesteps both issues: register a custom `flex_chunked` attn
# function that reads BlockMask from a shared state dict, and have the trainer
# leave `attention_mask=None` on the model forward. Linear-attn layers see
# None (their normal no-mask path); softmax layers run FlexAttention with the
# block-sparse BlockMask via HF's flex_attention_forward integration.


def install_flex_chunked_attention(model, state: dict) -> None:
    """Register `flex_chunked` attention impl that uses BlockMask from state.

    For models HF doesn't natively register for FlexAttention (notably the
    hybrid Qwen3.5 architecture, where the linear-attn layers' mask routing
    cannot accept a BlockMask through the usual `attention_mask` channel).
    The BlockMask lives in `state["block_mask"]` and is consumed by every
    softmax-attention forward via HF's `flex_attention_forward`; the trainer
    is responsible for not passing any 4D `attention_mask` to model.forward
    so the linear-attn layers stay on their no-mask path.

    Falls back to plain causal SDPA when `state["block_mask"]` is missing,
    so unhooked forward calls (sanity checks, generation) still work.

    Conflicts with `install_smoothed_attention` and
    `install_hierarchical_sdpa_attention` — all three claim the same
    `_attn_implementation` slot. Caller is responsible for the exclusivity
    check.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.flex_attention import flex_attention_forward

    sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def flex_chunked(module, query, key, value, attention_mask, **kwargs):
        block_mask = state.get("block_mask", None)
        if block_mask is None:
            # No mask available — fall back to plain causal SDPA so eval /
            # generation / unhooked forward calls still produce output.
            kwargs.setdefault("is_causal", True)
            return sdpa_fn(module, query, key, value, None, **kwargs)
        # `attention_mask` from the model's prepare-mask pipeline is ignored
        # — we use the BlockMask from state. This bypasses HF's per-layer
        # routing entirely.
        return flex_attention_forward(
            module, query, key, value, block_mask, **kwargs,
        )

    ALL_ATTENTION_FUNCTIONS["flex_chunked"] = flex_chunked
    model.config._attn_implementation = "flex_chunked"
    for module in model.modules():
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "flex_chunked"
