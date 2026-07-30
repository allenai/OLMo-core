"""Render the attention mask of each pattern for a small 3-doc fixture.

Produces one .txt per pattern under examples/attention_patterns/ showing:
  - the input layout (position -> token label + chunk assignment)
  - the dense attention mask as an ASCII grid (rows = Q, cols = KV)
  - a chunk-boundary divider line and a legend

Designed for visual inspection, not for consumption by other scripts.
"""

from pathlib import Path

import torch

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern,
    PAD_CHUNK_ID,
    FREE_CHUNK_ID,
    build_dense_bool_mask,
    build_random_doc_edges,
)

OUT_DIR = Path("examples/attention_patterns")


NUM_DOCS = 7
TOKENS_PER_DOC_BODY = 4  # plus <|doc_start|> and <|doc_end|> → 6 total per doc


def build_fixture():
    """7-doc layout with query before, answer after, and pad tail.

    Each doc is 6 positions: <|doc_start|>, 4 content tokens, <|doc_end|>.
    Totals: 4 query + 7*6 = 42 doc + 3 answer + 2 pad = 51 positions.
    """
    labels = []
    chunk = []

    def add(label, cid, count=1):
        for _ in range(count):
            labels.append(label)
            chunk.append(cid)

    # Query
    for tok in ["Q:", "what", "is", "X?"]:
        add(tok, FREE_CHUNK_ID)

    # Docs
    for d in range(NUM_DOCS):
        add("<|ds|>", d)
        for t in range(TOKENS_PER_DOC_BODY):
            add(f"d{d}_t{t}", d)
        add("<|de|>", d)

    # Answer
    for tok in ["A:", "the", "ans"]:
        add(tok, FREE_CHUNK_ID)

    # Pad
    for _ in range(2):
        add("<pad>", PAD_CHUNK_ID)

    chunk_ids = torch.tensor([chunk], dtype=torch.int32)
    is_anchor = torch.zeros_like(chunk_ids, dtype=torch.bool)
    # <|doc_end|> positions are the anchors.
    for i, lab in enumerate(labels):
        if lab == "<|de|>":
            is_anchor[0, i] = True
    return labels, chunk_ids, is_anchor


def render_layout_header(labels, chunk_ids, is_anchor):
    """Return a string describing position -> (label, chunk, role)."""
    lines = ["Layout:"]
    lines.append(
        "  pos  | token        | chunk | role"
    )
    lines.append("  -----+--------------+-------+-----------------")
    for i, lab in enumerate(labels):
        cid = int(chunk_ids[0, i])
        if cid == FREE_CHUNK_ID:
            role = "FREE (Q/A)"
        elif cid == PAD_CHUNK_ID:
            role = "PAD"
        else:
            role = f"doc{cid}" + (" [ANCHOR]" if bool(is_anchor[0, i]) else "")
        lines.append(f"  {i:>3}  | {lab:<12} | {cid:>5} | {role}")
    return "\n".join(lines)


def render_mask(labels, chunk_ids, mask):
    """ASCII grid with row/col headers showing chunks and tokens.

    Rows = query positions, cols = KV positions.
    ● = attend, · = masked.
    """
    S = mask.shape[-1]

    # Build a short label per column (3 chars wide).
    def short(i):
        lab = labels[i]
        if lab == "<|ds|>":
            return "ds"
        if lab == "<|de|>":
            return "de"
        if lab == "<pad>":
            return "pd"
        if lab in ("Q:", "A:"):
            return lab.rstrip(":")
        # "d3_t2" -> "3t2"
        if len(lab) >= 5 and lab[0] == "d" and lab[2] == "_":
            return lab[1] + "t" + lab[4]
        return (lab + "   ")[:3]

    header_cells = [short(i) for i in range(S)]

    # Column headers: 3 lines of single-character per column to keep width.
    # Instead of 3 rows, just one row with chunk indicator above (F/0/1/2/P).
    def role_char(i):
        cid = int(chunk_ids[0, i])
        if cid == FREE_CHUNK_ID:
            return "F"
        if cid == PAD_CHUNK_ID:
            return "P"
        return str(cid)

    col_role = "      " + " ".join(role_char(i) for i in range(S))
    col_idx = "      " + " ".join(f"{i%10}" for i in range(S))

    lines = [col_role, col_idx]
    lines.append("      " + "-" * (S * 2 - 1))

    for r in range(S):
        row_role = role_char(r)
        row_cells = []
        for c in range(S):
            if mask[0, r, c]:
                row_cells.append("\u25cf")  # ●
            else:
                row_cells.append("\u00b7")  # ·
        lines.append(f" {row_role} {r:>2} |{' '.join(row_cells)}")
    return "\n".join(lines)


def describe_pattern(p: AttentionPattern) -> str:
    if p.name == "standard":
        return "Full causal attention (doc tokens see each other freely)."
    if p.name == "chunked":
        return "Within-doc only. Cross-doc attention is blocked."
    if p.name == "doc_window":
        return (
            f"Doc-neighbor window k={p.doc_window_k}. "
            f"Doc i attends to docs [i-{p.doc_window_k}, i]."
        )
    if p.name == "last_token_anchor":
        return (
            "Within-doc + each doc's <|doc_end|> token is globally attendable "
            "(causal-adapted Longformer-style anchor)."
        )
    if p.name == "token_window":
        return (
            f"Within-doc + raw token-level causal window w={p.token_window_w} "
            "(can cross doc boundaries)."
        )
    if p.name == "bigbird":
        return (
            f"Window k={p.doc_window_k} + doc-end anchors + "
            f"{p.num_random_doc_edges} random earlier-doc edges per doc (seed="
            f"{p.random_seed})."
        )
    return p.name


def write_pattern_file(pattern, labels, chunk_ids, is_anchor, doc_random=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if pattern.needs_anchor_tensor():
        kwargs["is_anchor"] = is_anchor
    if doc_random is not None:
        kwargs["doc_random"] = doc_random

    mask = build_dense_bool_mask(pattern, chunk_ids, **kwargs)

    tag = pattern.tag().replace("-", "_")
    path = OUT_DIR / f"{tag}.txt"

    title = f"Attention pattern: {pattern.name}  (tag: {pattern.tag()})"
    underline = "=" * len(title)

    body = [
        title,
        underline,
        "",
        describe_pattern(pattern),
        "",
        render_layout_header(labels, chunk_ids, is_anchor),
        "",
        "Attention mask (rows = Q, cols = KV; \u25cf = attend, \u00b7 = blocked)",
        "Top row (F/0/1/2/P) = role of that column (FREE/doc0/doc1/doc2/PAD).",
        "",
        render_mask(labels, chunk_ids, mask),
        "",
    ]

    # If the pattern uses random edges, dump the doc-pair adjacency for clarity.
    if doc_random is not None:
        dr = doc_random[0]
        body += [
            "Doc-level random adjacency (doc_random[i, j] = True iff doc i",
            "has a random edge to doc j). Window + anchors are applied on top.",
            "",
            "     " + " ".join(f"d{j}" for j in range(dr.shape[1])),
            "     " + "---" * dr.shape[1],
        ]
        for i in range(dr.shape[0]):
            row = " ".join(" \u25cf" if dr[i, j] else " \u00b7" for j in range(dr.shape[1]))
            body.append(f" d{i} |{row}")
        body.append("")

    path.write_text("\n".join(body))
    print(f"  wrote {path}  ({mask.sum().item()} / {mask.numel()} cells attend)")


def main():
    labels, chunk_ids, is_anchor = build_fixture()

    patterns = [
        AttentionPattern(name="standard"),
        AttentionPattern(name="chunked"),
        AttentionPattern(name="doc_window", doc_window_k=1),
        AttentionPattern(name="doc_window", doc_window_k=2),
        AttentionPattern(name="last_token_anchor"),
        AttentionPattern(name="token_window", token_window_w=3),
        AttentionPattern(name="bigbird", doc_window_k=1, num_random_doc_edges=1),
    ]

    for p in patterns:
        doc_random = None
        if p.needs_random_edges():
            # 3 docs -> build a [3, 3] adjacency. With r=1, each doc picks one
            # random earlier doc.
            adj = build_random_doc_edges(
                num_docs=3, num_edges=p.num_random_doc_edges, seed=p.random_seed,
                max_docs=3,
            )
            doc_random = adj.unsqueeze(0)
        write_pattern_file(p, labels, chunk_ids, is_anchor, doc_random=doc_random)

    print(f"\nWrote {len(patterns)} files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
