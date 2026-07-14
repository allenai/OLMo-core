"""Print per-layer masks and cumulative receptive field for the
`hierarchical_anchor` pattern on a toy example.

Usage:
    python -m scripts.diag.diag_hierarchical_masks                # 8-chunk toy
    python -m scripts.diag.diag_hierarchical_masks --num-chunks 16 --num-anchors 2
    python -m scripts.diag.diag_hierarchical_masks --tokens-per-chunk 2 --include-free

The toy example assigns one or two tokens per chunk so the resulting masks
are small enough to eyeball. Two views are printed:

  1. *Chunk-block* view: collapses each (s, e) chunk span to a single cell;
     rows/cols are chunk indices. Easiest to verify the hierarchical
     pattern's structure directly.
  2. *Token* view (only when --tokens-per-chunk > 1 or --include-free): the
     raw (S, S) mask. Useful to confirm the block invariant (every token in
     an attended chunk is attended to) and that free tokens route correctly.

For each view we print three things:
  - The layer's mask M_l (the edges this layer adds)
  - The cumulative receptive field after layers 0..l (compose: RF_l = M_l @ RF_{l-1})
  - The number of attended (q, k) cells, for compute accounting.
"""

import argparse

import torch

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern,
    FREE_CHUNK_ID,
    build_hierarchical_per_layer_mask,
    compute_layer_strides,
)


def build_toy_chunk_ids(
    num_chunks: int, tokens_per_chunk: int, include_free: bool,
) -> torch.Tensor:
    """Construct a (1, S) chunk_ids tensor.

    Layout (with include_free=True, tokens_per_chunk=2, num_chunks=4):
        [-1, 0, 0, 1, 1, 2, 2, 3, 3, -1]
        ^free   ^chunk 0   ^chunk 1  ...   ^free (answer)
    """
    parts = []
    if include_free:
        parts.append(torch.tensor([FREE_CHUNK_ID], dtype=torch.int32))
    for i in range(num_chunks):
        parts.append(torch.full((tokens_per_chunk,), i, dtype=torch.int32))
    if include_free:
        parts.append(torch.tensor([FREE_CHUNK_ID], dtype=torch.int32))
    return torch.cat(parts, dim=0).unsqueeze(0)


def collapse_to_chunk_view(mask_bs: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
    """Collapse a (S, S) bool mask down to (C, C) at chunk-block granularity.

    Cell (i, j) is True iff *any* token-token edge between chunk i tokens and
    chunk j tokens is attended. With block-style attention, every token in an
    attended chunk pair attends, so "any" matches "all" — this is just a
    visualization shortcut.
    """
    ids = chunk_ids[0]
    chunks = sorted(set(int(v) for v in ids.tolist() if v >= 0))
    out = torch.zeros((len(chunks), len(chunks)), dtype=torch.bool)
    for ci, qc in enumerate(chunks):
        q_mask = (ids == qc)
        for cj, kc in enumerate(chunks):
            k_mask = (ids == kc)
            sub = mask_bs[q_mask][:, k_mask]
            out[ci, cj] = bool(sub.any().item())
    return out


def render_grid(mask: torch.Tensor, header: str = "") -> str:
    """Pretty-print a (R, C) bool mask with row/col indices."""
    R, C = mask.shape
    lines = []
    if header:
        lines.append(header)
    lines.append("       " + "".join(f"{j:>3}" for j in range(C)))
    for i in range(R):
        row = "".join(f"  {'X' if mask[i, j] else '.'}" for j in range(C))
        lines.append(f"  i={i:>2}{row}")
    return "\n".join(lines)


def cumulative_receptive_field(layer_masks: torch.Tensor) -> torch.Tensor:
    """Given (L, R, C) bool, return (L, R, C) where entry [l] is the union
    over paths of length l+1 through M_l ∘ ... ∘ M_0 — i.e., the set of
    keys whose information can reach query q after layers 0..l.

    RF_0 = M_0
    RF_{l+1}(q, k) = OR_{m} (M_{l+1}(q, m) AND RF_l(m, k))
    """
    L, R, C = layer_masks.shape
    rf = torch.zeros_like(layer_masks)
    rf[0] = layer_masks[0]
    for l in range(1, L):
        # Boolean matmul: rf[l] = M_l @ rf[l-1]
        rf[l] = (layer_masks[l].float() @ rf[l - 1].float()) > 0
    return rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-chunks", type=int, default=8)
    ap.add_argument("--num-anchors", type=int, default=2)
    ap.add_argument("--stride-base", type=int, default=2)
    ap.add_argument("--tokens-per-chunk", type=int, default=1)
    ap.add_argument("--include-free", action="store_true",
                    help="Add a leading and trailing FREE token to test "
                         "free-token routing.")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="Number of transformer layers. Defaults to "
                         "ceil(log_b(num_chunks)) — the minimum needed for "
                         "full coverage with one layer per stride level.")
    args = ap.parse_args()

    chunk_ids = build_toy_chunk_ids(
        num_chunks=args.num_chunks,
        tokens_per_chunk=args.tokens_per_chunk,
        include_free=args.include_free,
    )
    pattern = AttentionPattern(
        name="hierarchical_anchor",
        num_anchors=args.num_anchors,
        stride_base=args.stride_base,
    )

    # Default num_layers to one layer per stride level (the minimum hierarchy).
    if args.num_layers is None:
        s, K = 1, 0
        while s < max(args.num_chunks, 2):
            s *= args.stride_base
            K += 1
        args.num_layers = max(K, 1)

    schedule = compute_layer_strides(
        num_chunks=args.num_chunks,
        stride_base=args.stride_base,
        num_transformer_layers=args.num_layers,
    )
    print(f"Pattern: {pattern.tag()}  num_chunks={args.num_chunks} "
          f"tokens_per_chunk={args.tokens_per_chunk} "
          f"free={args.include_free}  num_layers={args.num_layers}")
    print(f"Layer stride schedule: {schedule}")
    print()

    masks = build_hierarchical_per_layer_mask(
        pattern, chunk_ids, num_transformer_layers=args.num_layers,
    )  # (L, 1, S, S)
    chunk_view = torch.stack(
        [collapse_to_chunk_view(masks[l, 0], chunk_ids) for l in range(args.num_layers)]
    )  # (L, C, C)
    rf = cumulative_receptive_field(chunk_view)

    for l in range(args.num_layers):
        s = schedule[l]
        anchor_set = sorted(set(range(0, args.num_chunks, s)))
        edges = int(chunk_view[l].sum().item())
        print(render_grid(
            chunk_view[l],
            header=f"=== Layer {l} (stride={s}, anchors={anchor_set}, "
                   f"chunk-block edges={edges}) ===",
        ))
        print()
        print(render_grid(
            rf[l], header=f"--- Cumulative RF after layers 0..{l} ---",
        ))
        print()

    if args.tokens_per_chunk > 1 or args.include_free:
        print("=" * 60)
        print("Token-level view (raw (S, S) mask):")
        print("=" * 60)
        for l in range(args.num_layers):
            print(render_grid(
                masks[l, 0],
                header=f"--- Layer {l} (stride={schedule[l]}) ---",
            ))
            print()


if __name__ == "__main__":
    main()
