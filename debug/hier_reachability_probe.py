"""
Debug: does the document-chunked ``hierarchical_dilated`` mask let cross-document information
actually propagate for a contradiction task?  We build the REAL per-layer allowed mask
(olmo_core.nn.attention.chunked_mask.build_chunked_allowed_mask) on a synthetic N-document
sequence and compose it across the model's depth to measure reachability.

Two things matter for contradiction detection:
  (1) the FREE answer token must see the docs (it attends everything at every layer -> trivial), and
  (2) CONTEXT docs must mix with each other so each doc's representation becomes contradiction-aware.

We report, for the LAST context doc, how many of the earlier docs it can reach (via any path that
respects the per-layer masks) by the final layer -- i.e. how much cross-doc mixing actually happens.

CPU only.  Run:  python debug/hier_reachability_probe.py
"""
from __future__ import annotations
import torch
from olmo_core.nn.attention.chunked_mask import AttentionPattern, build_chunked_allowed_mask, FREE_CHUNK_ID

N_LAYERS = 36           # qwen3_4B depth
DOC_LEN = 4             # tokens per context doc (structure only; reachability is doc-granular)
FREE_TAIL = 4           # query/answer FREE tokens at the end


def make_chunk_ids(num_docs: int) -> torch.Tensor:
    ids = []
    for d in range(num_docs):
        ids += [d] * DOC_LEN
    ids += [FREE_CHUNK_ID] * FREE_TAIL
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, S)


def per_layer_masks(pattern: AttentionPattern, chunk_ids: torch.Tensor):
    """List of (S,S) bool allowed masks, one per layer."""
    return [build_chunked_allowed_mask(pattern, chunk_ids, layer_idx=l)[0] for l in range(N_LAYERS)]


def compose_reachability(masks):
    """Boolean reachability after all layers: edge at step k must use layer-k's mask.
    R = A0; R = (R @ A_k) for k>=1 (a length-(k+1) walk exists)."""
    R = masks[0].clone()
    for A in masks[1:]:
        R = (R.float() @ A.float()) > 0
    return R  # R[i,j] = info at token j can reach token i within depth


def doc_token_ranges(num_docs):
    return {d: range(d * DOC_LEN, (d + 1) * DOC_LEN) for d in range(num_docs)}


def hops_to_reach(masks, src_rows, dst_rows, max_hops):
    """Min #layers for info at dst (key) to reach src (query), context-context only."""
    R = None
    for k in range(max_hops):
        R = masks[k].clone() if R is None else (R.float() @ masks[k].float()) > 0
        if R[src_rows][:, dst_rows].any():
            return k + 1
    return None  # unreachable within max_hops


def report(num_docs: int, pattern: AttentionPattern, label: str):
    chunk_ids = make_chunk_ids(num_docs)
    S = chunk_ids.numel()
    masks = per_layer_masks(pattern, chunk_ids)
    ranges = doc_token_ranges(num_docs)
    last = num_docs - 1
    last_rows = list(ranges[last])

    # reachable-doc COUNT for the last context doc after k layers (bandwidth of cross-doc mixing)
    def reach_count_at(k):
        R = masks[0].clone()
        for A in masks[1:k]:
            R = (R.float() @ A.float()) > 0
        return sum(1 for d in range(last) if R[last_rows][:, list(ranges[d])].any())

    # hop-distance for the FARTHEST context doc (doc 0) to reach the last context doc
    h_far = hops_to_reach(masks, last_rows, list(ranges[0]), N_LAYERS)
    prof = {k: reach_count_at(k) for k in (3, 6, 12, 24, N_LAYERS)}

    print(f"  [{label}]  docs={num_docs}")
    print(f"     hops for farthest doc (0) -> last doc: {h_far}   (full attn = 1; fewer = tighter mixing)")
    print(f"     #earlier docs the last doc has mixed with, by layer k:  "
          + "  ".join(f"L{k}:{v}/{last}" for k, v in prof.items()))


def main():
    torch.manual_seed(0)
    # Q: is it "rotating vs not" or just "stride never gets big enough"?
    # Rotating cycle=L caps max stride at m^(L-1). With m=2, dilation_n=4 (3 back-steps), spanning 100
    # docs needs 3*2^(L-1) >= 99 -> L >= 7. Sweep cycle at 100 docs and watch hops collapse to saturating.
    print("=== CYCLE SWEEP @ 100 docs (m=2, n=4): does a longer cycle == saturating? ===")
    for L in (3, 4, 5, 7, 10):
        report(100, AttentionPattern(name="hierarchical_dilated", dilation_n=4, dilation_m=2, dilation_cycle=L),
               f"rotating cyc{L}  (max stride m^{L-1}={2**(L-1)})")
    report(100, AttentionPattern(name="hierarchical_dilated", dilation_n=4, dilation_m=2, dilation_cycle=64),
           "saturating (cycle>depth)")
    print("=== control: bigger BASE m instead of longer cycle (cyc3, m=8 -> max stride 64) ===")
    report(100, AttentionPattern(name="hierarchical_dilated", dilation_n=4, dilation_m=8, dilation_cycle=3),
           "rotating cyc3 m=8  (max stride 8^2=64)")


if __name__ == "__main__":
    main()
