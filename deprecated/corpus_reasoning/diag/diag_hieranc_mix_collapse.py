"""Verify that an all-FREE chunk_ids batch collapses the per-layer
hierarchical mask to plain causal.

This is the load-bearing assumption behind `standard_mix_prob` for the
hierarchical pattern: when the dataset skips wrap_documents() for an
example, the pre-hook produces all-FREE chunk_ids, and we need every
layer's hierarchical mask to come out identical to a plain causal mask
so that example trains under standard attention.
"""

import torch

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern, FREE_CHUNK_ID, build_hierarchical_sdpa_mask,
    compute_layer_strides,
)


def main():
    S = 64
    chunk_ids = torch.full((1, S), FREE_CHUNK_ID, dtype=torch.int32)
    pattern = AttentionPattern(name="hierarchical_anchor", num_anchors=2, stride_base=2)

    # Build masks at every stride level we'd use for n_chunks ~ 20.
    schedule = compute_layer_strides(num_chunks=20, stride_base=2,
                                     num_transformer_layers=36)
    unique_strides = sorted(set(schedule))
    print(f"unique strides in schedule: {unique_strides}")

    dtype = torch.bfloat16
    min_val = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((S, S), min_val, dtype=dtype), diagonal=1,
    )

    for s in unique_strides:
        m = build_hierarchical_sdpa_mask(
            chunk_ids, stride=s, num_anchors=pattern.num_anchors, dtype=dtype,
        )  # (1, 1, S, S)
        m2d = m[0, 0]
        diff = (m2d != causal)
        print(f"stride={s}: differs from causal at {int(diff.sum())} / "
              f"{S * S} cells")
        assert int(diff.sum()) == 0, (
            f"stride={s}: hierarchical mask should collapse to causal when "
            f"all chunk_ids are FREE, but it differs at {int(diff.sum())} cells"
        )

    print("\nALL CHECKS PASSED — all-FREE input collapses to plain causal at every stride")


if __name__ == "__main__":
    main()
