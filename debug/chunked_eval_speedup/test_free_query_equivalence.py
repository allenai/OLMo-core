"""Proof that the FREE-query fast path is exact, not an approximation.

CLAIM. The chunked rule (src/olmo_core/nn/attention/document_chunked.py:9) is

    allowed = causal & not_pad & (context_ok | q_free | kv_free)

so for any query token whose chunk id is FREE, `q_free` short-circuits the parenthesis to True and
the rule is elementwise identical to `causal & not_pad` -- vLLM's default paged-causal mask. On a
step where EVERY query token is FREE, rebuilding the chunked BlockMask therefore cannot change a
single allowed/disallowed decision, and can be skipped.

This checks the claim exhaustively over randomized chunk layouts rather than trusting the algebra,
and separately checks that `_query_positions_all_free` -- the gate that decides whether to take the
fast path -- fires exactly when it should and never on a step carrying a context query.

CPU only, no GPU, no vLLM.
"""

import os
import sys

import numpy as np

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, REPO_SRC)

FREE, PAD = -1, -2


def chunked_allowed(q_chunk, kv_chunk, q_pos, kv_pos):
    """The full chunked rule, elementwise, as a dense boolean matrix."""
    causal = q_pos[:, None] >= kv_pos[None, :]
    not_pad = (q_chunk[:, None] != PAD) & (kv_chunk[None, :] != PAD)
    same = (q_chunk[:, None] == kv_chunk[None, :]) & (q_chunk[:, None] >= 0)
    q_free = q_chunk[:, None] == FREE
    kv_free = kv_chunk[None, :] == FREE
    return causal & not_pad & (same | q_free | kv_free)


def causal_allowed(q_chunk, kv_chunk, q_pos, kv_pos):
    """What vLLM's default mask does: causal, pad-gated. No chunk structure at all."""
    causal = q_pos[:, None] >= kv_pos[None, :]
    not_pad = (q_chunk[:, None] != PAD) & (kv_chunk[None, :] != PAD)
    return causal & not_pad


def random_layout(rng, T, n_docs):
    """A sequence: optional leading FREE instruction, n_docs contiguous documents, trailing FREE."""
    row = np.full(T, FREE, dtype=np.int32)
    lead = rng.integers(0, 40)
    body = T - lead - rng.integers(1, 40)
    if n_docs > 0 and body > lead:
        per = max(1, (body - lead) // n_docs)
        for c in range(n_docs):
            s = lead + c * per
            e = min(s + per - 1, body - 1)
            if s <= e:
                row[s : e + 1] = c
    return row


def main():
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(300):
        T = int(rng.integers(60, 400))
        n_docs = int(rng.integers(0, 12))
        chunk = random_layout(rng, T, n_docs)
        pos = np.arange(T)

        full = chunked_allowed(chunk, chunk, pos, pos)
        caus = causal_allowed(chunk, chunk, pos, pos)

        # (1) For every FREE query ROW, the two masks must agree exactly.
        free_rows = np.flatnonzero(chunk == FREE)
        assert np.array_equal(full[free_rows], caus[free_rows]), (
            f"trial {trial}: chunked != causal on a FREE query row"
        )
        checked += len(free_rows)

        # (2) Sanity in the other direction: on CONTEXT rows the two must genuinely DIFFER
        #     somewhere, otherwise this test would pass vacuously on a degenerate layout.
        ctx_rows = np.flatnonzero(chunk >= 0)
        if len(ctx_rows) and n_docs >= 2:
            assert not np.array_equal(full[ctx_rows], caus[ctx_rows]), (
                f"trial {trial}: chunked == causal on CONTEXT rows -- test is vacuous"
            )

    print(f"(1) chunked == causal on every FREE query row: {checked} rows over 300 layouts OK")
    print("(2) chunked != causal on context rows (test is not vacuous)                    OK")

    # ---- (3) the gate itself -----------------------------------------------------------
    os.environ["CHUNK_FREE_QUERY_FASTPATH"] = "1"
    for name in list(sys.modules):
        if name.endswith("vllm_chunked_patch"):
            del sys.modules[name]
    import importlib

    import torch

    m = importlib.import_module("corpus_reasoning.lib.vllm_chunked_patch")

    class CM:
        def __init__(self, qlens, slens):
            self.num_reqs = len(qlens)
            self.query_start_loc_cpu = torch.tensor(
                np.concatenate([[0], np.cumsum(qlens)]), dtype=torch.int32
            )
            self.seq_lens = torch.tensor(slens, dtype=torch.int32)

    T = 200
    rows = np.stack([random_layout(np.random.default_rng(i), T, 6) for i in range(3)])
    slens = [T, T, T]

    # decode step: 1 query token each, at the last position (trailing FREE) -> gate must fire
    assert m._query_positions_all_free(rows, np.array(slens), CM([1, 1, 1], slens), 3) is True
    print("(3a) gate fires on an all-FREE decode step                                     OK")

    # a step whose query window reaches back into document territory -> gate must NOT fire
    big = int(T - np.flatnonzero(rows[0] >= 0)[0])  # spans the first document
    assert m._query_positions_all_free(rows, np.array(slens), CM([big, 1, 1], slens), 3) is False
    print("(3b) gate refuses a step containing a CONTEXT query                            OK")

    # prefill of the whole sequence -> must NOT fire
    assert m._query_positions_all_free(rows, np.array(slens), CM([T, T, T], slens), 3) is False
    print("(3c) gate refuses a full-sequence prefill step                                 OK")

    print("\nFREE-QUERY FAST PATH IS EXACT")


if __name__ == "__main__":
    main()
