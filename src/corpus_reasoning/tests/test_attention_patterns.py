"""Unit tests for the attention pattern factory in chunked_attention.py.

Each test builds a small hand-constructed chunk_ids tensor, evaluates the
pattern via (a) build_flex_mask_mod on a dense meshgrid and (b) build_dense_bool_mask,
and compares both against a hand-computed reference mask.

Run:
    python -m scripts.tests.test_attention_patterns
"""

import torch

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern,
    PAD_CHUNK_ID,
    FREE_CHUNK_ID,
    build_dense_bool_mask,
    build_flex_mask_mod,
    build_random_doc_edges,
)


def _eval_flex_dense(mask_mod, B, S, device="cpu"):
    """Evaluate a flex mask_mod on a full (B, S, S) meshgrid."""
    out = torch.empty((B, S, S), dtype=torch.bool, device=device)
    q = torch.arange(S, device=device)
    kv = torch.arange(S, device=device)
    Q, KV = torch.meshgrid(q, kv, indexing="ij")
    for b in range(B):
        b_t = torch.full_like(Q, b)
        h_t = torch.zeros_like(Q)
        out[b] = mask_mod(b_t, h_t, Q, KV)
    return out


def _assert_masks_equal(actual, expected, pattern_name, context=""):
    if actual.shape != expected.shape:
        raise AssertionError(f"[{pattern_name}] shape mismatch {actual.shape} vs {expected.shape}")
    diff = actual ^ expected
    if diff.any():
        idxs = torch.nonzero(diff, as_tuple=False)
        sample = idxs[:10].tolist()
        raise AssertionError(
            f"[{pattern_name}] {int(diff.sum())} mismatched entries. First: {sample}. {context}"
        )


def _build_fixture():
    """3 docs of 3 tokens each, flanked by 2-token query and 2-token answer.

    Layout (S=13; we leave it unpadded to S=13 to make the indices readable):
        0 1 | 2 3 4 | 5 6 7 | 8 9 10 | 11 12
       FREE | DOC 0 | DOC 1 | DOC 2  | FREE (answer)
    Anchor positions: 4, 7, 10 (last token of each doc).
    """
    chunk_ids = torch.tensor(
        [[FREE_CHUNK_ID, FREE_CHUNK_ID,
          0, 0, 0,
          1, 1, 1,
          2, 2, 2,
          FREE_CHUNK_ID, FREE_CHUNK_ID]],
        dtype=torch.int32,
    )
    is_anchor = torch.zeros_like(chunk_ids, dtype=torch.bool)
    is_anchor[0, 4] = True
    is_anchor[0, 7] = True
    is_anchor[0, 10] = True
    return chunk_ids, is_anchor


def _expected(chunk_ids, is_anchor, context_fn):
    """Reference mask builder: apply a context_fn(qc, kc, q, kv, anchor_kv) for
    context-context edges, AND in causal + pad + free-rule."""
    B, S = chunk_ids.shape
    out = torch.zeros((B, S, S), dtype=torch.bool)
    for b in range(B):
        for q in range(S):
            for kv in range(S):
                qc = int(chunk_ids[b, q])
                kc = int(chunk_ids[b, kv])
                if q < kv:
                    continue
                if qc == PAD_CHUNK_ID or kc == PAD_CHUNK_ID:
                    continue
                q_free = qc == FREE_CHUNK_ID
                kv_free = kc == FREE_CHUNK_ID
                if q_free or kv_free:
                    out[b, q, kv] = True
                    continue
                # Both doc positions — defer to context_fn.
                anchor_kv = bool(is_anchor[b, kv])
                out[b, q, kv] = context_fn(qc, kc, q, kv, anchor_kv)
    return out


def test_chunked():
    chunk_ids, is_anchor = _build_fixture()
    p = AttentionPattern(name="chunked")
    expected = _expected(chunk_ids, is_anchor, lambda qc, kc, q, k, a: qc == kc)
    dense = build_dense_bool_mask(p, chunk_ids)
    flex = _eval_flex_dense(build_flex_mask_mod(p, chunk_ids), 1, chunk_ids.shape[1])
    _assert_masks_equal(dense, expected, "chunked/dense")
    _assert_masks_equal(flex, expected, "chunked/flex")


def test_doc_window_k1():
    chunk_ids, is_anchor = _build_fixture()
    p = AttentionPattern(name="doc_window", doc_window_k=1)
    expected = _expected(chunk_ids, is_anchor,
                         lambda qc, kc, q, k, a: (qc - kc) in (0, 1))
    dense = build_dense_bool_mask(p, chunk_ids)
    flex = _eval_flex_dense(build_flex_mask_mod(p, chunk_ids), 1, chunk_ids.shape[1])
    _assert_masks_equal(dense, expected, "doc_window_k1/dense")
    _assert_masks_equal(flex, expected, "doc_window_k1/flex")


def test_doc_window_k0_equals_chunked():
    chunk_ids, is_anchor = _build_fixture()
    p = AttentionPattern(name="doc_window", doc_window_k=0)
    chunked = build_dense_bool_mask(AttentionPattern(name="chunked"), chunk_ids)
    dw0 = build_dense_bool_mask(p, chunk_ids)
    _assert_masks_equal(dw0, chunked, "doc_window_k0==chunked")


def test_doc_window_k_infinite_equals_standard_on_doc_region():
    """A very large k should allow all causal cross-doc edges — equivalent to
    standard causal on the doc-token subgrid."""
    chunk_ids, is_anchor = _build_fixture()
    p = AttentionPattern(name="doc_window", doc_window_k=999)
    dw = build_dense_bool_mask(p, chunk_ids)
    std = build_dense_bool_mask(AttentionPattern(name="standard"), chunk_ids)
    # Over the doc-token region only (rows/cols 2..10) they should agree.
    doc_slice = slice(2, 11)
    _assert_masks_equal(dw[:, doc_slice, doc_slice], std[:, doc_slice, doc_slice],
                        "doc_window_k999==standard on doc region")


def test_last_token_anchor():
    chunk_ids, is_anchor = _build_fixture()
    p = AttentionPattern(name="last_token_anchor")
    expected = _expected(
        chunk_ids, is_anchor,
        lambda qc, kc, q, k, a: (qc == kc) or a,
    )
    dense = build_dense_bool_mask(p, chunk_ids, is_anchor=is_anchor)
    flex = _eval_flex_dense(build_flex_mask_mod(p, chunk_ids, is_anchor=is_anchor),
                            1, chunk_ids.shape[1])
    _assert_masks_equal(dense, expected, "anchor/dense")
    _assert_masks_equal(flex, expected, "anchor/flex")


def test_token_window_w2():
    chunk_ids, is_anchor = _build_fixture()
    w = 2
    p = AttentionPattern(name="token_window", token_window_w=w)
    expected = _expected(
        chunk_ids, is_anchor,
        lambda qc, kc, q, k, a: (qc == kc) or ((q - k) <= w),
    )
    dense = build_dense_bool_mask(p, chunk_ids)
    flex = _eval_flex_dense(build_flex_mask_mod(p, chunk_ids), 1, chunk_ids.shape[1])
    _assert_masks_equal(dense, expected, "token_window/dense")
    _assert_masks_equal(flex, expected, "token_window/flex")


def test_standard():
    chunk_ids, is_anchor = _build_fixture()
    p = AttentionPattern(name="standard")
    # Plain causal, only pad is blocked; free/doc are treated the same.
    B, S = chunk_ids.shape
    expected = torch.zeros((B, S, S), dtype=torch.bool)
    for b in range(B):
        for q in range(S):
            for kv in range(S):
                if q < kv:
                    continue
                if chunk_ids[b, q] == PAD_CHUNK_ID or chunk_ids[b, kv] == PAD_CHUNK_ID:
                    continue
                expected[b, q, kv] = True
    dense = build_dense_bool_mask(p, chunk_ids)
    flex = _eval_flex_dense(build_flex_mask_mod(p, chunk_ids), 1, chunk_ids.shape[1])
    _assert_masks_equal(dense, expected, "standard/dense")
    _assert_masks_equal(flex, expected, "standard/flex")


def test_bigbird_window_plus_anchor():
    chunk_ids, is_anchor = _build_fixture()
    p = AttentionPattern(name="bigbird", doc_window_k=1, num_random_doc_edges=0)
    # No random edges → bigbird reduces to window + anchor.
    expected = _expected(
        chunk_ids, is_anchor,
        lambda qc, kc, q, k, a: ((qc - kc) in (0, 1)) or a,
    )
    dense = build_dense_bool_mask(p, chunk_ids, is_anchor=is_anchor)
    flex = _eval_flex_dense(
        build_flex_mask_mod(p, chunk_ids, is_anchor=is_anchor),
        1, chunk_ids.shape[1],
    )
    _assert_masks_equal(dense, expected, "bigbird/no-random/dense")
    _assert_masks_equal(flex, expected, "bigbird/no-random/flex")


def test_bigbird_with_random_edges():
    chunk_ids, is_anchor = _build_fixture()
    # Put an edge from doc 2 to doc 0 manually.
    doc_random = torch.zeros((1, 3, 3), dtype=torch.bool)
    doc_random[0, 2, 0] = True
    p = AttentionPattern(name="bigbird", doc_window_k=1, num_random_doc_edges=1)
    expected = _expected(
        chunk_ids, is_anchor,
        lambda qc, kc, q, k, a: (
            ((qc - kc) in (0, 1)) or a
            or (qc == 2 and kc == 0)
        ),
    )
    dense = build_dense_bool_mask(p, chunk_ids, is_anchor=is_anchor, doc_random=doc_random)
    flex = _eval_flex_dense(
        build_flex_mask_mod(p, chunk_ids, is_anchor=is_anchor, doc_random=doc_random),
        1, chunk_ids.shape[1],
    )
    _assert_masks_equal(dense, expected, "bigbird/random/dense")
    _assert_masks_equal(flex, expected, "bigbird/random/flex")


def test_padding_is_always_blocked():
    """Add pad tokens at the tail and confirm they're unreachable."""
    chunk_ids = torch.tensor(
        [[FREE_CHUNK_ID, 0, 0, 1, 1, FREE_CHUNK_ID,
          PAD_CHUNK_ID, PAD_CHUNK_ID]], dtype=torch.int32,
    )
    is_anchor = torch.zeros_like(chunk_ids, dtype=torch.bool)
    is_anchor[0, 2] = True
    is_anchor[0, 4] = True

    for p in [
        AttentionPattern(name="standard"),
        AttentionPattern(name="chunked"),
        AttentionPattern(name="doc_window", doc_window_k=1),
        AttentionPattern(name="last_token_anchor"),
        AttentionPattern(name="token_window", token_window_w=100),
        AttentionPattern(name="bigbird", doc_window_k=1),
    ]:
        kwargs = {}
        if p.needs_anchor_tensor():
            kwargs["is_anchor"] = is_anchor
        mask = build_dense_bool_mask(p, chunk_ids, **kwargs)
        # Rows for pad positions must be all False.
        assert not mask[0, 6:, :].any(), f"{p.name}: pad row attended"
        # Cols for pad positions must be all False.
        assert not mask[0, :, 6:].any(), f"{p.name}: pad col attended"


def test_qa_tokens_always_see_everything():
    """Core invariant — FREE tokens (Q/A) must attend to and be attended by
    all non-pad positions."""
    chunk_ids, is_anchor = _build_fixture()

    for p in [
        AttentionPattern(name="chunked"),
        AttentionPattern(name="doc_window", doc_window_k=0),
        AttentionPattern(name="doc_window", doc_window_k=1),
        AttentionPattern(name="last_token_anchor"),
        AttentionPattern(name="token_window", token_window_w=0),
        AttentionPattern(name="bigbird", doc_window_k=0),
    ]:
        kwargs = {}
        if p.needs_anchor_tensor():
            kwargs["is_anchor"] = is_anchor
        mask = build_dense_bool_mask(p, chunk_ids, **kwargs)

        # Free rows: can a free query attend causally to every non-pad kv?
        free_positions = (chunk_ids[0] == FREE_CHUNK_ID).nonzero().squeeze(-1).tolist()
        S = chunk_ids.shape[1]
        for q in free_positions:
            for kv in range(q + 1):  # causal
                if chunk_ids[0, kv] == PAD_CHUNK_ID:
                    continue
                assert mask[0, q, kv], (
                    f"{p.name}: free q={q} cannot attend to kv={kv}"
                )
        # Free cols: can every causal query attend to a free kv?
        for kv in free_positions:
            for q in range(kv, S):
                if chunk_ids[0, q] == PAD_CHUNK_ID:
                    continue
                assert mask[0, q, kv], (
                    f"{p.name}: q={q} cannot attend to free kv={kv}"
                )


def test_random_doc_edges_builder():
    """build_random_doc_edges should produce deterministic, diagonal-free,
    lower-triangular-only adjacency."""
    adj = build_random_doc_edges(num_docs=5, num_edges=2, seed=0, max_docs=8)
    assert adj.shape == (8, 8)
    # Must be False on diagonal and upper triangle.
    for i in range(8):
        assert not adj[i, i], "edge includes self-loop"
        for j in range(i + 1, 8):
            assert not adj[i, j], f"edge is future-looking ({i}->{j})"
    # Deterministic seed.
    adj2 = build_random_doc_edges(num_docs=5, num_edges=2, seed=0, max_docs=8)
    assert (adj == adj2).all()
    # Each non-zero row has at most num_edges True, and rows for docs < num_docs
    # are the only ones populated.
    for i in range(5):
        assert int(adj[i].sum()) == min(2, i)
    for i in range(5, 8):
        assert int(adj[i].sum()) == 0


ALL_TESTS = [
    test_chunked,
    test_doc_window_k1,
    test_doc_window_k0_equals_chunked,
    test_doc_window_k_infinite_equals_standard_on_doc_region,
    test_last_token_anchor,
    test_token_window_w2,
    test_standard,
    test_bigbird_window_plus_anchor,
    test_bigbird_with_random_edges,
    test_padding_is_always_blocked,
    test_qa_tokens_always_see_everything,
    test_random_doc_edges_builder,
]


if __name__ == "__main__":
    failures = []
    for t in ALL_TESTS:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures.append(t.__name__)
    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed")
    print(f"\nall {len(ALL_TESTS)} tests passed")
