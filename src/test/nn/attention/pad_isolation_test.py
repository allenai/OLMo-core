"""Pad positions must not attend to each other.

The collator marks every pad position with the same ``example_ids`` sentinel (-1). Under
the packed-example rule ``example_id[q] == example_id[kv]`` that made the whole pad tail
one mutually-visible causal segment, so FlexAttention computed it in full -- at the
shipped single-image geometry (~4 examples in a 16,384 slot) that was ~92% of all
computed blocks, spent on tokens whose outputs are discarded.
"""

import torch

from olmo_core.nn.attention.backend import FlexAttentionBackend

_S = 2048
_DEV = torch.device("cpu")


def _example_ids(n_examples: int, tokens_each: int) -> torch.Tensor:
    eid = torch.full((1, _S), -1, dtype=torch.long)
    pos = 0
    for i in range(n_examples):
        eid[0, pos : pos + tokens_each] = i
        pos += tokens_each
    return eid


def _computed_blocks(eid: torch.Tensor) -> int:
    bm = FlexAttentionBackend.build_block_mask_from_vectors(1, _S, _DEV, example_id=eid)
    # Naming is counter-intuitive: `kv_num_blocks` counts *partially* masked blocks (the
    # ones needing mask_mod evaluated per element) and `full_kv_num_blocks` counts fully
    # unmasked ones. Both are computed, so the total is their sum.
    partial = int(bm.kv_num_blocks.sum())
    full = int(bm.full_kv_num_blocks.sum()) if bm.full_kv_num_blocks is not None else 0
    return partial + full


def _mask_mod(eid: torch.Tensor):
    backend = FlexAttentionBackend(head_dim=64, n_heads=4)
    return backend._build_mask_mod(None, None, None, eid)


def test_pad_does_not_attend_to_other_pad():
    eid = _example_ids(2, 256)  # 512 real tokens, 1536 pad
    mod = _mask_mod(eid)
    b, h = torch.tensor(0), torch.tensor(0)
    pad_q, other_pad = torch.tensor(1500), torch.tensor(1000)

    assert not bool(mod(b, h, pad_q, other_pad))
    # ...but the row is not empty: a pad position still sees itself, so softmax is defined.
    assert bool(mod(b, h, pad_q, pad_q))


def test_pad_and_real_stay_mutually_invisible():
    eid = _example_ids(2, 256)
    mod = _mask_mod(eid)
    b, h = torch.tensor(0), torch.tensor(0)
    real_q, pad_kv = torch.tensor(300), torch.tensor(1500)
    pad_q, real_kv = torch.tensor(1500), torch.tensor(300)

    assert not bool(mod(b, h, real_q, pad_kv))
    assert not bool(mod(b, h, pad_q, real_kv))


def test_real_example_attention_is_unchanged():
    """Within an example: causal. Across examples: blocked. Neither involves the pad rule."""
    eid = _example_ids(2, 256)
    mod = _mask_mod(eid)
    b, h = torch.tensor(0), torch.tensor(0)

    # causal inside example 0
    assert bool(mod(b, h, torch.tensor(200), torch.tensor(100)))
    assert not bool(mod(b, h, torch.tensor(100), torch.tensor(200)))
    # example 1 cannot see example 0
    assert not bool(mod(b, h, torch.tensor(300), torch.tensor(100)))
    # causal inside example 1
    assert bool(mod(b, h, torch.tensor(400), torch.tensor(300)))


def test_sparse_pack_costs_far_fewer_blocks_than_it_did():
    """A mostly-empty pack must not cost a full causal triangle over its pad tail."""
    sparse = _computed_blocks(_example_ids(2, 256))  # 512 of 2048 real
    dense = _computed_blocks(_example_ids(8, 256))  # 2048 of 2048 real

    # With the pad tail collapsed to a diagonal, a 25%-full pack is far cheaper than a
    # full one. Before the fix the sparse pack cost *more* blocks than the dense one,
    # because its pad tail was a single large causal segment.
    assert sparse < dense


def test_flex_and_dense_rules_agree():
    """The rule is duplicated in two places; pin that they cannot drift apart.

    ``FlexAttentionBackend._build_mask_mod`` expresses it per (q, kv) pair, while
    ``MultimodalLM.forward`` builds the same thing as a dense ``(B, S, S)`` tensor. Any
    divergence would make the two backends produce different hidden states.
    """
    import torch as t

    t.manual_seed(0)
    for _ in range(50):
        s_len = int(t.randint(8, 40, (1,)))
        eid = t.full((1, s_len), -1, dtype=t.long)
        pos, ex = 0, 0
        while pos < s_len and float(t.rand(1)) < 0.8:
            n = int(t.randint(1, 6, (1,)))
            eid[0, pos : pos + n] = ex
            pos += n
            ex += 1

        # Dense form, as built in MultimodalLM.forward.
        dense = eid[:, :, None] == eid[:, None, :]
        dense &= (eid >= 0)[:, :, None]
        dense.diagonal(dim1=-2, dim2=-1)[:] = True

        # Flex form, evaluated pointwise through the real predicate.
        mod = _mask_mod(eid)
        b = t.tensor(0)
        flex = t.zeros_like(dense)
        for q in range(s_len):
            for kv in range(s_len):
                flex[0, q, kv] = bool(mod(b, b, t.tensor(q), t.tensor(kv)))

        # mask_mod folds in the causal base; the dense tensor is the and_mask only.
        causal = t.tril(t.ones(s_len, s_len, dtype=t.bool))[None]
        assert t.equal(flex, dense & causal)


def test_padding_never_influences_real_positions():
    """Invariant: no pad-row rule can perturb a real position's output.

    This holds structurally -- the collator right-pads, attention is causal, and
    ``same_example`` blocks real<->pad both ways -- so real tokens are a strict prefix that
    never reads a pad. Worth pinning, because a future change that let pad reach real
    would break it silently.

    Note what this does **not** show: because it holds for *any* pad-row rule, it is not
    evidence that this PR's rule specifically is behaviour-preserving. The per-quadrant
    tests above are what pin the rule.
    """
    import torch as t

    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer.config import TransformerConfig

    t.manual_seed(0)
    vocab, seq, real = 256, 64, 20
    lm = TransformerConfig.olmo2_1M(
        vocab_size=vocab, attn_backend=AttentionBackendName("torch")
    ).build(init_device="cpu")
    lm.eval()

    ids = t.randint(3, vocab, (1, seq))
    eid = t.full((1, seq), -1, dtype=t.long)
    eid[0, :12] = 0
    eid[0, 12:real] = 1  # two packed examples, then a 44-token pad tail

    def run(tokens):
        same = eid[:, :, None] == eid[:, None, :]
        same &= (eid >= 0)[:, :, None]
        same.diagonal(dim1=-2, dim2=-1)[:] = True
        with t.no_grad():
            return lm(tokens, and_mask=same.unsqueeze(1))

    scrambled = ids.clone()
    scrambled[0, real:] = t.randint(3, vocab, (seq - real,))
    assert t.equal(run(ids)[:, :real], run(scrambled)[:, :real])
