"""Tests for the COMPACTED block-skip path (``enable_block_skip(compact=True)``).

Compaction gathers the kept tokens into a shorter sequence instead of masking them, which is what
turns block skipping from a FLOP-meter saving into a wall-clock one. These tests pin the two
properties that make it safe to use on a plain-attention model: it agrees with the masked path,
and the tokens it skips come out untouched. The GatedDeltaNet caveat (the causal short convolution
sees different neighbours after compaction) is documented in :mod:`olmo_core.nn.block_skip` and is
deliberately NOT asserted here -- these models have no recurrent mixer.
"""

import pytest
import torch

from olmo_core.nn.transformer import TransformerConfig


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = TransformerConfig.llama_like(
        d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, vocab_size=128
    )
    model = cfg.build(init_device="cpu")
    model.init_weights()
    return model


def _forced(model, keep, **fwd):
    """Run ``model`` with every routed block pinned to the same keep mask."""
    holder = model._block_skip["holder"]
    B, T = keep.shape
    orig = holder.begin_forward

    def wrapped(*a, **kw):
        orig(*a, **kw)
        for li in model._block_skip["routed"]:
            holder._choice_cache[(int(li), B, T)] = keep

    holder.begin_forward = wrapped
    try:
        return model(**fwd)
    finally:
        holder.begin_forward = orig


def _keep_mask(B, T, frac):
    keep = torch.zeros(B, T, dtype=torch.bool)
    k = max(1, int(round(frac * T)))
    keep[:, torch.linspace(0, T - 1, k).round().long().unique()] = True
    return keep


@pytest.mark.parametrize("frac", [1.0, 0.5, 0.25])
def test_compact_matches_masked(frac):
    ids = torch.randint(0, 128, (1, 32))
    keep = _keep_mask(1, 32, frac)

    # ``_tiny_model`` is seeded, so the two models are already identical and the routers are
    # deterministically initialised; no state-dict copy is needed (and copying after
    # ``enable_block_skip`` would fail on the router keys the base model does not have).
    masked = _tiny_model()
    masked.enable_block_skip(target=0.5, compact=False)
    compact = _tiny_model()
    compact.enable_block_skip(target=0.5, compact=True)

    with torch.no_grad():
        a = _forced(masked, keep, input_ids=ids)
        b = _forced(compact, keep, input_ids=ids)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_compact_matches_masked_packed():
    """With ``cu_doc_lens`` the compacted path must re-derive both the boundaries and the RoPE
    positions; getting either wrong changes the answer rather than just the speed.

    GPU-only: the compacted path hands intra-document masking to the ordinary attention backend
    (that is where the saving comes from), and ``TorchAttentionBackend`` refuses ``cu_doc_lens``.
    """
    from olmo_core.nn.attention import AttentionBackendName

    # T must be a multiple of the FlexAttention block size (128): the masked reference builds
    # its block mask at that granularity and reshapes by it.
    T, n_docs = 256, 4
    dev = torch.device("cuda")
    ids = torch.randint(0, 128, (1, T), device=dev)
    doc_lens = torch.full((1, n_docs), T // n_docs, dtype=torch.int32, device=dev)
    keep = _keep_mask(1, T, 0.5).to(dev)

    def build():
        torch.manual_seed(0)
        cfg = TransformerConfig.llama_like(
            d_model=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=128,
            attn_backend=AttentionBackendName.flash_2,
        )
        m = cfg.build(init_device="cpu").to(dev)
        m.init_weights(device=dev)
        return m.to(torch.bfloat16)

    masked = build()
    masked.enable_block_skip(target=0.5, compact=False)
    compact = build()
    compact.enable_block_skip(target=0.5, compact=True)

    kw = dict(input_ids=ids, doc_lens=doc_lens, max_doc_lens=[T // n_docs])
    with torch.no_grad():
        a = _forced(masked, keep, **kw)
        b = _forced(compact, keep, **kw)
    # bf16 through three blocks, and the two paths use different attention kernels by design
    torch.testing.assert_close(a.float(), b.float(), atol=6e-2, rtol=6e-2)


def test_compact_leaves_skipped_tokens_untouched():
    """A skipped token's hidden state must pass a routed block completely unchanged.

    Checked on :func:`block_skip_forward` directly rather than through the model, so the decision
    can be pinned without relying on the holder's per-forward cache.
    """
    from olmo_core.nn.block_skip import block_skip_forward

    model = _tiny_model()
    model.enable_block_skip(target=0.5, compact=True)
    block = model.blocks["0"]
    h = torch.randn(1, 32, 64)
    keep = _keep_mask(1, 32, 0.25)
    p = torch.full((1, 32), 0.7)
    with torch.no_grad():
        out = block_skip_forward(block, h, {}, p, keep)
    torch.testing.assert_close(out[0][~keep[0]], h[0][~keep[0]], atol=0.0, rtol=0.0)
    assert not torch.equal(out[0][keep[0]], h[0][keep[0]])


def test_compact_gradients_match_masked():
    ids = torch.randint(0, 128, (1, 24))
    keep = _keep_mask(1, 24, 0.5)
    grads = []
    for compact in (False, True):
        model = _tiny_model()
        model.enable_block_skip(target=0.5, compact=compact)
        _forced(model, keep, input_ids=ids).sum().backward()
        grads.append(
            torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        )
    torch.testing.assert_close(grads[0], grads[1], atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("frac", [0.5, 0.25])
def test_capacity_keeps_exactly_k_and_indices_agree(frac):
    """Fixed-capacity routing must keep exactly ``round(frac*T)`` tokens per block and hand the
    compacted path indices that really are ``keep.nonzero()`` -- the whole point is to use them
    INSTEAD of that call, so a mismatch would silently reorder the sequence."""
    T = 32
    ids = torch.randint(0, 128, (1, T))
    model = _tiny_model()
    model.enable_block_skip(target=frac, compact=True, capacity_frac=frac)
    for r in model.bskip_routers.values():  # break the keep-all init so top-k is non-trivial
        torch.nn.init.normal_(r.w.weight, std=0.5)
    model.train()
    model(ids).sum().backward()

    holder = model._block_skip["holder"]
    expect_k = max(1, int(round(frac * T)))
    assert holder._idx_cache, "capacity mode recorded no indices"
    for key, idx in holder._idx_cache.items():
        keep = holder._choice_cache[key]
        assert idx.shape[-1] == expect_k
        assert int(keep.sum()) == expect_k
        torch.testing.assert_close(idx[0], keep[0].nonzero(as_tuple=True)[0])
