"""FlexAttention backend equivalence tests.

:class:`~olmo_core.nn.attention.backend.FlexAttentionBackend` must reproduce the dense
:class:`~olmo_core.nn.attention.backend.TorchAttentionBackend` masking semantics
``(causal | or_mask) & and_mask`` exactly, for the multimodal masks (bidirectional image
tokens via ``or_mask``; subsegment branch isolation via ``and_mask``). These run on CPU
(FlexAttention executes eagerly without a GPU), so no ``@requires_gpu``.
"""

import pytest
import torch

from olmo_core.nn.attention.backend import FlexAttentionBackend, TorchAttentionBackend


def _make_masks(B: int, S: int, kind: str):
    """Build (or_mask, and_mask) like ``MultimodalLM.forward`` does."""
    or_mask = and_mask = None
    if kind in ("or", "both"):
        is_img = torch.zeros(B, S, dtype=torch.bool)
        is_img[:, 2:6] = True  # a contiguous block of image tokens
        or_mask = (is_img[:, :, None] & is_img[:, None, :]).unsqueeze(1)  # (B,1,S,S)
    if kind in ("and", "both"):
        seg = torch.zeros(B, S, dtype=torch.long)
        seg[:, :4] = 10000  # shared prefix (ATTEND_ALL)
        seg[:, 4:10] = 0  # branch 0
        seg[:, 10:] = 1  # branch 1
        and_mask = (seg[:, :, None] <= seg[:, None, :]).unsqueeze(1)
    return or_mask, and_mask


@pytest.mark.parametrize("n_heads, n_kv_heads", [(4, 4), (4, 2)], ids=["mha", "gqa"])
@pytest.mark.parametrize("kind", ["causal", "or", "and", "both"])
def test_flex_matches_torch_backend(kind, n_heads, n_kv_heads):
    B, S, D = 2, 16, 8
    torch.manual_seed(1)
    q = torch.randn(B, S, n_heads, D)
    k = torch.randn(B, S, n_kv_heads, D)
    v = torch.randn(B, S, n_kv_heads, D)
    or_mask, and_mask = _make_masks(B, S, kind)

    kw = dict(head_dim=D, n_heads=n_heads, n_kv_heads=n_kv_heads, scale=D**-0.5)
    ref = TorchAttentionBackend(**kw)((q, k, v), or_mask=or_mask, and_mask=and_mask)
    out = FlexAttentionBackend(**kw)((q, k, v), or_mask=or_mask, and_mask=and_mask)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_flex_per_token_recovery():
    """The per-token vectors recovered from the masks reproduce the originals."""
    B, S = 2, 16
    or_mask, and_mask = _make_masks(B, S, "both")
    is_image, seg_code, example_id = FlexAttentionBackend._per_token_from_masks(or_mask, and_mask)

    # is_image is exactly the diagonal-implied per-token image flag.
    expected_is_image = torch.zeros(B, S, dtype=torch.bool)
    expected_is_image[:, 2:6] = True
    assert torch.equal(is_image, expected_is_image)

    # seg_code preserves the `seg[q] <= seg[kv]` preorder exactly.
    recovered = (seg_code[:, :, None] <= seg_code[:, None, :]).unsqueeze(1)
    assert torch.equal(recovered, and_mask)

    # A single (unpacked) example -> every token is in the same example.
    assert torch.equal(example_id, torch.zeros(B, S, dtype=torch.int64))


def test_flex_per_token_recovery_packed():
    """With a block-diagonal `and_mask` (two packed examples) the recovered example_id
    labels the two contiguous blocks and seg_code stays example-local."""
    B, S = 1, 12
    # Two packed examples: tokens [0:5] and [5:12], each a single causal segment.
    example = torch.zeros(B, S, dtype=torch.long)
    example[:, 5:] = 1
    same = example[:, :, None] == example[:, None, :]
    # within each example a constant subsegment -> seg rule all-true; AND the block-diagonal.
    and_mask = same.unsqueeze(1)
    _, seg_code, example_id = FlexAttentionBackend._per_token_from_masks(None, and_mask)
    assert torch.equal(example_id, example)
    # the recovered (example_eq & seg_rule) must reproduce the block-diagonal and_mask
    recovered = (
        (example_id[:, :, None] == example_id[:, None, :])
        & (seg_code[:, :, None] <= seg_code[:, None, :])
    ).unsqueeze(1)
    assert torch.equal(recovered, and_mask)


def test_flex_block_mask_reused_across_calls(monkeypatch):
    """The BlockMask is built once for a given (or_mask, and_mask) and reused on subsequent
    calls (all transformer layers + the AC recompute pass the same mask objects), and rebuilt
    when the mask objects change."""
    import torch.nn.attention.flex_attention as fa

    from olmo_core.nn.attention import backend as bk

    bk._flex_block_mask_cache.update(key=None, refs=(None, None), block_mask=None)
    calls = {"n": 0}
    orig = fa.create_block_mask

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(fa, "create_block_mask", counting)

    B, S, D = 2, 16, 8
    q = torch.randn(B, S, 4, D)
    k = torch.randn(B, S, 4, D)
    v = torch.randn(B, S, 4, D)
    om, am = _make_masks(B, S, "both")
    be = FlexAttentionBackend(head_dim=D, n_heads=4, n_kv_heads=4, scale=D**-0.5)
    for _ in range(4):  # same mask objects (like 4 layers) -> built once
        be((q, k, v), or_mask=om, and_mask=am)
    assert calls["n"] == 1
    om2, am2 = _make_masks(B, S, "both")  # new mask objects (next step) -> rebuilt
    be((q, k, v), or_mask=om2, and_mask=am2)
    assert calls["n"] == 2
