from unittest import mock

import pytest
import torch

from olmo_core.nn.attention import (
    Attention,
    AttentionBackendName,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention import position_window as pw
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import (
    FLASH_2_MARKS,
    GPU_MARKS,
    requires_flash_attn_2,
    requires_gpu,
)

DEVICES = [
    pytest.param(torch.device("cpu"), id="cpu"),
    pytest.param(torch.device("cuda"), id="cuda", marks=GPU_MARKS),
]


def _naive_reference(q, k, v, pos, window, doc=None):
    """Explicit fp32 softmax over the (causal-by-index, window-by-position, same-doc) mask."""
    B, T, Hq, D = q.shape
    n_rep = Hq // k.shape[2]
    k = k.repeat_interleave(n_rep, dim=2)
    v = v.repeat_interleave(n_rep, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * D**-0.5
    pos_l = pos.tolist()
    doc_l = doc.tolist() if doc is not None else None
    allowed_l = [[[False] * T for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for i in range(T):
            for j in range(i + 1):
                d = pos_l[b][i] - pos_l[b][j]
                ok = 0 <= d < window
                if doc_l is not None:
                    ok = ok and doc_l[b][i] == doc_l[b][j]
                allowed_l[b][i][j] = ok
    allowed = torch.tensor(allowed_l, dtype=torch.bool, device=q.device)
    scores = scores.masked_fill(~allowed[:, None], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())


def _random_compacted_positions(B, T, spread, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(spread, generator=g)[:T].sort().values for _ in range(B)])


def _olmo3_like(window, backend, *, all_swa=False, n_layers=4, dtype=torch.float32, device="cuda"):
    torch.manual_seed(0)
    cfg = TransformerConfig.olmo3_30M(
        vocab_size=256,
        n_layers=n_layers,
        sliding_window=SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=not all_swa,
            pattern=[window] if all_swa else [window, window, window, -1],
        ),
        attn_backend=backend,
    )
    model = cfg.build(init_device=device)
    model.init_weights(device=torch.device(device))
    return model.to(dtype).eval()


# --------------------------------------------------------------------------------------------------
# helpers (CPU, fast)
# --------------------------------------------------------------------------------------------------


def test_needs_position_window_detects_gaps():
    contiguous = torch.arange(10)[None].repeat(2, 1) + torch.tensor([[0], [7]])
    assert not pw.needs_position_window(contiguous)
    gapped = torch.tensor([[0, 1, 2, 9, 10, 11]])
    assert pw.needs_position_window(gapped)
    # packed rows restart per document: contiguous within each document is still contiguous
    packed = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    assert not pw.needs_position_window(packed, cu)
    assert pw.needs_position_window(packed)  # without the doc boundaries the restart is a gap
    assert pw.needs_position_window(torch.tensor([[0, 1, 2, 0, 1, 3, 4]]), cu)
    assert not pw.needs_position_window(None)


def test_bias_path_window():
    pos = torch.tensor([[0, 1, 2, 20, 21]])
    causal = torch.full((5, 5), float("-inf")).triu(1)[None, None]
    out = pw.apply_position_window_to_bias(causal, pos, window=16)
    visible = out[0, 0] > -1e30
    expected = pw.position_window_allowed(pos, 16)[0]
    assert torch.equal(visible, expected)
    assert not visible[3, 0] and visible[3, 3] and visible[4, 3]


# --------------------------------------------------------------------------------------------------
# (c) reference check: random compacted positions vs a naive dense-mask implementation
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("packed", [False, True], ids=["rows", "packed"])
def test_matches_naive_reference(device, packed):
    torch.manual_seed(0)
    B, T, Hq, Hk, D, W = (1 if packed else 2), 300, 4, 2, 32, 40
    pos = _random_compacted_positions(B, T, 4 * T, seed=1)
    cu = doc = None
    if packed:
        # three documents; each restarts its own positions at a random offset
        cu = torch.tensor([0, 90, 210, 300], dtype=torch.int32)
        doc = pw._doc_ids(cu, B, T, "cpu")
        pos = torch.cat(
            [_random_compacted_positions(1, n, 3 * n, seed=s) for s, n in enumerate((90, 120, 90))],
            1,
        )
    q, k, v = (torch.randn(B, T, h, D, device=device) for h in (Hq, Hk, Hk))
    for t in (q, k, v):
        t.requires_grad_(True)
    pos_d = pos.to(device)
    cu_d = cu.to(device) if cu is not None else None
    out = pw.position_window_attention(q, k, v, pos_d, W, cu_doc_lens=cu_d)
    ref = _naive_reference(q, k, v, pos_d, W, doc)
    torch.testing.assert_close(out, ref.to(out.dtype), atol=2e-5, rtol=2e-5)

    g = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), g)
    grads_ref = torch.autograd.grad(ref.to(out.dtype), (q, k, v), g)
    for a, b in zip(grads, grads_ref):
        torch.testing.assert_close(a, b, atol=5e-5, rtol=5e-5)


@requires_gpu
def test_block_mask_cached_per_position_tensor():
    pos = _random_compacted_positions(1, 256, 1024, seed=0).cuda()
    m1 = pw._build_block_mask(pos, 64, None)
    assert pw._build_block_mask(pos, 64, None) is m1
    assert pw._build_block_mask(pos, 32, None) is not m1
    assert pw._build_block_mask(pos.clone(), 64, None) is not m1


# --------------------------------------------------------------------------------------------------
# (b) a key whose ORIGINAL distance >= W is masked even though its compacted distance < W
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_far_key_masked_attention_level(device):
    torch.manual_seed(0)
    attn = Attention(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        window_size=16,
        backend=AttentionBackendName.torch,
        init_device=str(device),
    ).eval()
    pos = torch.tensor([[0, 1, 2, 20, 21, 22, 23, 40]], device=device)
    x = torch.randn(1, 8, 64, device=device)
    with torch.no_grad():
        y = attn(x, position_ids=pos)
        x2 = x.clone()
        x2[0, 0] += 10.0  # perturb compacted index 0 (original position 0)
        y2 = attn(x2, position_ids=pos)
    # positions 1, 2 are within 16 of position 0; 20+ are not (compacted distance 3..6 < 16)
    assert not torch.allclose(y[0, 1:3], y2[0, 1:3])
    torch.testing.assert_close(y[0, 3:], y2[0, 3:], atol=0, rtol=0)
    # and the old index-window path would have leaked
    with mock.patch.object(pw, "needs_position_window", return_value=False):
        with torch.no_grad():
            y_idx = attn(x, position_ids=pos)
            y2_idx = attn(x2, position_ids=pos)
    assert not torch.allclose(y_idx[0, 3:7], y2_idx[0, 3:7])


@requires_gpu
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(AttentionBackendName.torch, id="torch"),
        pytest.param(AttentionBackendName.flash_2, id="flash_2", marks=FLASH_2_MARKS),
    ],
)
def test_far_tokens_invisible_through_all_swa_model(backend):
    """All-SWA olmo3-like model (no full layers): with a gap far beyond n_layers * W, the tokens
    before the gap cannot reach any token after it, although every compacted distance is < W."""
    W, n_layers = 16, 4
    dtype = torch.float32 if backend == AttentionBackendName.torch else torch.bfloat16
    model = _olmo3_like(W, backend, all_swa=True, n_layers=n_layers, dtype=dtype)
    pos = torch.cat([torch.arange(4), 200 + torch.arange(10)])[None].cuda()
    ids = torch.randint(0, 256, (1, 14), device="cuda")
    ids2 = ids.clone()
    ids2[0, :4] = (ids2[0, :4] + 7) % 256
    with torch.no_grad():
        a = model(ids, position_ids=pos)
        b = model(ids2, position_ids=pos)
    torch.testing.assert_close(a[0, 4:], b[0, 4:], atol=0, rtol=0)
    assert not torch.allclose(a[0, :4], b[0, :4])


# --------------------------------------------------------------------------------------------------
# (a) keep-all compaction (contiguous positions) == the normal forward
# --------------------------------------------------------------------------------------------------


@requires_gpu
def test_keep_all_matches_normal_forward_fp32():
    model = _olmo3_like(16, AttentionBackendName.torch)
    ids = torch.randint(0, 256, (2, 96), device="cuda")
    pos = torch.arange(96, device="cuda")[None].repeat(2, 1)
    with torch.no_grad():
        ref = model(ids)
        same = model(ids, position_ids=pos)  # contiguous -> backend path
        # force the position-window kernel on the same contiguous positions
        with mock.patch.object(pw, "needs_position_window", return_value=True):
            forced = model(ids, position_ids=pos)
    assert (same - ref).abs().max().item() <= 1e-5
    assert (forced - ref).abs().max().item() <= 1e-5


@requires_gpu
@requires_flash_attn_2
def test_keep_all_matches_flash_bf16():
    model = _olmo3_like(16, AttentionBackendName.flash_2, dtype=torch.bfloat16)
    ids = torch.randint(0, 256, (2, 300), device="cuda")
    pos = torch.arange(300, device="cuda")[None].repeat(2, 1)
    with torch.no_grad():
        ref = model(ids)
        same = model(ids, position_ids=pos)
        with mock.patch.object(pw, "needs_position_window", return_value=True):
            forced = model(ids, position_ids=pos)
    torch.testing.assert_close(same, ref, atol=0, rtol=0)
    # flex vs flash in bf16: kernel rounding only
    torch.testing.assert_close(forced.float(), ref.float(), atol=5e-2, rtol=5e-2)


@requires_gpu
def test_dropped_model_matches_dense_reference_fp32():
    """Compacted row through the olmo3-like model == the same model with every SWA layer's
    attention replaced by the naive position-window reference."""
    W = 16
    model = _olmo3_like(W, AttentionBackendName.torch)
    T = 64
    pos = _random_compacted_positions(1, T, 4 * T, seed=3).cuda()
    ids = torch.randint(0, 256, (1, T), device="cuda")
    with torch.no_grad():
        out = model(ids, position_ids=pos)

        def ref_attn(q, k, v, position_ids, window, *, cu_doc_lens=None, scale=None):
            return _naive_reference(q, k, v, position_ids, window).to(q.dtype).contiguous()

        with mock.patch.object(pw, "position_window_attention", side_effect=ref_attn) as m:
            ref = model(ids, position_ids=pos)
        assert m.call_count == 3  # the three SWA layers; the full layer keeps its backend
    assert (out - ref).abs().max().item() <= 1e-5


@requires_gpu
def test_compiled_blocks_match_eager():
    """``apply_compile`` (the trainer's ``--compile``): the position-window path traces/breaks
    cleanly inside compiled blocks and gives the eager forward + gradients."""
    W, T = 16, 256
    pos = _random_compacted_positions(1, T, 4 * T, seed=5).cuda()
    ids = torch.randint(0, 256, (1, T), device="cuda")
    outs, grads = [], []
    for compiled in (False, True):
        model = _olmo3_like(W, AttentionBackendName.torch).train()
        if compiled:
            model.apply_compile()
        out = model(ids, position_ids=pos.clone())
        out.float().logsumexp(-1).mean().backward()
        outs.append(out.detach())
        grads.append(model.blocks["0"].attention.w_q.weight.grad.detach().clone())
    torch.testing.assert_close(outs[1], outs[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grads[1], grads[0], atol=1e-4, rtol=1e-4)
