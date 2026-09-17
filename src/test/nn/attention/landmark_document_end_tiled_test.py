"""Independent reference/finite-difference checks for query-tiled attention."""

import pytest
import torch

from olmo_core.nn.attention.landmark_document_end import document_end_compressive_attention
from olmo_core.nn.attention.landmark_document_end_tiled import (
    tiled_document_end_compressive_attention,
)
from olmo_core.nn.attention.summary_mask import build_summary_roles


def _roles(ids, device="cpu"):
    return build_summary_roles(
        torch.tensor(ids, device=device),
        doc_start_id=10,
        doc_end_id=11,
        summary_token_id=12,
        eos_id=13,
        pad_id=14,
    )


def _inputs(device="cpu", dtype=torch.float64, n_kv_heads=1):
    # Different lengths and document counts across rows; free tokens between
    # documents, packed examples, repeated local document IDs, and a pad-only row.
    r = _roles(
        [
            [1, 10, 2, 11, 12, 3, 10, 4, 5, 11, 12, 6, 7, 13, 14, 14],
            [10, 11, 12, 1, 13, 2, 10, 3, 11, 12, 4, 13, 14, 14, 14, 14],
            [14] * 16,
        ],
        device,
    )
    torch.manual_seed(73)
    q = torch.randn(3, 4, 16, 3, dtype=dtype, device=device, requires_grad=True)
    k, v = [
        torch.randn(3, n_kv_heads, 16, 3, dtype=dtype, device=device, requires_grad=True)
        for _ in range(2)
    ]
    return q, k, v, r


@pytest.mark.parametrize("tile", [1, 3, 8, 64])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_outputs_and_gradients_match_reference(tile, kv_heads):
    q, k, v, r = _inputs(n_kv_heads=kv_heads)
    actual = tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=tile)
    repeat = q.shape[1] // k.shape[1]
    expected = document_end_compressive_attention(
        q,
        k.repeat_interleave(repeat, 1),
        v.repeat_interleave(repeat, 1),
        r,
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    ge = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


def test_gradcheck():
    r = _roles([[1, 10, 2, 11, 12, 10, 11, 12, 3]])
    q = torch.randn(1, 2, 9, 2, dtype=torch.float64, requires_grad=True)
    k, v = [torch.randn(1, 1, 9, 2, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    assert torch.autograd.gradcheck(
        lambda q, k, v: tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=3),
        (q, k, v),
        fast_mode=True,
    )


@pytest.mark.parametrize("start,count", [(6, 3), (15, 1), (0, 1)])
def test_query_offsets_and_noncontiguous_inputs(start, count):
    q, k, v, r = _inputs()
    # Last-dimension stride > 1, as well as an arbitrary query offset.
    q = q.repeat_interleave(2, -1)[..., ::2][:, :, start : start + count]
    k = k.repeat_interleave(2, -1)[..., ::2]
    v = v.repeat_interleave(2, -1)[..., ::2]
    actual = tiled_document_end_compressive_attention(
        q, k, v, r, softmax_scale=0.7, query_start=start, query_tile_size=2
    )
    expected = document_end_compressive_attention(
        q,
        k.repeat_interleave(4, 1),
        v.repeat_interleave(4, 1),
        r,
        softmax_scale=0.7,
        query_start=start,
    )
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(
        expected.square().sum(), (q, k, v), allow_unused=True, materialize_grads=True
    )
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_low_precision_and_extreme_logits(dtype):
    q, k, v, r = _inputs(dtype=dtype)
    q, k = q * 40, k * 40
    actual = tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=3)
    expected = document_end_compressive_attention(
        q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), r
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    grads = torch.autograd.grad(actual.float().sum(), (q, k, v))
    assert all(torch.isfinite(g).all() for g in grads)


def test_only_linear_inputs_saved_and_budget_enforced(monkeypatch):
    from olmo_core.nn.attention import landmark_document_end_tiled as tiled

    q, k, v, r = _inputs()
    saved = []
    tiles = []
    original = tiled._tile_probs

    def record(q, k, *args):
        tiles.append((q.shape[2], k.shape[2]))
        return original(q, k, *args)

    monkeypatch.setattr(tiled, "_tile_probs", record)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        out = tiled_document_end_compressive_attention(
            q, k, v, r, query_tile_size=64, score_budget=3 * 4 * 16 * 2
        )
    assert [t.shape for t in saved] == [q.shape, k.shape, v.shape, r.shape, (3, 16)]
    out.sum().backward()
    assert max(query for query, _ in tiles) == 2
    assert all(query * key * 3 * 4 <= 3 * 4 * 16 * 2 for query, key in tiles)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_forward_backward(dtype):
    q, k, v, r = _inputs(device="cuda", dtype=dtype, n_kv_heads=2)
    actual = tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=5)
    expected = document_end_compressive_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), r
    )
    tol = 0.03 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    ge = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=tol, rtol=tol)


def test_autocast_does_not_lower_accumulator_precision():
    q, k, v, r = _inputs(dtype=torch.float32)
    expected = tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=4)
    grad = torch.randn_like(expected)
    ge = torch.autograd.grad(expected, (q, k, v), grad)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=4)
        ga = torch.autograd.grad(actual, (q, k, v), grad)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


def test_layer_backend_selection_and_gradients():
    from olmo_core.nn.attention import AttentionConfig, AttentionType
    from olmo_core.nn.rope import RoPEConfig
    from olmo_core.exceptions import OLMoConfigurationError

    cfg = AttentionConfig(
        name=AttentionType.document_end_compressive_landmark,
        n_heads=4,
        n_kv_heads=2,
        bias=False,
        rope=RoPEConfig(),
        document_end_query_tile_size=3,
    )
    tiled = cfg.build(16, layer_idx=0, n_layers=1)
    cfg.document_end_backend = "reference"
    reference = cfg.build(16, layer_idx=0, n_layers=1)
    reference.load_state_dict(tiled.state_dict())
    torch.manual_seed(8)
    x = torch.randn(1, 12, 16, requires_grad=True)
    r = _roles([[1, 10, 2, 11, 12, 10, 3, 4, 11, 12, 5, 6]])
    actual, expected = tiled(x, summary_roles=r), reference(x, summary_roles=r)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, e in zip(tiled.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, e.grad, atol=2e-6, rtol=2e-5)
    cfg.document_end_backend = "invalid"
    with pytest.raises(OLMoConfigurationError, match="document_end_backend"):
        cfg.build(16, layer_idx=0, n_layers=1)
    cfg.document_end_backend = "tiled"
    cfg.name = AttentionType.default
    with pytest.raises(OLMoConfigurationError, match="require document-end"):
        cfg.build(16, layer_idx=0, n_layers=1)


def test_no_landmarks_reduces_to_causal_attention():
    import torch.nn.functional as F

    torch.manual_seed(9)
    q, k, v = [torch.randn(1, 2, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    r = _roles([[1, 2, 3, 4, 5, 6, 7]])
    actual = tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=2)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_long_ragged_documents_cross_many_tiles(dtype):
    torch.manual_seed(91)
    tokens = [1, 10] + [2] * 65 + [11, 12, 10] + [3] * 19 + [11, 12] + [4] * 7
    T = len(tokens)
    r = _roles([tokens])
    q = torch.randn(1, 2, T, 8, dtype=dtype, requires_grad=True)
    k, v = [torch.randn(1, 1, T, 8, dtype=dtype, requires_grad=True) for _ in range(2)]
    actual = tiled_document_end_compressive_attention(q, k, v, r, query_tile_size=7)
    expected = document_end_compressive_attention(
        q,
        k.repeat_interleave(2, 1),
        v.repeat_interleave(2, 1),
        r,
    )
    tol = 0.03 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    ge = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=tol, rtol=tol)
