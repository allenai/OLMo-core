import pytest
import torch

from olmo_core.data.document_chunk_landmark import ChunkSegment, emit_document_end_landmark
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.landmark import (
    build_landmark_masks,
    compressive_landmark_grouped_softmax,
)
from olmo_core.nn.attention.landmark_document_end import document_end_compressive_attention
from olmo_core.nn.attention.summary_mask import build_summary_roles


def roles(ids):
    return build_summary_roles(
        torch.tensor([ids]),
        doc_start_id=10,
        doc_end_id=11,
        summary_token_id=12,
        eos_id=13,
        pad_id=14,
    )


def test_emitter():
    ids, mask = emit_document_end_landmark(
        [
            ChunkSegment([1], [False], False),
            ChunkSegment([10, 2, 11], [False] * 3, True),
            ChunkSegment([10, 3, 4, 11], [False] * 4, True),
            ChunkSegment([5], [True], False),
        ],
        mem_id=12,
    )
    assert ids == [1, 10, 2, 11, 12, 10, 3, 4, 11, 12, 5]
    assert mask == [False] * 10 + [True]


def test_periodic_forward_and_gradients():
    torch.manual_seed(42)
    r = roles([10, 1, 11, 12, 10, 2, 11, 12, 3, 4, 5, 6])
    q, k, v = [torch.randn(1, 2, 12, 4, requires_grad=True) for _ in range(3)]
    actual = document_end_compressive_attention(q, k, v, r)
    # Last section is open; periodic mask's final own-landmark key is masked,
    # so use only the first 11 positions to avoid inventing an extra landmark.
    q1, k1, v1 = q[:, :, :11], k[:, :, :11], v[:, :, :11]
    mask, mem, local = build_landmark_masks(11, 4, q.device, q.dtype)
    logits = (q1 @ k1.transpose(-1, -2)) / 2 + mask
    probs = compressive_landmark_grouped_softmax(
        logits, -1, mem.expand(1, 2, 11, 11), local.expand(1, 1, 11, 11)
    )
    expected = probs @ v1
    torch.testing.assert_close(actual[:, :, :11], expected)
    ga = torch.autograd.grad(actual[:, :, :11].square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)


def test_ragged_uniform_scores_and_padding():
    # With zero scores every gate is equal. Doc 1 has 4 values including LM,
    # doc 2 has 5; both receive the same total gate mass at the final query.
    r = roles([1, 10, 2, 11, 12, 10, 3, 4, 11, 12, 5, 14])
    q = k = torch.zeros(1, 1, 12, 1)
    v = torch.arange(12.0).view(1, 1, 12, 1).requires_grad_()
    out = document_end_compressive_attention(q, k, v, r)
    expected = (0 + (1 + 2 + 3 + 4) / 4 + (5 + 6 + 7 + 8 + 9) / 5 + 10) / 4
    torch.testing.assert_close(out[0, 0, 10, 0], torch.tensor(expected))
    assert out[0, 0, 11, 0] == 0
    out.sum().backward()
    assert torch.isfinite(v.grad).all()


def test_packing_isolation_and_causality():
    ids = [1, 10, 2, 11, 12, 5, 13]
    r = roles(ids + ids)
    torch.manual_seed(1)
    q, k, v = [torch.randn(1, 2, 14, 3) for _ in range(3)]
    full = document_end_compressive_attention(q, k, v, r)
    separate = document_end_compressive_attention(q[:, :, 7:], k[:, :, 7:], v[:, :, 7:], roles(ids))
    torch.testing.assert_close(full[:, :, 7:], separate)
    v[:, :, 5:] += 100
    changed = document_end_compressive_attention(q, k, v, r)
    torch.testing.assert_close(full[:, :, :5], changed[:, :, :5])


def test_layer_build_forward_backward_and_missing_roles():
    layer = AttentionConfig(
        name=AttentionType.document_end_compressive_landmark, n_heads=2, n_kv_heads=1, bias=False
    ).build(8, layer_idx=0, n_layers=1)
    x = torch.randn(1, 6, 8, requires_grad=True)
    with pytest.raises(ValueError, match="requires summary_roles"):
        layer(x)
    out = layer(x, summary_roles=roles([1, 10, 2, 11, 12, 5]))
    assert out.shape == x.shape
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_model_role_reconstruction_train_and_eval():
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.feed_forward import FeedForwardConfig

    config = TransformerConfig.llama_like(
        d_model=16,
        vocab_size=32,
        n_layers=1,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=32, bias=False),
    )
    config.block.sequence_mixer.name = AttentionType.document_end_compressive_landmark
    config.document_end_landmark_attention = dict(
        doc_start_id=10,
        doc_end_id=11,
        landmark_token_id=12,
        eos_id=13,
        pad_id=14,
    )
    config = TransformerConfig.from_dict(config.as_config_dict())
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    ids = torch.tensor([[1, 10, 2, 11, 12, 5]])
    train = model(ids)
    train.square().mean().backward()
    model.eval()
    with torch.no_grad():
        evaluation = model(ids)
    torch.testing.assert_close(train, evaluation)


def test_ragged_gradcheck():
    r = roles([10, 11, 12, 10, 1, 11, 12, 2])
    tensors = [torch.randn(1, 1, 8, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    assert torch.autograd.gradcheck(
        lambda q, k, v: document_end_compressive_attention(q, k, v, r),
        tensors,
        fast_mode=True,
    )


@pytest.mark.parametrize(
    "prompt",
    [
        [1, 10, 2, 11, 12, 10, 3, 4, 11, 12, 5],
        [10, 1, 11, 12],
        [1],
    ],
)
def test_cached_decode_matches_full_and_resets(prompt):
    from olmo_core.nn.attention.kv_cache import KVCacheManager

    layer = AttentionConfig(
        name=AttentionType.document_end_compressive_landmark, n_heads=2, n_kv_heads=1, bias=False
    ).build(8, layer_idx=0, n_layers=1)
    count = len(prompt)
    torch.manual_seed(7)
    q = torch.randn(1, count + 3, 2, 4)
    k, v = [torch.randn(1, count + 3, 1, 4) for _ in range(2)]
    # No-document prompts remain instruction roles under the shared builder;
    # this is mathematically identical to direct causal query attention.
    expected = layer._attend(q, k, v, roles(prompt + [6, 7, 8]))
    layer.kv_cache_manager = KVCacheManager(1, count + 3, 1, 4, torch.device("cpu"), q.dtype)
    with torch.no_grad():
        for _ in range(2):
            layer.kv_cache_manager.zero_cache()
            layer._document_roles = roles(prompt)
            prefill = layer._sdpa_cached(q[:, :count], k[:, :count], v[:, :count])
            torch.testing.assert_close(prefill, expected[:, :count])
            layer._document_roles = None
            for i in range(count, count + 3):
                actual = layer._sdpa_cached(q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1])
                torch.testing.assert_close(actual, expected[:, i : i + 1])


@pytest.mark.parametrize(
    "ids",
    [
        [10, 1, 11],
        [10, 1, 11, 12, 12],
        [12],
        [11],
        [10, 10, 11, 12],
        [10, 1, 13],
        [10, 1, 11, 2, 12],
        [14, 1],
    ],
)
def test_invalid_layout(ids):
    from olmo_core.data.document_chunk_landmark import validate_document_end_landmark_layout

    with pytest.raises(ValueError, match="Invalid document-end layout"):
        validate_document_end_landmark_layout(
            ids, doc_start_id=10, doc_end_id=11, landmark_token_id=12, eos_id=13, pad_id=14
        )


def test_valid_packed_layout():
    from olmo_core.data.document_chunk_landmark import validate_document_end_landmark_layout

    validate_document_end_landmark_layout(
        [1, 10, 2, 11, 12, 5, 13, 10, 11, 12, 3, 13, 14],
        doc_start_id=10,
        doc_end_id=11,
        landmark_token_id=12,
        eos_id=13,
        pad_id=14,
    )


def test_full_model_cached_logits_with_rope():
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.feed_forward import FeedForwardConfig

    cfg = TransformerConfig.llama_like(
        d_model=16,
        vocab_size=32,
        n_layers=2,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=32, bias=False),
    )
    cfg.block.sequence_mixer.name = AttentionType.document_end_compressive_landmark
    cfg.document_end_landmark_attention = dict(
        doc_start_id=10, doc_end_id=11, landmark_token_id=12, eos_id=13, pad_id=14
    )
    model = cfg.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    model.eval()
    tokens = torch.tensor([[1, 10, 2, 11, 12, 10, 3, 4, 11, 12, 5, 6, 7]])
    with torch.no_grad():
        expected = model(tokens)
        for block in model.blocks.values():
            block.attention.init_kv_cache_manager(1, tokens.shape[1])
        actual = [model(tokens[:, :11])]
        actual.extend(model(tokens[:, i : i + 1]) for i in range(11, 13))
        torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=2e-6, rtol=2e-5)
