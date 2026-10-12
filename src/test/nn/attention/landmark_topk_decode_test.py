"""Hard top-k landmark block retrieval at decode time (the landmark attention paper's inference
procedure, Mohtashami & Jaggi 2023, section 3.2), enabled via
``GenerationConfig.landmark_top_k_blocks`` -> ``set_landmark_eval_decode(..., top_k=...)``.

These exercise the eager decode math only (no Triton kernel), so they run on CPU. The key behaviors
validated:

* with ``top_k`` set, only the ``top_k`` highest-scoring blocks' content receives nonzero weight --
  every other past block is hard-zeroed (not just down-weighted);
* the retrieved blocks are exactly the argmax of the query-landmark scores;
* attention weights renormalize over the local block plus the retrieved blocks (still sum to one);
* ``top_k >= number of past blocks`` reproduces the dense (soft-gated) decode bit-for-bit;
* the sparse variant retrieves whole chunks (max over a chunk's landmark scores).
"""

import os

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig


def _build(name, *, mem_freq, head_dim, num_landmarks=None):
    kwargs = dict(
        name=name,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    )
    if num_landmarks is not None:
        kwargs["num_landmarks"] = num_landmarks
    attn = AttentionConfig(**kwargs).build(head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


def _decode_probs(attn, q, k, *, qpos, total):
    """Per-position output weights, read off by using one-hot values."""
    v = torch.eye(total).view(1, 1, total, total)
    with torch.no_grad():
        return attn._decode_one(q, k, v, qpos).view(-1)


def _support(probs):
    return sorted(m for m in range(probs.numel()) if probs[m].abs() > 1e-6)


def test_fast_topk_decode_retrieval():
    # FastLandmarkAttention requires mem_freq >= 15 -> block_size = 16; landmarks at j % 16 == 15.
    attn = _build(AttentionType.fast_landmark, mem_freq=15, head_dim=53)
    # Prompt of 3 full blocks (landmarks at 15, 31, 47); generated query at 52.
    total, qpos, P = 53, 52, 48
    landmarks = [15, 31, 47]
    local = list(range(48, 53))

    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)

    # Dense baseline: every block's content is (softly) reachable; landmark values get zero weight.
    attn.set_landmark_eval_decode(P, "extend_last_block")
    dense = _decode_probs(attn, q, k, qpos=qpos, total=total)
    assert _support(dense) == sorted(set(range(total)) - set(landmarks))

    # top_k=1: only the argmax landmark's block content survives; the rest is hard-zeroed.
    scores = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    best = landmarks[int(scores[landmarks].argmax())]
    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=1)
    probs = _decode_probs(attn, q, k, qpos=qpos, total=total)
    retrieved_content = list(range(best - 15, best))  # block content, excluding the landmark
    assert _support(probs) == sorted(retrieved_content + local)
    # Renormalization: weights still sum to one over the restricted support.
    assert abs(float(probs.sum()) - 1.0) < 1e-4

    # top_k=2: the two best blocks.
    order = sorted(landmarks, key=lambda m: -float(scores[m]))
    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=2)
    probs2 = _decode_probs(attn, q, k, qpos=qpos, total=total)
    expected = sorted(local + [m for b in order[:2] for m in range(b - 15, b)])
    assert _support(probs2) == expected

    # top_k >= number of blocks: identical to the dense decode.
    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=3)
    assert torch.equal(_decode_probs(attn, q, k, qpos=qpos, total=total), dense)
    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=100)
    assert torch.equal(_decode_probs(attn, q, k, qpos=qpos, total=total), dense)

    # clear() drops top-k along with eval mode.
    attn.clear_landmark_eval_decode()
    assert attn._eval_top_k is None


def test_fast_prompt_token_decode_per_block_with_topk():
    # The generation loop decodes the *final prompt token* as the first decode step (rather than
    # letting prefill produce the first generated token) so hard top-k retrieval also gates that
    # first token. That query has ``qpos < prompt_len``, so it must (a) use the per-block decode --
    # not the one-long-local-block generated-token rule -- and (b) still apply top-k.
    attn = _build(AttentionType.fast_landmark, mem_freq=15, head_dim=53)

    # (a) Routing: a prompt-position query in an *earlier* block than the prompt boundary would be
    # mangled by the eval-mode rule (its "local block" [section_start, qpos] is empty). With eval
    # mode set it must instead match the cleared per-block decode bit-for-bit.
    total, qpos, P = 53, 20, 40  # query at 20 sits in block [16..31]; prompt boundary at 40
    torch.manual_seed(3)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    attn.clear_landmark_eval_decode()
    per_block = _decode_probs(attn, q, k, qpos=qpos, total=total)
    attn.set_landmark_eval_decode(P, "extend_last_block")
    assert torch.equal(_decode_probs(attn, q, k, qpos=qpos, total=total), per_block)

    # (b) top-k applies to such a prompt-position query. Use the final prompt token (qpos = P-1) with
    # three past blocks; top_k=1 keeps only the argmax block's content (the rest hard-zeroed).
    total, qpos, P = 53, 52, 53
    landmarks = [15, 31, 47]
    local = list(range(48, 53))
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    scores = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    best = landmarks[int(scores[landmarks].argmax())]
    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=1)
    probs = _decode_probs(attn, q, k, qpos=qpos, total=total)
    assert _support(probs) == sorted(list(range(best - 15, best)) + local)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_fast_topk_decode_default_path():
    # top-k also applies to the default (non-eval) per-block decode path.
    attn = _build(AttentionType.fast_landmark, mem_freq=15, head_dim=53)
    total, qpos = 53, 52
    landmarks = [15, 31, 47]

    torch.manual_seed(1)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    scores = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    best = landmarks[int(scores[landmarks].argmax())]

    attn._eval_top_k = 1  # the non-eval path is not reachable via set_landmark_eval_decode
    probs = _decode_probs(attn, q, k, qpos=qpos, total=total)
    # Own (partial) block [48..52] plus the retrieved block's content.
    assert _support(probs) == sorted(list(range(best - 15, best)) + list(range(48, 53)))
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_fast_topk_per_head_selection():
    # The permissive scheme: each head retrieves its own blocks.
    attn = _build(AttentionType.fast_landmark, mem_freq=15, head_dim=53)
    total, qpos, P = 53, 52, 48
    landmarks = torch.tensor([15, 31, 47])

    torch.manual_seed(2)
    H = 4
    q = torch.randn(1, H, 1, total)
    k = torch.randn(1, H, total, total)
    # Force different heads to prefer different landmarks.
    for h in range(H):
        k[0, h, landmarks[h % 3]] = 10.0 * q[0, h, 0]

    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=1)
    v = torch.eye(total).expand(1, H, total, total).contiguous()
    with torch.no_grad():
        probs = attn._decode_one(q, k, v, qpos).view(H, total)
    for h in range(H):
        best = int(landmarks[h % 3])
        support = set(_support(probs[h]))
        # The forced head-specific block is retrieved (its gate is ~1, so its content holds nearly
        # all the mass); the other blocks' content is hard-zeroed.
        assert support & set(range(best - 15, best))
        for other in (int(m) for m in landmarks if int(m) != best):
            assert not (support & set(range(other - 15, other)))


def test_sparse_topk_decode_retrieval():
    os.environ["LM_SPARSE_KERNEL"] = "0"
    # block_size = mem_freq + num_landmarks = 4; landmarks at j % 4 == 3.
    attn = _build(AttentionType.sparse_landmark, mem_freq=3, num_landmarks=1, head_dim=15)
    # Prompt of 3 full chunks (landmarks at 3, 7, 11); generated query at 14.
    total, qpos, P = 15, 14, 12
    landmarks = [3, 7, 11]
    local = [12, 13, 14]

    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)

    # Dense baseline: the sparse mixer attends past chunks via their landmark *values*.
    attn.set_landmark_eval_decode(P, "extend_last_block")
    dense = _decode_probs(attn, q, k, qpos=qpos, total=total)
    assert _support(dense) == sorted(landmarks + local)

    # top_k=1: only the argmax landmark survives.
    scores = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    best = landmarks[int(scores[landmarks].argmax())]
    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=1)
    probs = _decode_probs(attn, q, k, qpos=qpos, total=total)
    assert _support(probs) == sorted([best] + local)
    assert abs(float(probs.sum()) - 1.0) < 1e-4

    # top_k >= number of chunks: identical to dense.
    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=3)
    assert torch.equal(_decode_probs(attn, q, k, qpos=qpos, total=total), dense)


def test_topk_validation():
    attn = _build(AttentionType.fast_landmark, mem_freq=15, head_dim=16)
    with pytest.raises(OLMoConfigurationError, match="top_k"):
        attn.set_landmark_eval_decode(16, "extend_last_block", top_k=0)

    from olmo_core.generate.generation_module import GenerationConfig

    with pytest.raises(ValueError, match="landmark_top_k_blocks"):
        GenerationConfig(pad_token_id=0, eos_token_id=1, landmark_top_k_blocks=0)
