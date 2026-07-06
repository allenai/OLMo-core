"""Correctness of the "one long local block" decode used for landmark HELMET/RULER-style eval.

These exercise the eager decode math of the landmark mixers (no Triton kernel needed), so they run
on CPU. The key behaviors validated:

* eval-mode decode treats *all* post-prompt positions as one growing local block (no per-block
  rollover), and reaches earlier prompt blocks only through their landmark tokens;
* ``extend_last_block`` vs ``generation_only`` differ exactly in whether the prompt's final partial
  block content is in the local block;
* the default (non-eval) decode is unchanged.
"""

import os

import torch

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


def _support_set(attn, *, qpos, total):
    """Positions that receive nonzero output weight, read off by using one-hot values."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    v = torch.eye(total).view(1, 1, total, total)  # value[m] = one-hot(m)
    with torch.no_grad():
        out = attn._decode_one(q, k, v, qpos).view(-1)
    return sorted(m for m in range(total) if out[m].abs() > 1e-6)


def test_sparse_eval_decode_support_sets():
    os.environ["LM_SPARSE_KERNEL"] = "0"
    # block_size = mem_freq + num_landmarks = 4; landmarks at j % 4 == 3.
    attn = _build(AttentionType.sparse_landmark, mem_freq=3, num_landmarks=1, head_dim=15)
    total, qpos, P = 15, 14, 10  # generated query at 14, prompt boundary at 10

    # extend_last_block: own block = [(10//4)*4 .. 14] = [8..14], plus past landmarks {3, 7}.
    attn.set_landmark_eval_decode(P, "extend_last_block")
    assert _support_set(attn, qpos=qpos, total=total) == sorted({3, 7, 8, 9, 10, 11, 12, 13, 14})

    # generation_only: own block = [10..14], plus past landmarks {3, 7}; the partial prompt block
    # content (8, 9) is not directly attended.
    attn.set_landmark_eval_decode(P, "generation_only")
    assert _support_set(attn, qpos=qpos, total=total) == sorted({3, 7, 10, 11, 12, 13, 14})

    # default (non-eval) decode is unchanged: chunk [12..14] + landmarks {3, 7, 11}.
    attn.clear_landmark_eval_decode()
    assert _support_set(attn, qpos=qpos, total=total) == sorted({3, 7, 11, 12, 13, 14})


def test_fast_eval_decode_one_long_block():
    # FastLandmarkAttention requires mem_freq >= 15 -> block_size = 16; landmark at j % 16 == 15.
    attn = _build(AttentionType.fast_landmark, mem_freq=15, head_dim=40)
    total, qpos, P = 40, 39, 20  # section_start (extend) = (20 // 16) * 16 = 16

    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    v_id = torch.eye(total).view(1, 1, total, total)

    attn.set_landmark_eval_decode(P, "extend_last_block")
    with torch.no_grad():
        probs = attn._decode_one(q, k, v_id, qpos).view(-1)
    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    # The whole growing local block [16..39] is directly attended (no rollover at 32).
    assert all(probs[m].abs() > 1e-6 for m in range(16, 40))
    # The landmark gates block [0..15]: its content (0..14) gets gated weight; the landmark token
    # itself receives no output weight (by design of the grouped softmax).
    assert any(probs[m].abs() > 1e-6 for m in range(15))
    assert probs[15].abs() <= 1e-6

    # eval-mode output differs from the default per-block decode for a cross-boundary query.
    rng = torch.Generator().manual_seed(1)
    v = torch.randn(1, 1, total, total, generator=rng)
    with torch.no_grad():
        attn.set_landmark_eval_decode(P, "extend_last_block")
        o_ext = attn._decode_one(q, k, v, qpos)
        attn.set_landmark_eval_decode(P, "generation_only")
        o_gen = attn._decode_one(q, k, v, qpos)
        attn.clear_landmark_eval_decode()
        o_def = attn._decode_one(q, k, v, qpos)
    assert not torch.allclose(o_ext, o_def, atol=1e-5)
    assert not torch.allclose(o_ext, o_gen, atol=1e-5)
